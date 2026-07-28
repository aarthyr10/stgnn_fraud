"""Error-decorrelated stacking ensemble for the Elliptic prior-shift study.

Five base learners -- three tree models (Random Forest, XGBoost, LightGBM),
one linear model (logistic regression) and the frozen GCN-GRU -- produce
time-blocked probabilities that are combined by a meta-learner fitted on a
held-out validation window.  Four meta-learners are supported (decision
tree, logistic regression, random forest, XGBoost) so the combiner itself
can be ablated.

The module deliberately keeps two split definitions side by side:

``LEGACY_SPLIT``     train 1-34, test 35-49 (the original strict-inductive
                     protocol; there is no validation window, so a stack
                     fitted under it would have to reuse test data and is
                     therefore not offered).
``LEAK_AWARE_SPLIT`` train 1-34, validate 35-41, test 42-49.  The meta-
                     learner is fitted on the validation window only, so it
                     can never inherit base-model training predictions.

Because every Elliptic transaction node occurs in exactly one timestep, a
per-node embedding "sequence" is length one.  ``window`` therefore prepends
``window - 1`` population-mean embeddings from the preceding weeks so the
recurrent head has something temporal to read; ``window=1`` reproduces the
degenerate single-step behaviour of the original pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.data.loader import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN
from app.data.snapshots import Snapshot
from app.models.gcn import StaticGCN
from app.models.gcn_gru import GcnGruHybrid

log = logging.getLogger(__name__)

TREE_LEARNERS = ("random_forest", "xgboost", "lightgbm")
LINEAR_LEARNERS = ("logistic_regression",)
GNN_LEARNERS = ("gcn_gru",)
BASE_LEARNERS = TREE_LEARNERS + LINEAR_LEARNERS + GNN_LEARNERS
META_LEARNERS = ("decision_tree", "logistic_regression",
                 "random_forest", "xgboost")


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitSpec:
    """A temporal split of the 49 Elliptic weeks.

    ``val`` is empty when ``val_end == train_end`` (the legacy protocol).
    """

    name: str
    train_end: int = 34
    val_end: int = 34
    test_end: int = 49
    shutdown: int = 43

    @property
    def train(self) -> List[int]:
        return list(range(1, self.train_end + 1))

    @property
    def val(self) -> List[int]:
        return list(range(self.train_end + 1, self.val_end + 1))

    @property
    def test(self) -> List[int]:
        return list(range(self.val_end + 1, self.test_end + 1))

    @property
    def post_shutdown(self) -> List[int]:
        return [t for t in self.test if t >= self.shutdown]

    @property
    def has_validation(self) -> bool:
        return len(self.val) > 0


LEGACY_SPLIT = SplitSpec("legacy", train_end=34, val_end=34, test_end=49)
LEAK_AWARE_SPLIT = SplitSpec("leak_aware", train_end=34, val_end=41,
                             test_end=49)


# --------------------------------------------------------------------------
# feature assembly
# --------------------------------------------------------------------------
def stack_window(
    snaps: Sequence[Snapshot], timesteps: Sequence[int],
    *, labelled_only: bool = True,
    drop_timestep_column: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate snapshot features/labels/weeks over ``timesteps``."""
    xs, ys, ts = [], [], []
    for t in timesteps:
        s = snaps[t - 1]
        if s.x.size(0) == 0:
            continue
        x = s.x.cpu().numpy()
        y = s.y.cpu().numpy()
        if labelled_only:
            keep = y != LABEL_UNKNOWN
            if not keep.any():
                continue
            x, y = x[keep], y[keep]
        xs.append(x)
        ys.append(y)
        ts.append(np.full(len(y), t, dtype=np.int64))
    if not xs:
        width = snaps[0].x.size(1) if snaps else 0
        return (np.zeros((0, width), dtype=np.float32),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    X = np.concatenate(xs, axis=0)
    if drop_timestep_column:
        X = X[:, 1:]
    return X, np.concatenate(ys), np.concatenate(ts)


# --------------------------------------------------------------------------
# base learners
# --------------------------------------------------------------------------
def _make_tabular_learner(name: str, seed: int, n_jobs: int = -1):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, max_depth=None, n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, n_jobs=n_jobs,
            random_state=seed, eval_metric="logloss", tree_method="hist",
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, n_jobs=n_jobs,
            random_state=seed, verbose=-1,
        )
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, C=1.0, n_jobs=n_jobs)
    raise ValueError(f"unknown tabular learner: {name}")


@dataclass
class TabularBase:
    name: str
    model: object
    scaler: StandardScaler | None
    illicit_col: int
    p_train_illicit: float

    def predict_illicit(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros(0)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        p = self.model.predict_proba(X)
        return p[:, self.illicit_col].astype(np.float64)


def train_tabular_base(
    name: str, snaps: Sequence[Snapshot], split: SplitSpec, seed: int,
    *, drop_timestep_column: bool = False, n_jobs: int = -1,
) -> TabularBase:
    X, y, _ = stack_window(snaps, split.train,
                           drop_timestep_column=drop_timestep_column)
    scaler = None
    if name == "logistic_regression":
        scaler = StandardScaler().fit(X)
        X_fit = scaler.transform(X)
    else:
        X_fit = X
    model = _make_tabular_learner(name, seed, n_jobs=n_jobs)
    model.fit(X_fit, y)
    classes = np.asarray(model.classes_)
    illicit_col = int(np.where(classes == LABEL_ILLICIT)[0][0])
    return TabularBase(
        name=name, model=model, scaler=scaler, illicit_col=illicit_col,
        p_train_illicit=float((y == LABEL_ILLICIT).mean()),
    )


# --------------------------------------------------------------------------
# GCN-GRU base learner (vectorised)
# --------------------------------------------------------------------------
def train_gcn(
    snaps: Sequence[Snapshot], split: SplitSpec, seed: int,
    *, epochs: int = 200, lr: float = 2e-3, weight_decay: float = 5e-4,
) -> StaticGCN:
    torch.manual_seed(seed)
    model = StaticGCN(in_dim=snaps[0].x.size(1))
    train_y = torch.cat([snaps[t - 1].y for t in split.train])
    lab = train_y[train_y != LABEL_UNKNOWN]
    n_il = max(int((lab == LABEL_ILLICIT).sum()), 1)
    n_li = max(int((lab == LABEL_LICIT).sum()), 1)
    total = n_il + n_li
    w = torch.tensor([total / (2.0 * n_li), total / (2.0 * n_il)],
                     dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for t in split.train:
            s = snaps[t - 1]
            if s.x.size(0) == 0:
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(s.x, s.edge_index), s.y, weight=w,
                                   ignore_index=LABEL_UNKNOWN)
            loss.backward()
            opt.step()
            total_loss += float(loss)
        if epoch == 1 or epoch % 25 == 0:
            log.info("  gcn epoch %3d  loss=%.4f", epoch, total_loss)
    model.eval()
    return model


@torch.no_grad()
def embed_all(snaps: Sequence[Snapshot], gcn: StaticGCN,
              ) -> Dict[int, torch.Tensor]:
    gcn.eval()
    return {s.t: (gcn.encode(s.x, s.edge_index) if s.x.size(0) else
                  torch.zeros(0, gcn.embed_dim))
            for s in snaps}


def _population_context(embeds: Dict[int, torch.Tensor],
                        window: int) -> Dict[int, torch.Tensor]:
    """Mean embedding of each week, used as temporal context.

    Elliptic nodes live in a single week, so a per-node sequence has length
    one.  The recurrent head is given the ``window - 1`` preceding weekly
    population means as context before the node's own embedding.
    """
    means: Dict[int, torch.Tensor] = {}
    for t, e in embeds.items():
        means[t] = (e.mean(0) if e.size(0) else
                    torch.zeros(next(iter(embeds.values())).size(1)))
    return means


def build_sequences(embeds: Dict[int, torch.Tensor], t: int, window: int,
                    means: Dict[int, torch.Tensor]) -> torch.Tensor:
    """(n_nodes, window, embed_dim) sequence tensor for week ``t``."""
    own = embeds[t]
    n, d = own.size(0), own.size(1)
    if n == 0:
        return torch.zeros(0, max(window, 1), d)
    if window <= 1:
        return own.unsqueeze(1)
    ctx_ts = [tt for tt in range(t - window + 1, t) if tt in means]
    ctx = ([means[tt] for tt in ctx_ts] or [torch.zeros(d)])
    while len(ctx) < window - 1:
        ctx.insert(0, ctx[0])
    ctx_tensor = torch.stack(ctx, dim=0).unsqueeze(0).expand(n, -1, -1)
    return torch.cat([ctx_tensor, own.unsqueeze(1)], dim=1)


def train_gru_head(
    snaps: Sequence[Snapshot], embeds: Dict[int, torch.Tensor],
    gcn_state: dict, split: SplitSpec, seed: int,
    *, epochs: int = 60, window: int = 1, batch_size: int = 512,
    lr: float = 2e-3, weight_decay: float = 1e-4, in_dim: int = 166,
) -> GcnGruHybrid:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GcnGruHybrid(gcn_weights_path=None)
    model.gcn = StaticGCN(in_dim=in_dim)
    model.gcn.load_state_dict(gcn_state)
    for p in model.gcn.parameters():
        p.requires_grad_(False)
    model.gcn.eval()

    means = _population_context(embeds, window)
    seqs, ys = [], []
    for t in split.train:
        s = snaps[t - 1]
        if s.x.size(0) == 0:
            continue
        keep = (s.y != LABEL_UNKNOWN).numpy()
        if not keep.any():
            continue
        seq = build_sequences(embeds, t, window, means)[torch.from_numpy(keep)]
        seqs.append(seq)
        ys.append(s.y[torch.from_numpy(keep)])
    if not seqs:
        model.eval()
        return model
    X = torch.cat(seqs, dim=0)
    Y = torch.cat(ys, dim=0)

    n_il = max(int((Y == LABEL_ILLICIT).sum()), 1)
    n_li = max(int((Y == LABEL_LICIT).sum()), 1)
    total = n_il + n_li
    w = torch.tensor([total / (2.0 * n_li), total / (2.0 * n_il)],
                     dtype=torch.float32)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(X.size(0))
        running = 0.0
        for start in range(0, X.size(0), batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[idx]), Y[idx], weight=w)
            loss.backward()
            opt.step()
            running += float(loss) * len(idx)
        if epoch == 1 or epoch % 20 == 0:
            log.info("  gru epoch %3d  loss=%.4f", epoch, running / X.size(0))
    model.eval()
    return model


@torch.no_grad()
def gru_illicit_scores(
    model: GcnGruHybrid, embeds: Dict[int, torch.Tensor],
    timesteps: Sequence[int], window: int = 1,
) -> Dict[int, np.ndarray]:
    model.eval()
    means = _population_context(embeds, window)
    out: Dict[int, np.ndarray] = {}
    for t in timesteps:
        seq = build_sequences(embeds, t, window, means)
        if seq.size(0) == 0:
            out[t] = np.zeros(0)
            continue
        chunks = []
        for start in range(0, seq.size(0), 4096):
            logits = model(seq[start:start + 4096])
            chunks.append(F.softmax(logits, dim=-1)[:, LABEL_ILLICIT].numpy())
        out[t] = np.concatenate(chunks).astype(np.float64)
    return out


# --------------------------------------------------------------------------
# meta level
# --------------------------------------------------------------------------
@dataclass
class BaseScores:
    """Per-week illicit probabilities for every base learner."""

    names: List[str]
    scores: Dict[str, Dict[int, np.ndarray]]
    labels: Dict[int, np.ndarray]
    p_train_illicit: float = 0.116
    meta: dict = field(default_factory=dict)

    def matrix(self, timesteps: Sequence[int], *, labelled_only: bool = True,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows, ys, ts = [], [], []
        for t in timesteps:
            y = self.labels[t]
            if y.size == 0:
                continue
            keep = ((y == LABEL_ILLICIT) | (y == LABEL_LICIT)
                    if labelled_only else np.ones(len(y), dtype=bool))
            if not keep.any():
                continue
            cols = [self.scores[n][t][keep] for n in self.names]
            rows.append(np.column_stack(cols))
            ys.append(y[keep])
            ts.append(np.full(int(keep.sum()), t, dtype=np.int64))
        if not rows:
            return (np.zeros((0, len(self.names))), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64))
        return np.concatenate(rows), np.concatenate(ys), np.concatenate(ts)


def _make_meta_learner(name: str, seed: int):
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, min_samples_leaf=20)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                      random_state=seed, min_samples_leaf=5)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4,
                             learning_rate=0.08, n_jobs=-1,
                             random_state=seed, eval_metric="logloss",
                             tree_method="hist")
    raise ValueError(f"unknown meta learner: {name}")


@dataclass
class StackModel:
    name: str
    model: object
    illicit_col: int
    base_names: List[str]

    def predict_illicit_per_t(
        self, base: BaseScores, timesteps: Sequence[int],
    ) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for t in timesteps:
            n = len(base.labels[t])
            if n == 0:
                out[t] = np.zeros(0)
                continue
            X = np.column_stack([base.scores[b][t] for b in self.base_names])
            out[t] = self.model.predict_proba(X)[:, self.illicit_col]
        return out


def fit_stack(
    base: BaseScores, split: SplitSpec, meta_name: str, seed: int,
) -> StackModel:
    if not split.has_validation:
        raise ValueError(
            "fitting a stack needs a validation window; use LEAK_AWARE_SPLIT",
        )
    X, y, _ = base.matrix(split.val)
    model = _make_meta_learner(meta_name, seed)
    model.fit(X, y)
    classes = np.asarray(model.classes_)
    return StackModel(
        name=meta_name, model=model,
        illicit_col=int(np.where(classes == LABEL_ILLICIT)[0][0]),
        base_names=list(base.names),
    )


# --------------------------------------------------------------------------
# error decorrelation
# --------------------------------------------------------------------------
def error_correlation(
    base: BaseScores, timesteps: Sequence[int],
    thresholds: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, List[str], float]:
    """Pairwise correlation of per-node 0/1 error vectors, plus the
    fraction of nodes on which the learners disagree about being wrong."""
    X, y, _ = base.matrix(timesteps)
    if X.shape[0] == 0:
        k = len(base.names)
        return np.full((k, k), np.nan), list(base.names), float("nan")
    errs = []
    for j, name in enumerate(base.names):
        thr = (thresholds or {}).get(name, 0.5)
        pred = (X[:, j] >= thr).astype(np.int64)
        errs.append((pred != y).astype(np.float64))
    E = np.column_stack(errs)
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(E, rowvar=False)
    any_err = E.any(axis=1)
    all_err = E.all(axis=1)
    disagreement = float((any_err & ~all_err).mean())
    return C, list(base.names), disagreement
