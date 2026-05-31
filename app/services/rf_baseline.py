from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.data.loader import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN
from app.data.snapshots import Snapshot, time_split

log = logging.getLogger(__name__)


@dataclass
class RFBundle:
    model: RandomForestClassifier
    p_train_illicit: float

    @property
    def p_train(self) -> np.ndarray:
        return np.array([1.0 - self.p_train_illicit, self.p_train_illicit])


def train_rf_baseline(
    snaps: List[Snapshot],
    *,
    n_estimators: int = 500,
    max_depth: int | None = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> RFBundle:
    train_range, _, _ = time_split()
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for s in snaps:
        if s.t not in train_range or s.x.size(0) == 0:
            continue
        labelled = (s.y != LABEL_UNKNOWN).cpu().numpy()
        if not labelled.any():
            continue
        x_chunks.append(s.x.cpu().numpy()[labelled])
        y_chunks.append(s.y.cpu().numpy()[labelled])
    if not x_chunks:
        log.warning("No labelled training data for RF baseline.")

        clf = RandomForestClassifier(
            n_estimators=5, max_depth=2, random_state=seed,
        )
        clf.fit(np.zeros((4, 166)), np.array([0, 0, 1, 1]))
        return RFBundle(model=clf, p_train_illicit=0.116)

    X = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)

    p_train_illicit = float((y == LABEL_ILLICIT).mean())
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=n_jobs, random_state=seed,
    )
    clf.fit(X, y)
    log.info(
        "RF trained on %d labelled training nodes (p_train_illicit=%.3f)",
        X.shape[0], p_train_illicit,
    )
    return RFBundle(model=clf, p_train_illicit=p_train_illicit)


def predict_rf_per_timestep(
    bundle: RFBundle,
    snaps: List[Snapshot],
    timesteps: List[int] | None = None,
) -> Dict[int, dict]:
    if timesteps is None:
        _, _, test_range = time_split()
        timesteps = list(test_range)

    out: Dict[int, dict] = {}
    classes = bundle.model.classes_

    licit_col = int(np.where(classes == LABEL_LICIT)[0][0])\
        if (classes == LABEL_LICIT).any() else 0
    illicit_col = int(np.where(classes == LABEL_ILLICIT)[0][0])\
        if (classes == LABEL_ILLICIT).any() else 1

    for t in timesteps:
        snap = snaps[t - 1]
        if snap.x.size(0) == 0:
            out[t] = {"p": np.zeros((0, 2)), "y": np.zeros(0, dtype=np.int64),
                      "global_idx": np.zeros(0, dtype=np.int64)}
            continue
        X = snap.x.cpu().numpy()
        raw_p = bundle.model.predict_proba(X)
        p = np.zeros((X.shape[0], 2), dtype=np.float64)
        p[:, LABEL_LICIT] = raw_p[:, licit_col]
        p[:, LABEL_ILLICIT] = raw_p[:, illicit_col]

        p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        out[t] = {
            "p": p,
            "y": snap.y.cpu().numpy(),
            "global_idx": snap.global_idx.cpu().numpy(),
        }
    return out


def save_rf(bundle: RFBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(
            {"model": bundle.model,
             "p_train_illicit": bundle.p_train_illicit},
            fh,
        )


def load_rf(path: str | Path) -> RFBundle:
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    return RFBundle(model=state["model"],
                    p_train_illicit=float(state["p_train_illicit"]))
