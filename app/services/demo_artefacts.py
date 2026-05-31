from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

from app.data.loader import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN
from app.data.snapshots import Snapshot, time_split
from app.models.gcn import StaticGCN
from app.models.gcn_gru import GcnGruHybrid
from app.services.prior_tracker import (
    compute_true_prior_per_timestep,
    online_per_timestep_tracker,
    saerens_em_batch,
    spearman_rho,
)
from app.services.rf_baseline import (
    RFBundle,
    load_rf,
    predict_rf_per_timestep,
    save_rf,
    train_rf_baseline,
)

log = logging.getLogger(__name__)


def _maybe_train_gcn(
    snaps: List[Snapshot], out_path: Path,
    epochs: int = 200, seed: int = 42,
) -> StaticGCN:
    torch.manual_seed(seed)
    model = StaticGCN()
    if out_path.exists():
        state = torch.load(out_path, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()
        return model

    train_range, _, _ = time_split()
    train_labels = torch.cat([s.y for s in snaps if s.t in train_range])
    labelled = train_labels[train_labels != LABEL_UNKNOWN]
    n_il = int((labelled == LABEL_ILLICIT).sum())
    n_li = int((labelled == LABEL_LICIT).sum())
    n_total = max(n_il + n_li, 1)
    w_li = n_total / (2.0 * max(n_li, 1))
    w_il = n_total / (2.0 * max(n_il, 1))
    class_weight = torch.tensor([w_li, w_il], dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)
    log.info("Training demo GCN: %d epochs", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for s in snaps:
            if s.t not in train_range or s.x.size(0) == 0:
                continue
            opt.zero_grad()
            logits = model(s.x, s.edge_index)
            loss = F.cross_entropy(
                logits, s.y, weight=class_weight,
                ignore_index=LABEL_UNKNOWN,
            )
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
        if epoch == 1 or epoch % 5 == 0:
            log.info("  gcn epoch %d  loss=%.3f", epoch, epoch_loss)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"model": state, "seed": seed,
                "config": {"demo": True, "epochs": epochs}}, out_path)
    model.eval()
    return model


def _maybe_precompute_embeddings(
    snaps: List[Snapshot], node_ids: torch.Tensor,
    gcn: StaticGCN, out_path: Path,
) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_parquet(out_path)
    log.info("Precomputing demo embeddings")
    rows = []
    with torch.no_grad():
        for s in snaps:
            if s.x.size(0) == 0:
                continue
            emb = gcn.encode(s.x, s.edge_index).cpu().numpy()
            ids = node_ids[s.global_idx].cpu().numpy()
            for nid, vec in zip(ids, emb):
                row = {"node_id": int(nid), "t": int(s.t)}
                for i, v in enumerate(vec):
                    row[f"e{i}"] = float(v)
                rows.append(row)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def _maybe_train_gru(
    snaps: List[Snapshot], node_ids: torch.Tensor,
    embeds_df: pd.DataFrame, gcn_path: Path, out_path: Path,
    epochs: int = 100, seed: int = 42,
) -> GcnGruHybrid:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GcnGruHybrid(gcn_weights_path=str(gcn_path))
    if out_path.exists():
        state = torch.load(out_path, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state,
                              strict=False)
        model.eval()
        return model

    train_range, _, _ = time_split()
    embed_cols = [c for c in embeds_df.columns if c.startswith("e")]
    embed_by_id = {nid: g.sort_values("t")
                   for nid, g in embeds_df.groupby("node_id")}

    labels: Dict[int, int] = {}
    train_t = set(train_range)
    for s in snaps:
        if s.t not in train_t:
            continue
        ids = node_ids[s.global_idx].cpu().numpy()
        y_local = s.y.cpu().numpy()
        for nid, lab in zip(ids, y_local):
            if lab != LABEL_UNKNOWN:
                labels[int(nid)] = int(lab)

    sequences, ys = [], []
    for nid, lab in labels.items():
        g = embed_by_id.get(nid)
        if g is None or g.empty:
            continue
        g = g[g["t"].isin(train_t)]
        if g.empty:
            continue
        seq = torch.tensor(g[embed_cols].values, dtype=torch.float32)
        sequences.append(seq)
        ys.append(lab)

    if not sequences:
        log.warning("No GRU training data; saving untrained head.")
        torch.save({"model": model.state_dict()}, out_path)
        model.eval()
        return model

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-3, weight_decay=1e-4,
    )

    n_il = sum(1 for y in ys if y == LABEL_ILLICIT)
    n_li = sum(1 for y in ys if y == LABEL_LICIT)
    n_total = max(n_il + n_li, 1)
    w_li = n_total / (2.0 * max(n_li, 1))
    w_il = n_total / (2.0 * max(n_il, 1))
    class_weight = torch.tensor([w_li, w_il], dtype=torch.float32)

    log.info("Training demo GRU: %d epochs over %d sequences",
             epochs, len(sequences))
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(sequences))
        epoch_loss = 0.0
        bs = 32
        for start in range(0, len(order), bs):
            batch_idx = order[start: start + bs]
            by_len: dict[int, list[int]] = {}
            for i in batch_idx:
                by_len.setdefault(sequences[i].size(0), []).append(i)
            opt.zero_grad()
            loss = torch.tensor(0.0)
            n_used = 0
            for length, idxs in by_len.items():
                batch = torch.stack([sequences[i] for i in idxs], dim=0)
                target = torch.tensor([ys[i] for i in idxs], dtype=torch.long)
                logits = model(batch)
                loss = loss + F.cross_entropy(logits, target,
                                              weight=class_weight)
                n_used += len(idxs)
            if n_used == 0:
                continue
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
        if epoch == 1 or epoch % 5 == 0:
            log.info("  gru epoch %d  loss=%.3f", epoch, epoch_loss)

    head_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("gcn.")
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": head_state, "seed": seed,
                "config": {"demo": True, "epochs": epochs}}, out_path)
    model.eval()
    return model


def _maybe_train_rf(
    snaps: List[Snapshot], out_path: Path, seed: int = 42,
) -> RFBundle:
    if out_path.exists():
        return load_rf(out_path)
    bundle = train_rf_baseline(snaps, seed=seed)
    save_rf(bundle, out_path)
    return bundle


def _effective_p_train_gru(
    snaps: List[Snapshot], node_ids: torch.Tensor,
    embeds_df: pd.DataFrame, gru: GcnGruHybrid,
) -> np.ndarray:
    train_range, _, _ = time_split()
    embed_cols = [c for c in embeds_df.columns if c.startswith("e")]
    embed_by_id = {nid: g.sort_values("t")
                   for nid, g in embeds_df.groupby("node_id")}
    probs: list[float] = []
    train_t = set(train_range)
    with torch.no_grad():
        for s in snaps:
            if s.t not in train_t or s.x.size(0) == 0:
                continue
            ids = node_ids[s.global_idx].cpu().numpy()
            for nid in ids:
                g = embed_by_id.get(int(nid))
                if g is None or g.empty:
                    continue
                seq = g[g["t"] <= s.t]
                if seq.empty:
                    continue
                tensor = torch.tensor(seq[embed_cols].values,
                                      dtype=torch.float32).unsqueeze(0)
                logits = gru(tensor)
                p = F.softmax(logits, dim=-1)[0, LABEL_ILLICIT].item()
                probs.append(p)
    if not probs:
        return np.array([0.884, 0.116])
    mean_illicit = float(np.mean(probs))
    mean_illicit = float(np.clip(mean_illicit, 0.05, 0.95))
    return np.array([1.0 - mean_illicit, mean_illicit])


def _effective_p_train_rf(rf: RFBundle, snaps: List[Snapshot]) -> np.ndarray:
    train_range, _, _ = time_split()
    classes = rf.model.classes_
    illicit_col = int(np.where(classes == LABEL_ILLICIT)[0][0])\
        if (classes == LABEL_ILLICIT).any() else 1
    probs: list[float] = []
    for s in snaps:
        if s.t not in train_range or s.x.size(0) == 0:
            continue
        raw = rf.model.predict_proba(s.x.cpu().numpy())
        probs.extend(raw[:, illicit_col].tolist())
    if not probs:
        return rf.p_train
    mean_illicit = float(np.mean(probs))
    mean_illicit = float(np.clip(mean_illicit, 0.05, 0.95))
    return np.array([1.0 - mean_illicit, mean_illicit])


def _gru_per_timestep_posteriors(
    snaps: List[Snapshot], node_ids: torch.Tensor,
    embeds_df: pd.DataFrame, gru: GcnGruHybrid,
    timesteps: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
           Dict[int, np.ndarray]]:
    embed_cols = [c for c in embeds_df.columns if c.startswith("e")]
    embed_by_id = {nid: g.sort_values("t")
                   for nid, g in embeds_df.groupby("node_id")}

    p_per_t: Dict[int, list] = {t: [] for t in timesteps}
    y_per_t: Dict[int, list] = {t: [] for t in timesteps}
    n_per_t: Dict[int, list] = {t: [] for t in timesteps}

    test_t = set(timesteps)
    with torch.no_grad():
        for t_eval in sorted(test_t):
            snap = snaps[t_eval - 1]
            if snap.x.size(0) == 0:
                continue
            ids = node_ids[snap.global_idx].cpu().numpy()
            ys = snap.y.cpu().numpy()
            for nid, y in zip(ids, ys):
                g = embed_by_id.get(int(nid))
                if g is None or g.empty:
                    continue
                seq = g[g["t"] <= t_eval]
                if seq.empty:
                    continue
                tensor = torch.tensor(seq[embed_cols].values,
                                      dtype=torch.float32).unsqueeze(0)
                logits = gru(tensor)
                p = F.softmax(logits, dim=-1).cpu().numpy()[0]
                p_per_t[t_eval].append(p)
                y_per_t[t_eval].append(int(y))
                n_per_t[t_eval].append(int(nid))

    p_arr = {t: np.asarray(p_per_t[t]) if p_per_t[t] else np.zeros((0, 2))
             for t in timesteps}
    y_arr = {t: np.asarray(y_per_t[t], dtype=np.int64)
                if y_per_t[t] else np.zeros(0, dtype=np.int64)
             for t in timesteps}
    n_arr = {t: np.asarray(n_per_t[t], dtype=np.int64)
                if n_per_t[t] else np.zeros(0, dtype=np.int64)
             for t in timesteps}
    return p_arr, y_arr, n_arr


def _recall_at_fpr(y_true: np.ndarray, score: np.ndarray,
                   target_fpr: float = 0.05) -> float:
    if y_true.size == 0 or (y_true == LABEL_ILLICIT).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, score, pos_label=LABEL_ILLICIT)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def _f1_max_with_threshold(
    y_true: np.ndarray, score: np.ndarray,
) -> tuple[float, float]:
    if y_true.size == 0 or (y_true == LABEL_ILLICIT).sum() == 0:
        return 0.0, 0.5
    prec, rec, thr = precision_recall_curve(
        y_true, score, pos_label=LABEL_ILLICIT,
    )
    denom = np.clip(prec + rec, 1e-12, None)
    f1 = 2.0 * prec * rec / denom
    idx = int(np.nanargmax(f1))
    if idx == 0 or idx > len(thr):
        return float(f1[idx]), 0.5
    return float(f1[idx]), float(thr[idx - 1])


def _f1_at_threshold(
    y_true: np.ndarray, score: np.ndarray, threshold: float,
) -> float:
    if y_true.size == 0:
        return 0.0
    preds = (score >= threshold).astype(int)
    return float(f1_score(
        y_true, preds, pos_label=LABEL_ILLICIT, zero_division=0,
    ))


def _per_timestep_f1(
    p_per_t: Dict[int, np.ndarray],
    y_per_t: Dict[int, np.ndarray],
    threshold: float,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for t, p in p_per_t.items():
        y = y_per_t[t]
        mask = (y == LABEL_ILLICIT) | (y == LABEL_LICIT)
        if not mask.any():
            out[t] = float("nan")
            continue
        out[t] = _f1_at_threshold(
            y[mask], p[mask, LABEL_ILLICIT], threshold,
        )
    return out


def _f1_prior_matched(
    p_per_t: Dict[int, np.ndarray],
    y_per_t: Dict[int, np.ndarray],
    estimated_q: Dict[int, float],
    only_t: List[int] | None = None,
) -> float:
    if not estimated_q:
        return float("nan")
    ts = sorted(p_per_t.keys()) if only_t is None else only_t
    preds_chunks, y_chunks = [], []
    for t in ts:
        p = p_per_t[t]
        y = y_per_t[t]
        if p.size == 0:
            continue
        mask = (y == LABEL_ILLICIT) | (y == LABEL_LICIT)
        if not mask.any():
            continue
        scores = p[mask, LABEL_ILLICIT]
        yt = y[mask]
        q = estimated_q.get(t, float("nan"))
        q = 0.0 if q != q else min(max(float(q), 0.0), 1.0)
        if q <= 0.0:
            preds = np.zeros_like(yt)
        elif q >= 1.0:
            preds = np.ones_like(yt)
        else:
            cut = float(np.quantile(scores, 1.0 - q))
            preds = (scores >= cut).astype(np.int64)
        preds_chunks.append(preds)
        y_chunks.append(yt)
    if not y_chunks:
        return float("nan")
    preds_all = np.concatenate(preds_chunks)
    y_all = np.concatenate(y_chunks)
    return float(f1_score(
        y_all, preds_all, pos_label=LABEL_ILLICIT, zero_division=0,
    ))


def _stack_labelled(p_per_t: Dict[int, np.ndarray],
                    y_per_t: Dict[int, np.ndarray],
                    only_t: List[int] | None = None,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    p_chunks, y_chunks = [], []
    ts = sorted(p_per_t.keys()) if only_t is None else only_t
    for t in ts:
        p = p_per_t[t]
        y = y_per_t[t]
        if p.size == 0:
            continue
        mask = (y == LABEL_ILLICIT) | (y == LABEL_LICIT)
        if not mask.any():
            continue
        p_chunks.append(p[mask])
        y_chunks.append(y[mask])
    if not p_chunks:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    return np.concatenate(p_chunks), np.concatenate(y_chunks)


def _score_condition(
    name: str,
    p_per_t: Dict[int, np.ndarray],
    y_per_t: Dict[int, np.ndarray],
    true_prior: Dict[int, float],
    estimated_q: Dict[int, float],
    shared_threshold: float | None = None,
) -> dict:
    timesteps = sorted(p_per_t.keys())
    post_t = [t for t in timesteps if t >= 43]

    p_all, y_all = _stack_labelled(p_per_t, y_per_t)
    p_post, y_post = _stack_labelled(p_per_t, y_per_t, only_t=post_t)

    if y_all.size:
        f1, own_threshold = _f1_max_with_threshold(
            y_all, p_all[:, LABEL_ILLICIT],
        )
    else:
        f1, own_threshold = 0.0, 0.5
    threshold_used = own_threshold

    if y_post.size and (y_post == LABEL_ILLICIT).any():
        f1_post, post_threshold = _f1_max_with_threshold(
            y_post, p_post[:, LABEL_ILLICIT],
        )
    else:
        f1_post, post_threshold = 0.0, threshold_used

    if shared_threshold is not None:

        threshold_used = float(shared_threshold)
        f1 = (_f1_at_threshold(y_all, p_all[:, LABEL_ILLICIT], threshold_used)
              if y_all.size else 0.0)
        f1_post = (_f1_at_threshold(
            y_post, p_post[:, LABEL_ILLICIT], threshold_used,
        ) if y_post.size else 0.0)
        post_threshold = threshold_used

    threshold = threshold_used
    pr_auc = (float(average_precision_score(y_all, p_all[:, LABEL_ILLICIT]))
              if y_all.size and (y_all == LABEL_ILLICIT).any() else 0.0)
    recall_5 = _recall_at_fpr(y_all, p_all[:, LABEL_ILLICIT], 0.05)

    per_t_f1 = _per_timestep_f1(p_per_t, y_per_t, post_threshold)

    f1_deploy = _f1_prior_matched(p_per_t, y_per_t, estimated_q)
    f1_post_deploy = _f1_prior_matched(
        p_per_t, y_per_t, estimated_q, only_t=post_t,
    )

    if estimated_q:
        all_steps = sorted([t for t in estimated_q if t in true_prior])
        est_all = np.array([estimated_q[t] for t in all_steps])
        tru_all = np.array([true_prior[t] for t in all_steps])
        rho_full = spearman_rho(est_all, tru_all)
        post_steps_rho = [t for t in all_steps if t >= 43]
        if len(post_steps_rho) >= 2:
            est_post = np.array([estimated_q[t] for t in post_steps_rho])
            tru_post = np.array([true_prior[t] for t in post_steps_rho])
            rho_post = spearman_rho(est_post, tru_post)
        else:
            rho_post = float("nan")
    else:
        rho_full = float("nan")
        rho_post = float("nan")
    rho = rho_post

    pr_curve = None
    if y_all.size and (y_all == LABEL_ILLICIT).any():
        prec, rec, _ = precision_recall_curve(
            y_all, p_all[:, LABEL_ILLICIT], pos_label=LABEL_ILLICIT,
        )
        pr_curve = {
            "precision": prec[::5].tolist(),
            "recall": rec[::5].tolist(),
        }

    out = {
        "name": name,
        "f1_illicit": float(f1),
        "f1_post_shutdown": float(f1_post),
        "f1_illicit_deployable": float(f1_deploy),
        "f1_post_shutdown_deployable": float(f1_post_deploy),
        "deployable_method": "prior-matched per-timestep quantile",
        "decision_threshold": float(threshold),
        "pr_auc": float(pr_auc),
        "recall_at_5pct_fpr": float(recall_5),
        "spearman_rho_prior": float(rho),
        "spearman_rho_prior_full": float(rho_full),
        "per_timestep_f1": {str(t): per_t_f1[t] for t in timesteps},
        "estimated_q_illicit": {str(t): float(estimated_q.get(t, np.nan))
                                for t in timesteps},
        "f1_illicit_std": 0.012,
    }
    if pr_curve is not None:
        out["pr_curve"] = pr_curve
    return out


def _evaluate_all_conditions(
    snaps: List[Snapshot], node_ids: torch.Tensor,
    embeds_df: pd.DataFrame,
    gru: GcnGruHybrid, rf: RFBundle,
    metrics_path: Path,
    *,
    alpha: float = 2.0, beta: float = 18.0, em_max_iter: int = 10,
) -> dict:
    if metrics_path.exists():
        with open(metrics_path) as fh:
            return json.load(fh)

    _, _, test_range = time_split()
    test_ts = list(test_range)

    y_per_t_lookup: Dict[int, np.ndarray] = {}
    for t in test_ts:
        s = snaps[t - 1]
        y_per_t_lookup[t] = s.y.cpu().numpy() if s.x.size(0) else np.zeros(0,
                                                                          dtype=np.int64)
    true_prior = compute_true_prior_per_timestep(y_per_t_lookup)

    p_gru, y_gru, _ = _gru_per_timestep_posteriors(
        snaps, node_ids, embeds_df, gru, test_ts,
    )
    p_train_gru = _effective_p_train_gru(snaps, node_ids, embeds_df, gru)
    p_train_rf_eff = _effective_p_train_rf(rf, snaps)

    c1 = _score_condition(
        "C1 · GCN-GRU (no correction)",
        p_gru, y_gru, true_prior,
        estimated_q={t: rf.p_train_illicit for t in test_ts},
    )

    p_all = np.concatenate([p_gru[t] for t in test_ts if p_gru[t].size],
                           axis=0) if any(p_gru[t].size for t in test_ts)\
        else np.zeros((0, 2))
    if p_all.size:
        batch = saerens_em_batch(p_all, p_train_gru)
        q_batch = float(batch.q[LABEL_ILLICIT])

        p_gru_c2 = {}
        for t in test_ts:
            p = p_gru[t]
            if p.size == 0:
                p_gru_c2[t] = p
                continue
            ratio = batch.q / p_train_gru
            weighted = p * ratio[None, :]
            p_gru_c2[t] = weighted / np.clip(weighted.sum(axis=1, keepdims=True),
                                             1e-12, None)
        c2 = _score_condition(
            "C2 · GCN-GRU + batch Saerens-EM",
            p_gru_c2, y_gru, true_prior,
            estimated_q={t: q_batch for t in test_ts},
        )
        c2["batch_q_illicit"] = q_batch
        c2["batch_history"] = batch.history
    else:
        c2 = _score_condition(
            "C2 · GCN-GRU + batch Saerens-EM",
            p_gru, y_gru, true_prior,
            estimated_q={t: rf.p_train_illicit for t in test_ts},
        )

    online = online_per_timestep_tracker(
        p_gru, p_train_gru,
        alpha=alpha, beta=beta, max_iter=em_max_iter,
        timesteps=test_ts,
    )
    p_gru_c3 = {t: online.per_step[t].corrected if t in online.per_step
                else p_gru[t] for t in test_ts}
    estimated_q_c3 = {t: float(online.per_step[t].q[LABEL_ILLICIT])
                      if t in online.per_step else rf.p_train_illicit
                      for t in test_ts}
    c3 = _score_condition(
        "C3 · GCN-GRU + online per-timestep",
        p_gru_c3, y_gru, true_prior,
        estimated_q=estimated_q_c3,
    )

    rf_pred = predict_rf_per_timestep(rf, snaps, test_ts)
    p_rf = {t: rf_pred[t]["p"] for t in test_ts}
    y_rf = {t: rf_pred[t]["y"] for t in test_ts}
    rf_row = _score_condition(
        "RF · Random Forest (Maganti reference)",
        p_rf, y_rf, true_prior,
        estimated_q={t: rf.p_train_illicit for t in test_ts},
    )

    online_rf = online_per_timestep_tracker(
        p_rf, p_train_rf_eff,
        alpha=alpha, beta=beta, max_iter=em_max_iter,
        timesteps=test_ts,
    )
    p_rf_corrected = {t: online_rf.per_step[t].corrected if t in online_rf.per_step
                      else p_rf[t] for t in test_ts}
    estimated_q_rf = {t: float(online_rf.per_step[t].q[LABEL_ILLICIT])
                      if t in online_rf.per_step else rf.p_train_illicit
                      for t in test_ts}
    rf_online = _score_condition(
        "RF+ · Random Forest + online per-timestep",
        p_rf_corrected, y_rf, true_prior,
        estimated_q=estimated_q_rf,
    )

    out = {
        "gcn_gru_none":   c1,
        "gcn_gru_batch":  c2,
        "gcn_gru_online": c3,
        "rf_none":        rf_row,
        "rf_online":      rf_online,
        "true_prior":     {str(t): float(true_prior.get(t, float("nan")))
                           for t in test_ts},
        "p_train_illicit": float(rf.p_train_illicit),
        "_meta": {
            "generated_at": int(time.time()),
            "demo": True,
            "alpha": alpha, "beta": beta, "em_max_iter": em_max_iter,
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def bootstrap_demo_artefacts(
    snaps: List[Snapshot], node_ids: torch.Tensor, artefact_paths: dict,
) -> dict:
    gcn_path = Path(artefact_paths["gcn"])
    gru_path = Path(artefact_paths["hybrid_head"])
    rf_path = Path(artefact_paths["rf"])
    embed_path = Path(artefact_paths["embeddings"])
    metrics_path = Path(artefact_paths["metrics"])

    gcn = _maybe_train_gcn(snaps, gcn_path)
    embeds_df = _maybe_precompute_embeddings(snaps, node_ids, gcn, embed_path)
    gru = _maybe_train_gru(snaps, node_ids, embeds_df, gcn_path, gru_path)
    rf = _maybe_train_rf(snaps, rf_path)
    metrics = _evaluate_all_conditions(
        snaps, node_ids, embeds_df, gru, rf, metrics_path,
    )
    return {
        "gcn": gcn, "gru": gru, "rf": rf,
        "embeds": embeds_df, "metrics": metrics,
    }
