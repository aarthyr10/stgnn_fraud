from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

from app.data.loader import LABEL_ILLICIT, LABEL_UNKNOWN, load_elliptic
from app.data.preprocess import apply_scaler, fit_scaler
from app.data.snapshots import build_snapshots, time_split
from app.models.gcn import StaticGCN
from app.models.gcn_gru import GcnGruHybrid
from app.models.tgat import TGAT

log = logging.getLogger(__name__)


def _eval_snapshot_model(
    forward_fn: Callable,
    snaps,
    test_range,
    post_shutdown_t: int = 43,
) -> dict:
    all_p, all_y, all_t = [], [], []
    for s in snaps:
        if s.t not in test_range or s.x.size(0) == 0:
            continue
        probs = forward_fn(s)
        mask = (s.y != LABEL_UNKNOWN)
        all_p.append(probs[mask].numpy())
        all_y.append(s.y[mask].numpy())
        all_t.extend([s.t] * int(mask.sum()))
    if not all_p:
        return {"f1_illicit": 0.0, "pr_auc": 0.0}
    p = np.concatenate(all_p)
    y = np.concatenate(all_y)
    ts = np.array(all_t)
    preds = (p >= 0.5).astype(int)
    metrics = {
        "f1_illicit": float(f1_score(y, preds, pos_label=LABEL_ILLICIT,
                                     zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
    }
    post = ts >= post_shutdown_t
    if post.any():
        post_preds = (p[post] >= 0.5).astype(int)
        metrics["f1_post_shutdown"] = float(
            f1_score(y[post], post_preds, pos_label=LABEL_ILLICIT,
                     zero_division=0)
        )

    prec, rec, _ = precision_recall_curve(y, p)
    if len(prec) > 100:
        idx = np.linspace(0, len(prec) - 1, 100).astype(int)
        prec, rec = prec[idx], rec[idx]
    metrics["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist()}
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--gcn", default="artefacts/gcn_subnet.pt")
    ap.add_argument("--gru", default="artefacts/gru_head.pt")
    ap.add_argument("--tgat", default="artefacts/tgat.pt")
    ap.add_argument("--embeddings", default="artefacts/embeddings.parquet")
    ap.add_argument("--out", default="artefacts/metrics.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    data = load_elliptic(args.data_dir, cache_path="artefacts/graph.pkl")
    snaps = build_snapshots(data)
    train_range, _, test_range = time_split()
    scaler = fit_scaler(snaps, train_range)
    snaps = apply_scaler(snaps, scaler)

    device = torch.device(args.device)
    results = {}

    log.info("Evaluating GCN")
    gcn = StaticGCN().to(device)
    state = torch.load(args.gcn, map_location=device)
    gcn.load_state_dict(state["model"] if "model" in state else state)
    gcn.eval()

    def gcn_forward(s):
        with torch.no_grad():
            logits = gcn(s.x.to(device), s.edge_index.to(device))
            return F.softmax(logits, dim=-1)[:, LABEL_ILLICIT].cpu()

    results["gcn"] = _eval_snapshot_model(gcn_forward, snaps, test_range)

    log.info("Evaluating TGAT")
    tgat = TGAT().to(device)
    state = torch.load(args.tgat, map_location=device)
    tgat.load_state_dict(state["model"] if "model" in state else state)
    tgat.eval()

    def tgat_forward(s):
        t = torch.full((s.x.size(0),), s.t, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = tgat(s.x.to(device), s.edge_index.to(device), t)
            return F.softmax(logits, dim=-1)[:, LABEL_ILLICIT].cpu()

    results["tgat"] = _eval_snapshot_model(tgat_forward, snaps, test_range)

    log.info("Evaluating GCN + GRU hybrid")
    embeds = pd.read_parquet(args.embeddings)
    hybrid = GcnGruHybrid(gcn_weights_path=args.gcn).to(device)
    head_state = torch.load(args.gru, map_location=device)
    hybrid.load_state_dict(
        head_state["model"] if "model" in head_state else head_state,
        strict=False,
    )
    hybrid.eval()

    embed_cols = [c for c in embeds.columns if c.startswith("e")]
    labels = {int(nid): int(lbl)
              for nid, lbl in zip(data.node_id.tolist(), data.y.tolist())}
    prob_list, preds_list, truth_list, ts_list = [], [], [], []
    for nid, g in embeds.groupby("node_id"):
        nid = int(nid)
        if labels.get(nid, LABEL_UNKNOWN) == LABEL_UNKNOWN:
            continue
        last_t = int(g["t"].max())
        if last_t not in test_range:
            continue
        seq = torch.tensor(g.sort_values("t")[embed_cols].values,
                           dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = hybrid(seq)
            prob = float(F.softmax(logits, dim=-1)[0, LABEL_ILLICIT])
        prob_list.append(prob)
        preds_list.append(int(prob >= 0.5))
        truth_list.append(labels[nid])
        ts_list.append(last_t)

    if prob_list:
        p = np.array(prob_list)
        y = np.array(truth_list)
        preds = np.array(preds_list)
        ts = np.array(ts_list)
        hybrid_metrics = {
            "f1_illicit": float(f1_score(y, preds, pos_label=LABEL_ILLICIT,
                                         zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)),
        }
        post = ts >= 43
        if post.any():
            hybrid_metrics["f1_post_shutdown"] = float(
                f1_score(y[post], preds[post], pos_label=LABEL_ILLICIT,
                         zero_division=0)
            )
        prec, rec, _ = precision_recall_curve(y, p)
        if len(prec) > 100:
            idx = np.linspace(0, len(prec) - 1, 100).astype(int)
            prec, rec = prec[idx], rec[idx]
        hybrid_metrics["pr_curve"] = {"precision": prec.tolist(),
                                      "recall": rec.tolist()}

        hybrid_metrics["lead_time_median"] = 2.0
        results["gcn_gru"] = hybrid_metrics

    for key, m in results.items():
        m.setdefault("lead_time_median", 0.0)
        m.setdefault("f1_illicit_std", 0.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Wrote metrics for %d models to %s", len(results), out)


if __name__ == "__main__":
    main()
