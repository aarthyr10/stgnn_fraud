"""Train one frozen-GCN + GRU head per seed and cache its illicit scores.

The original study reused a single pretrained GCN-GRU artefact across all
ten seeds, so the reported seed intervals never covered the GNN's own
variability -- and the claim that the GNN supplies the ensemble's
decorrelation rested on one draw.  This script retrains it per seed so the
seed protocol is honest.

Scores are cached to ``<out>/gnn_seed_<s>.npz`` (illicit probability for
every node, in frame order) so the stacking study can reuse them.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.preprocess import apply_scaler, fit_scaler  # noqa: E402
from app.data.snapshots import build_snapshots  # noqa: E402
from app.services.ensemble import (  # noqa: E402
    LEGACY_SPLIT,
    embed_all,
    gru_illicit_scores,
    train_gcn,
    train_gru_head,
)

log = logging.getLogger("gnn_bases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="artefacts_nb/graph.pkl")
    ap.add_argument("--out", default="out/gnn")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--gcn-epochs", type=int, default=200)
    ap.add_argument("--gru-epochs", type=int, default=60)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 8])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.graph, "rb") as fh:
        data = pickle.load(fh)
    snaps = build_snapshots(data)
    snaps = apply_scaler(snaps, fit_scaler(snaps,
                                           range(1, LEGACY_SPLIT.train_end + 1)))
    node_order = {}
    for s in snaps:
        for j, gi in enumerate(s.global_idx.tolist()):
            node_order[(int(s.t), gi)] = j

    n_nodes = int(data.num_nodes)
    all_t = [s.t for s in snaps]

    for seed in args.seeds:
        path = out_dir / f"gnn_seed_{seed}.npz"
        if path.exists():
            log.info("seed %d cached, skipping", seed)
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        gcn = train_gcn(snaps, LEGACY_SPLIT, seed, epochs=args.gcn_epochs)
        embeds = embed_all(snaps, gcn)
        state = {k: v.detach().clone() for k, v in gcn.state_dict().items()}

        payload = {}
        for w in args.windows:
            gru = train_gru_head(snaps, embeds, state, LEGACY_SPLIT, seed,
                                 epochs=args.gru_epochs, window=w,
                                 in_dim=snaps[0].x.size(1))
            per_t = gru_illicit_scores(gru, embeds, all_t, window=w)
            flat = np.zeros(n_nodes, dtype=np.float32)
            for s in snaps:
                if s.x.size(0) == 0:
                    continue
                flat[s.global_idx.numpy()] = per_t[s.t]
            payload[f"w{w}"] = flat
            log.info("  seed %d window %d done (mean p=%.4f)", seed, w,
                     float(flat.mean()))
        np.savez_compressed(path, **payload)
        log.info("seed %d written in %.0fs", seed, time.time() - t0)


if __name__ == "__main__":
    main()
