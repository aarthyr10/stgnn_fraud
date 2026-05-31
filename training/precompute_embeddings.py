from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from app.data.loader import load_elliptic
from app.data.preprocess import apply_scaler, fit_scaler
from app.data.snapshots import build_snapshots, time_split
from app.models.gcn import StaticGCN

log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--gcn", default="artefacts/gcn_subnet.pt")
    ap.add_argument("--out", default="artefacts/embeddings.parquet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    data = load_elliptic(args.data_dir, cache_path="artefacts/graph.pkl")
    snaps = build_snapshots(data)
    train_range, _, _ = time_split()
    scaler = fit_scaler(snaps, train_range)
    snaps = apply_scaler(snaps, scaler)

    device = torch.device(args.device)
    model = StaticGCN().to(device)
    state = torch.load(args.gcn, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    rows = []
    with torch.no_grad():
        for s in snaps:
            if s.x.size(0) == 0:
                continue
            emb = model.encode(s.x.to(device), s.edge_index.to(device))
            emb = emb.cpu().numpy()
            ids = data.node_id[s.global_idx].numpy()
            for node_id, vec in zip(ids, emb):
                row = {"node_id": int(node_id), "t": s.t}
                for i, v in enumerate(vec):
                    row[f"e{i}"] = float(v)
                rows.append(row)
            log.info("t=%d  embedded %d nodes", s.t, emb.shape[0])

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Wrote %d rows to %s (%.1f MB)",
             len(df), out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
