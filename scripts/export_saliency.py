from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from app.data.loader import LABEL_ILLICIT, NUM_TIMESTEPS
from app.data.snapshots import time_split
from app.services.cache import (
    get_embeddings,
    get_hybrid,
    get_node_ids,
    get_snapshots,
)
from app.services.explain import gru_saliency

SHUTDOWN_T = 43


def _illicit_node_ids(snaps, node_ids) -> list[int]:
    _, _, test_range = time_split()
    found: list[int] = []
    for t in sorted(test_range, reverse=True):
        snap = snaps[t - 1]
        if snap.x.size(0) == 0:
            continue
        y = snap.y.cpu().numpy()
        ids = node_ids[snap.global_idx].cpu().numpy()
        for nid in ids[y == LABEL_ILLICIT]:
            if int(nid) not in found:
                found.append(int(nid))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--graph-cache", default="artefacts/graph.pkl")
    ap.add_argument("--gcn", default="artefacts/gcn_subnet.pt")
    ap.add_argument("--head", default="artefacts/gru_head.pt")
    ap.add_argument("--embeddings", default="artefacts/embeddings.parquet")
    ap.add_argument("--node-id", type=int, default=-1)
    ap.add_argument("--out-dir", default="artefacts/report_assets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps, _ = get_snapshots(args.data_dir, args.graph_cache)
    node_ids = get_node_ids(args.data_dir, args.graph_cache)
    embeds_df = get_embeddings(args.embeddings)
    embed_cols = [c for c in embeds_df.columns if c.startswith("e")]
    embed_dim = len(embed_cols)
    by_id = {int(nid): g.sort_values("t")
             for nid, g in embeds_df.groupby("node_id")}

    if args.node_id >= 0:
        candidates = [args.node_id]
    else:
        candidates = _illicit_node_ids(snaps, node_ids)

    chosen = None
    seq_df = None
    for nid in candidates:
        g = by_id.get(int(nid))
        if g is not None and len(g) >= 1:
            chosen = int(nid)
            seq_df = g
            break

    if chosen is None:
        print(
            f"No illicit node found in {args.embeddings}. The embeddings may "
            "be from a different graph; regenerate them for this data:\n"
            f"  python -m scripts.run_pipeline --data-dir {args.data_dir} "
            "--force-retrain --alpha 5 --beta 10 --em-iter 12 "
            "--init-mode blend --seed 1337 --note frozen-seed1337"
        )
        return

    own_weeks = [int(t) for t in seq_df["t"].tolist()]
    emb = seq_df[embed_cols].to_numpy(dtype=np.float32)
    seq_full = np.zeros((NUM_TIMESTEPS, embed_dim), dtype=np.float32)
    for t, vec in zip(own_weeks, emb):
        if 1 <= t <= NUM_TIMESTEPS:
            seq_full[t - 1] = vec
    seq = torch.from_numpy(seq_full).unsqueeze(0)

    model = get_hybrid(args.gcn, args.head)
    sal = gru_saliency(model, seq, target_class=LABEL_ILLICIT)
    values = np.asarray(sal.saliency, dtype=float)
    weeks = list(range(1, len(values) + 1))

    order = np.argsort(values)[::-1]
    top = [(weeks[i], float(values[i])) for i in order[:3]]
    near_shutdown = [w for (w, _) in top if abs(w - SHUTDOWN_T) <= 2]

    csv_path = out_dir / f"gru_saliency_node{chosen}.csv"
    with open(csv_path, "w") as fh:
        fh.write("week,saliency,is_observed_week\n")
        for w, v in zip(weeks, values):
            fh.write(f"{w},{v:.6f},{int(w in own_weeks)}\n")

    png_path = out_dir / f"gru_saliency_node{chosen}.png"
    made_png = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 4.2))
        colors = ["#D1366B" if abs(w - SHUTDOWN_T) <= 2 else "#5B4BE3"
                  for w in weeks]
        ax.bar(weeks, values, color=colors)
        ax.axvline(SHUTDOWN_T, color="#0B0F1A", linestyle=":", linewidth=1.5)
        ax.text(SHUTDOWN_T, ax.get_ylim()[1] * 0.95, " shutdown t=43",
                fontsize=9, va="top")
        for w in own_weeks:
            ax.axvline(w, color="#0F9E73", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Week")
        ax.set_ylabel("Saliency (|gradient|)")
        ax.set_title(
            f"GRU temporal saliency, illicit node {chosen} "
            f"(observed week {own_weeks[0]})",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        made_png = True
    except ImportError:
        pass

    print("=" * 60)
    print(f"illicit node        : {chosen}")
    print(f"observed week(s)    : {own_weeks}")
    print("top-3 spike weeks   : "
          + ", ".join(f"t={w} ({v:.3f})" for w, v in top))
    if near_shutdown:
        print(f"spikes near shutdown: YES (weeks {near_shutdown})")
    else:
        print("spikes near shutdown: NO (no top-3 within 2 weeks of t=43)")
    print("note                : each transaction is observed in one week, "
          "so saliency concentrates at its own week.")
    print(f"csv  -> {csv_path}")
    print(f"png  -> {png_path}" if made_png
          else "png  -> skipped (pip install matplotlib to enable)")
    print("=" * 60)


if __name__ == "__main__":
    main()
