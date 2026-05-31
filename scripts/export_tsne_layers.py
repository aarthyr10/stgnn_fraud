from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from app.data.loader import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN
from app.services.cache import get_gcn, get_node_ids, get_snapshots
from app.services.explain import gcn_layer_embeddings


def _project(h: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    perp = min(perplexity, max(5, (h.shape[0] - 1) // 3))
    return TSNE(
        n_components=2, perplexity=perp, init="pca",
        random_state=seed,
    ).fit_transform(h)


def _separation(coords: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(coords, labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--graph-cache", default="artefacts/graph.pkl")
    ap.add_argument("--gcn", default="artefacts/gcn_subnet.pt")
    ap.add_argument("--timestep", type=int, default=49)
    ap.add_argument("--max-nodes", type=int, default=2000)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default="artefacts/report_assets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps, _ = get_snapshots(args.data_dir, args.graph_cache)
    node_ids = get_node_ids(args.data_dir, args.graph_cache)
    gcn = get_gcn(args.gcn)

    snap = snaps[args.timestep - 1]
    if snap.x.size(0) == 0:
        print(f"No nodes at timestep {args.timestep}.")
        return

    h1, h2 = gcn_layer_embeddings(gcn, snap)
    h1 = h1.cpu().numpy()
    h2 = h2.cpu().numpy()
    y = snap.y.cpu().numpy()
    ids = node_ids[snap.global_idx].cpu().numpy()

    labelled = y != LABEL_UNKNOWN
    if labelled.sum() < 10:
        print("Too few labelled nodes to project.")
        return

    idx = np.where(labelled)[0]
    if idx.shape[0] > args.max_nodes:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(idx, size=args.max_nodes, replace=False)

    yl = y[idx]
    coords1 = _project(h1[idx], args.perplexity, args.seed)
    coords2 = _project(h2[idx], args.perplexity, args.seed)
    sil1 = _separation(coords1, yl)
    sil2 = _separation(coords2, yl)

    rows = []
    for layer, coords in [(1, coords1), (2, coords2)]:
        for j, node in enumerate(idx):
            rows.append((
                args.timestep, layer, int(ids[node]),
                int(yl[j]), float(coords[j, 0]), float(coords[j, 1]),
            ))
    csv_path = out_dir / f"tsne_layers_t{args.timestep}.csv"
    with open(csv_path, "w") as fh:
        fh.write("timestep,layer,node_id,label,x,y\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")

    png_path = out_dir / f"tsne_layers_t{args.timestep}.png"
    made_png = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, layer, coords, sil in [
            (axes[0], 1, coords1, sil1),
            (axes[1], 2, coords2, sil2),
        ]:
            for cls, color, name in [
                (LABEL_LICIT, "#0F9E73", "licit"),
                (LABEL_ILLICIT, "#D1366B", "illicit"),
            ]:
                m = yl == cls
                ax.scatter(coords[m, 0], coords[m, 1], s=8,
                           c=color, label=name, alpha=0.65,
                           edgecolors="none")
            ax.set_title(
                f"GCN layer {layer}  (silhouette={sil:.3f})",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="upper right", frameon=False)
        fig.suptitle(
            f"t-SNE of GCN embeddings, timestep {args.timestep}",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        made_png = True
    except ImportError:
        pass

    print("=" * 60)
    print(f"timestep            : {args.timestep}")
    print(f"labelled nodes used : {idx.shape[0]} "
          f"({int((yl == LABEL_ILLICIT).sum())} illicit)")
    print(f"layer-1 silhouette  : {sil1:.4f}")
    print(f"layer-2 silhouette  : {sil2:.4f}")
    verdict = ("YES" if sil2 > sil1 else "NO")
    print(f"layer 2 separates illicit/licit more cleanly than layer 1? "
          f"{verdict}")
    print(f"csv  -> {csv_path}")
    print(f"png  -> {png_path}" if made_png
          else "png  -> skipped (pip install matplotlib to enable)")
    print("=" * 60)


if __name__ == "__main__":
    main()
