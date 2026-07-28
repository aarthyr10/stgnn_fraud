"""Build (and incrementally cache) the per-week topological and node2vec
feature blocks.  Both are causal by construction: Elliptic edges live inside
a single timestep, so a week's subgraph contains no later information.

Saves after every week so a long run is resumable.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ensemble_study import load_frame  # noqa: E402

log = logging.getLogger("enrich_cache")


def week_topo(ids, e, pivots: int) -> dict:
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    G.add_edges_from(zip(e.src.values, e.dst.values))
    U = G.to_undirected()
    indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
    pr = (nx.pagerank(G, alpha=0.85) if G.number_of_edges()
          else {n: 0.0 for n in G})
    clus, core, tri = nx.clustering(U), nx.core_number(U), nx.triangles(U)
    close = nx.closeness_centrality(G)
    k = min(pivots, len(G)) if pivots else None
    btw = nx.betweenness_centrality(G, k=k, seed=0)
    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eig = {n: 0.0 for n in G}
    out = {}
    for n in ids:
        di, do = indeg.get(n, 0), outdeg.get(n, 0)
        out[n] = {"deg_in": di, "deg_out": do, "deg": di + do,
                  "io_ratio": (di + 1) / (do + 1),
                  "pagerank": pr.get(n, 0.0), "clustering": clus.get(n, 0.0),
                  "kcore": core.get(n, 0), "triangles": tri.get(n, 0),
                  "closeness": close.get(n, 0.0),
                  "betweenness": btw.get(n, 0.0), "eigen": eig.get(n, 0.0)}
    return out


def week_n2v(ids, e, dim: int) -> dict:
    from node2vec import Node2Vec
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from(zip(e.src.values, e.dst.values))
    model = Node2Vec(G, dimensions=dim, walk_length=10, num_walks=10,
                     workers=2, quiet=True, seed=0).fit(window=5, min_count=1,
                                                        seed=0)
    return {n: (model.wv[str(n)] if str(n) in model.wv
                else np.zeros(dim, dtype=np.float32)) for n in ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="elldata")
    ap.add_argument("--out", default="out/enriched_cache.pkl")
    ap.add_argument("--betweenness-pivots", type=int, default=0,
                    help="0 = exact")
    ap.add_argument("--n2v-dim", type=int, default=16)
    ap.add_argument("--skip-node2vec", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df, edges, _ = load_frame(Path(args.data_dir))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = pickle.load(open(path, "rb")) if path.exists() else {
        "topo_rows": {}, "n2v_rows": {}, "weeks_topo": [], "weeks_n2v": []}

    groups = {int(t): g for t, g in df.groupby("t")}
    for t in sorted(groups):
        g = groups[t]
        ids = set(g.txId.values)
        e = edges[edges.src.isin(ids) & edges.dst.isin(ids)]
        if t not in state["weeks_topo"]:
            t0 = time.time()
            state["topo_rows"].update(week_topo(ids, e,
                                                args.betweenness_pivots))
            state["weeks_topo"].append(t)
            pickle.dump(state, open(path, "wb"))
            log.info("topo week %2d  n=%5d  %.0fs", t, len(ids),
                     time.time() - t0)
        if not args.skip_node2vec and t not in state["weeks_n2v"]:
            t0 = time.time()
            state["n2v_rows"].update(week_n2v(ids, e, args.n2v_dim))
            state["weeks_n2v"].append(t)
            pickle.dump(state, open(path, "wb"))
            log.info("n2v  week %2d  n=%5d  %.0fs", t, len(ids),
                     time.time() - t0)

    topo = pd.DataFrame.from_dict(state["topo_rows"], orient="index")
    topo.index.name = "txId"
    state["topo"] = topo.reset_index()
    if state["n2v_rows"]:
        n2v = pd.DataFrame.from_dict(
            state["n2v_rows"], orient="index",
            columns=[f"n2v{i}" for i in range(args.n2v_dim)])
        n2v.index.name = "txId"
        state["n2v"] = n2v.reset_index()
    pickle.dump(state, open(path, "wb"))
    log.info("cache complete: topo %s  n2v %s", state["topo"].shape,
             state.get("n2v", pd.DataFrame()).shape)


if __name__ == "__main__":
    main()
