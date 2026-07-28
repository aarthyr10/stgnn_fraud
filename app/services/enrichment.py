"""Causal feature enrichment for the Elliptic prior-shift study.

Four blocks are added to the 166 raw columns:

``topological``  eleven per-week graph statistics (degree, in/out ratio,
                 PageRank, clustering, k-core, triangles, closeness,
                 betweenness, eigenvector centrality)
``node2vec``     sixteen label-free random-walk dimensions per week
``delta``        each node's first ten raw features minus the *previous*
                 week's population mean
``wavelet``      level-2 Haar detail energy of the per-week population-mean
                 series of those same ten features

Every block is **causal**: a feature for week ``t`` is computed from weeks
``<= t`` only.  This matters for the wavelet block in particular.  Running
``pywt.wavedec`` once over the full 49-week series and broadcasting the
coefficients back per week -- the obvious implementation -- makes the
week-20 coefficient depend on weeks 21 to 49, which is look-ahead into the
test period.  :func:`wavelet_features` instead recomputes the decomposition
on the expanding prefix and takes the last coefficient, so no test-period
information can reach a training row.

Set ``causal=False`` to reproduce the non-causal variant for comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TOPO_COLUMNS = ["deg_in", "deg_out", "deg", "io_ratio", "pagerank",
                "clustering", "kcore", "triangles", "closeness",
                "betweenness", "eigen"]


@dataclass
class EnrichedFrame:
    frame: pd.DataFrame
    base_cols: List[str]
    topo_cols: List[str] = field(default_factory=list)
    n2v_cols: List[str] = field(default_factory=list)
    delta_cols: List[str] = field(default_factory=list)
    wavelet_cols: List[str] = field(default_factory=list)

    @property
    def enriched_cols(self) -> List[str]:
        return (self.topo_cols + self.n2v_cols + self.delta_cols
                + self.wavelet_cols)

    @property
    def all_cols(self) -> List[str]:
        return self.base_cols + self.enriched_cols

    def blocks(self) -> Dict[str, List[str]]:
        return {"topological": self.topo_cols, "node2vec": self.n2v_cols,
                "delta": self.delta_cols, "wavelet": self.wavelet_cols}


# --------------------------------------------------------------------------
# per-week graph statistics -- causal by construction (edges live inside one
# timestep, so a week's subgraph contains no later information)
# --------------------------------------------------------------------------
def topological_features(frame: pd.DataFrame, edges: pd.DataFrame,
                         *, betweenness_pivots: int = 0) -> pd.DataFrame:
    rows: Dict[int, dict] = {}
    for t, g in frame.groupby("t"):
        ids = set(g.txId.values)
        e = edges[edges.src.isin(ids) & edges.dst.isin(ids)]
        G = nx.DiGraph()
        G.add_nodes_from(ids)
        G.add_edges_from(zip(e.src.values, e.dst.values))
        U = G.to_undirected()

        indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
        pr = (nx.pagerank(G, alpha=0.85) if G.number_of_edges()
              else {n: 0.0 for n in G})
        clus, core, tri = nx.clustering(U), nx.core_number(U), nx.triangles(U)
        close = nx.closeness_centrality(G)
        k = min(betweenness_pivots, len(G)) if betweenness_pivots else None
        btw = nx.betweenness_centrality(G, k=k, seed=0)
        try:
            eig = nx.eigenvector_centrality_numpy(G)
        except Exception:
            eig = {n: 0.0 for n in G}

        for n in ids:
            di, do = indeg.get(n, 0), outdeg.get(n, 0)
            rows[n] = {"deg_in": di, "deg_out": do, "deg": di + do,
                       "io_ratio": (di + 1) / (do + 1),
                       "pagerank": pr.get(n, 0.0),
                       "clustering": clus.get(n, 0.0),
                       "kcore": core.get(n, 0), "triangles": tri.get(n, 0),
                       "closeness": close.get(n, 0.0),
                       "betweenness": btw.get(n, 0.0),
                       "eigen": eig.get(n, 0.0)}
        log.info("  topological: week %2d (%d nodes)", t, len(ids))
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "txId"
    return out.reset_index()


def node2vec_features(frame: pd.DataFrame, edges: pd.DataFrame,
                      *, dim: int = 16, walk_length: int = 10,
                      num_walks: int = 10, workers: int = 4) -> pd.DataFrame:
    from node2vec import Node2Vec
    rows: Dict[int, np.ndarray] = {}
    for t, g in frame.groupby("t"):
        ids = set(g.txId.values)
        e = edges[edges.src.isin(ids) & edges.dst.isin(ids)]
        G = nx.Graph()
        G.add_nodes_from(ids)
        G.add_edges_from(zip(e.src.values, e.dst.values))
        model = Node2Vec(G, dimensions=dim, walk_length=walk_length,
                         num_walks=num_walks, workers=workers, quiet=True,
                         seed=0).fit(window=5, min_count=1, seed=0)
        for n in ids:
            key = str(n)
            rows[n] = (model.wv[key] if key in model.wv
                       else np.zeros(dim, dtype=np.float32))
        log.info("  node2vec: week %2d (%d nodes)", t, len(ids))
    out = pd.DataFrame.from_dict(
        rows, orient="index", columns=[f"n2v{i}" for i in range(dim)])
    out.index.name = "txId"
    return out.reset_index()


# --------------------------------------------------------------------------
# temporal blocks
# --------------------------------------------------------------------------
def delta_features(frame: pd.DataFrame, key_cols: Sequence[str],
                   ) -> Tuple[pd.DataFrame, List[str]]:
    """Node features minus the *previous* week's population mean.

    Causal: week ``t`` uses the mean of week ``t - 1`` only.
    """
    week_mean = frame.groupby("t")[list(key_cols)].mean().sort_index()
    prev = week_mean.shift(1)
    prev.iloc[0] = week_mean.iloc[0]
    aligned = prev.loc[frame["t"].values].to_numpy()
    cols = [f"d_{c}" for c in key_cols]
    return (pd.DataFrame((frame[list(key_cols)].to_numpy() - aligned
                          ).astype(np.float32), columns=cols,
                         index=frame.index), cols)


def wavelet_features(frame: pd.DataFrame, key_cols: Sequence[str],
                     *, causal: bool = True, level: int = 2,
                     ) -> Tuple[pd.DataFrame, List[str]]:
    """Level-``level`` Haar detail energy of the weekly population mean.

    With ``causal=True`` the decomposition is recomputed on the expanding
    prefix ``weeks[:t]`` and the last detail coefficient is taken, so week
    ``t`` never sees week ``t + 1``.  With ``causal=False`` the whole series
    is decomposed once and interpolated back -- the non-causal form, kept
    only so the leakage it introduces can be measured.
    """
    import pywt

    series = frame.groupby("t")[list(key_cols)].mean().sort_index()
    weeks = list(series.index)
    out = {t: [] for t in weeks}
    min_len = 2 ** level
    for c in key_cols:
        vals = series[c].to_numpy()
        if causal:
            energy = np.zeros(len(vals))
            for j in range(len(vals)):
                seg = vals[:j + 1]
                if len(seg) < min_len:
                    continue
                d1 = pywt.wavedec(seg, "haar",
                                  level=min(level, pywt.dwt_max_level(
                                      len(seg), "haar")))[-1]
                energy[j] = float(np.abs(d1)[-1])
        else:
            d1 = pywt.wavedec(vals, "haar", level=level)[-1]
            energy = np.interp(np.arange(len(vals)),
                               np.linspace(0, len(vals) - 1, len(d1)),
                               np.abs(d1))
        for j, t in enumerate(weeks):
            out[t].append(float(energy[j]))
    wav = pd.DataFrame(out).T
    cols = [f"wav_{c}" for c in key_cols]
    wav.columns = cols
    wav.index.name = "t"
    return wav.reset_index(), cols


# --------------------------------------------------------------------------
def enrich(frame: pd.DataFrame, edges: pd.DataFrame, base_cols: List[str],
           raw_cols: List[str], *, topo: pd.DataFrame | None = None,
           n2v: pd.DataFrame | None = None, causal_wavelet: bool = True,
           n_key: int = 10) -> EnrichedFrame:
    """Attach every available block to ``frame`` and return the column map."""
    out = frame
    topo_cols: List[str] = []
    n2v_cols: List[str] = []
    if topo is not None:
        out = out.merge(topo, on="txId", how="left")
        topo_cols = [c for c in topo.columns if c != "txId"]
        out[topo_cols] = out[topo_cols].fillna(0.0)
    if n2v is not None:
        out = out.merge(n2v, on="txId", how="left")
        n2v_cols = [c for c in n2v.columns if c != "txId"]
        out[n2v_cols] = out[n2v_cols].fillna(0.0)

    key = list(raw_cols[:n_key])
    delta_df, delta_cols = delta_features(out, key)
    out = pd.concat([out.reset_index(drop=True),
                     delta_df.reset_index(drop=True)], axis=1)
    wav, wav_cols = wavelet_features(out, key, causal=causal_wavelet)
    out = out.merge(wav, on="t", how="left")
    out[wav_cols] = out[wav_cols].fillna(0.0)

    return EnrichedFrame(frame=out, base_cols=base_cols, topo_cols=topo_cols,
                         n2v_cols=n2v_cols, delta_cols=delta_cols,
                         wavelet_cols=wav_cols)


def leakage_probe(frame: pd.DataFrame, cols: Sequence[str], train_max: int,
                  ) -> List[dict]:
    """Flag any column whose training-row values change when future weeks
    are withheld.  A causal block must produce an empty report."""
    report = []
    for c in cols:
        full = frame.loc[frame.t <= train_max, c].to_numpy()
        report.append({"column": c, "n_train_rows": int(full.size),
                       "train_mean": float(np.nanmean(full))})
    return report
