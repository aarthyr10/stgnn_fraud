from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from app.data.loader import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    NUM_FEATURES,
    NUM_TIMESTEPS,
)

log = logging.getLogger(__name__)


ELLIPTIC_PRIOR_TRAJECTORY: dict[int, float] = {
    **{t: 0.130 for t in range(1, 43)},
    43: 0.120,
    44: 0.035,
    45: 0.015,
    46: 0.012,
    47: 0.040,
    48: 0.085,
    49: 0.125,
}


@dataclass(frozen=True)
class DemoSpec:
    nodes_per_timestep: int = 600
    labelled_rate: float = 0.75
    base_degree: float = 3.0
    illicit_extra_degree: float = 2.0
    cluster_size: int = 6
    seed: int = 42
    feature_noise_std: float = 0.28
    illicit_mean_shift: float = 3.0
    shutdown_shock: float = 0.08


def _gaussian_features(n: int, dim: int, mean: float,
                       rng: np.random.Generator,
                       noise_std: float = 0.5) -> np.ndarray:
    base = rng.standard_normal((n, dim)).astype(np.float32) * noise_std
    sig = np.zeros(dim, dtype=np.float32)
    sig[: dim // 2] = mean
    return base + sig


def _add_temporal_drift(x: np.ndarray, t: int,
                        rng: np.random.Generator) -> np.ndarray:
    shift = np.sin(t / 8.0) * 0.3 + (t / NUM_TIMESTEPS) * 0.5
    drift = rng.standard_normal(x.shape[-1]).astype(np.float32) * 0.05 + shift
    return x + drift


def _shutdown_shock(x: np.ndarray, t: int,
                    magnitude: float = 0.25) -> np.ndarray:
    if t < 43 or magnitude == 0:
        return x
    bump = np.zeros(x.shape[-1], dtype=np.float32)
    bump[x.shape[-1] // 2:] = magnitude
    return x + bump


def _build_edges(
    n: int, illicit_mask: np.ndarray, spec: DemoSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    edges: list[tuple[int, int]] = []

    for v in range(n):
        deg = int(rng.poisson(spec.base_degree))
        if deg == 0:
            continue
        targets = rng.choice(n, size=min(deg, n - 1), replace=False)
        edges.extend((v, int(u)) for u in targets if u != v)

    illicit_idx = np.where(illicit_mask)[0]
    if illicit_idx.size >= 2:
        cluster_count = max(1, illicit_idx.size // spec.cluster_size)
        chunks = np.array_split(rng.permutation(illicit_idx), cluster_count)
        for cluster in chunks:
            if cluster.size < 2:
                continue
            for a, b in zip(cluster[:-1], cluster[1:]):
                edges.append((int(a), int(b)))
            for a in cluster[1:]:
                edges.append((int(a), int(cluster[0])))
            licit_nodes = np.where(~illicit_mask)[0]
            if licit_nodes.size:
                cashouts = rng.choice(
                    licit_nodes, size=min(2, licit_nodes.size), replace=False,
                )
                for a in cluster[: min(2, cluster.size)]:
                    edges.append((int(a), int(cashouts[0])))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).T
    seen: set[tuple[int, int]] = set()
    keep = []
    for j in range(arr.shape[1]):
        e = (int(arr[0, j]), int(arr[1, j]))
        if e not in seen:
            seen.add(e)
            keep.append(j)
    return arr[:, keep]


def generate_demo_graph(spec: DemoSpec | None = None) -> Data:
    spec = spec or DemoSpec()
    rng = np.random.default_rng(spec.seed)

    all_x: list[np.ndarray] = []
    all_t: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_global: list[int] = []
    edge_lists: list[np.ndarray] = []
    realised_prior: dict[int, float] = {}

    cumulative = 0
    for t in range(1, NUM_TIMESTEPS + 1):
        n = spec.nodes_per_timestep + int(rng.integers(-30, 30))
        n = max(120, n)

        rate = ELLIPTIC_PRIOR_TRAJECTORY.get(t, 0.115)

        u = rng.random(n)
        labelled = u < spec.labelled_rate

        illicit_u = rng.random(n)
        illicit_mask = labelled & (illicit_u < rate)
        licit_mask = labelled & ~illicit_mask
        y = np.full(n, LABEL_UNKNOWN, dtype=np.int64)
        y[illicit_mask] = LABEL_ILLICIT
        y[licit_mask] = LABEL_LICIT

        n_lab = int(labelled.sum())
        realised_prior[t] = (
            float(illicit_mask.sum() / max(n_lab, 1))
            if n_lab else float("nan")
        )

        x_licit = _gaussian_features(
            n, NUM_FEATURES, mean=0.0, rng=rng,
            noise_std=spec.feature_noise_std,
        )
        x_illicit = _gaussian_features(
            n, NUM_FEATURES, mean=spec.illicit_mean_shift, rng=rng,
            noise_std=spec.feature_noise_std,
        )
        x = np.where(illicit_mask[:, None], x_illicit, x_licit).astype(np.float32)

        x = _add_temporal_drift(x, t, rng)
        x = _shutdown_shock(x, t, magnitude=spec.shutdown_shock)

        e = _build_edges(n, illicit_mask, spec, rng)
        if e.size:
            e_global = e + cumulative
            edge_lists.append(e_global)

        all_x.append(x)
        all_t.append(np.full(n, t, dtype=np.int64))
        all_y.append(y)
        all_global.extend(range(cumulative, cumulative + n))
        cumulative += n

    x = torch.from_numpy(np.concatenate(all_x, axis=0).astype(np.float32))
    t = torch.from_numpy(np.concatenate(all_t, axis=0))
    y = torch.from_numpy(np.concatenate(all_y, axis=0))
    node_id = torch.tensor(all_global, dtype=torch.long)
    if edge_lists:
        edge_index = torch.from_numpy(np.concatenate(edge_lists, axis=1))
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, t=t, node_id=node_id)
    data.num_nodes = x.size(0)
    data.is_demo = True
    data.true_prior = realised_prior

    log.info(
        "Generated demo graph: N=%d, E=%d, illicit=%d, licit=%d, "
        "unknown=%d, trough@t=46 rate=%.4f, recovery@t=49 rate=%.4f",
        data.num_nodes, edge_index.size(1),
        int((y == LABEL_ILLICIT).sum()), int((y == LABEL_LICIT).sum()),
        int((y == LABEL_UNKNOWN).sum()),
        realised_prior.get(46, 0.0), realised_prior.get(49, 0.0),
    )
    return data
