from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


LABEL_ILLICIT = 1
LABEL_LICIT = 0
LABEL_UNKNOWN = -1

NUM_FEATURES = 166
NUM_TIMESTEPS = 49


def load_elliptic(
    data_dir: str | Path,
    cache_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> Data:
    data_dir = Path(data_dir)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_reload:
            log.info("Loading cached graph from %s", cache_path)
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

    log.info("Reading CSVs from %s", data_dir)

    features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    n_total_cols = features.shape[1]
    n_extra = n_total_cols - 2
    if n_total_cols != 167:
        log.warning(
            "Elliptic feature CSV has %d columns (expected 167). "
            "Proceeding with detected shape; the model input will be "
            "padded or truncated to %d dims.",
            n_total_cols, NUM_FEATURES,
        )
    features.columns = ["txId", "timestep"] + [f"f{i}" for i in range(n_extra)]

    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    label_map = {"1": LABEL_ILLICIT, "2": LABEL_LICIT, "unknown": LABEL_UNKNOWN}
    classes["y"] = classes["class"].astype(str).map(label_map)
    if classes["y"].isna().any():
        unexpected = classes.loc[classes["y"].isna(), "class"].unique()
        raise ValueError(f"Unexpected class values in CSV: {unexpected}")

    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    node_ids = features["txId"].sort_values().reset_index(drop=True)
    id_to_idx = {tx: i for i, tx in enumerate(node_ids)}

    features = features.set_index("txId").loc[node_ids].reset_index()
    classes = classes.set_index("txId").reindex(node_ids).reset_index()

    feat_cols = ["timestep"] + [f"f{i}" for i in range(n_extra)]
    x = torch.tensor(features[feat_cols].values, dtype=torch.float32)

    if x.shape[1] < NUM_FEATURES:
        pad = torch.zeros(x.shape[0], NUM_FEATURES - x.shape[1])
        x = torch.cat([x, pad], dim=1)
    elif x.shape[1] > NUM_FEATURES:
        x = x[:, :NUM_FEATURES]

    t = torch.tensor(features["timestep"].values, dtype=torch.long)
    y = torch.tensor(classes["y"].fillna(LABEL_UNKNOWN).values, dtype=torch.long)
    node_id = torch.tensor(node_ids.values, dtype=torch.long)

    src = edges["txId1"].map(id_to_idx)
    dst = edges["txId2"].map(id_to_idx)
    mask = src.notna() & dst.notna()
    if (~mask).any():
        log.warning("Dropped %d edges with unknown endpoints", (~mask).sum())
    edge_index = torch.tensor(
        np.stack([src[mask].values, dst[mask].values]).astype(np.int64),
        dtype=torch.long,
    )

    data = Data(x=x, edge_index=edge_index, y=y, t=t, node_id=node_id)
    data.num_nodes = x.size(0)

    log.info(
        "Loaded graph: N=%d, E=%d, illicit=%d, licit=%d, unknown=%d, timesteps=%d",
        data.num_nodes, edge_index.size(1),
        int((y == LABEL_ILLICIT).sum()), int((y == LABEL_LICIT).sum()),
        int((y == LABEL_UNKNOWN).sum()), int(t.max()),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(data, fh)
        log.info("Cached graph to %s", cache_path)

    return data
