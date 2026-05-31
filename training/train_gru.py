from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from app.data.loader import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    NUM_TIMESTEPS,
    load_elliptic,
)
from app.data.snapshots import time_split
from app.models.gcn_gru import GcnGruHybrid

log = logging.getLogger(__name__)


class NodeSequenceDataset(Dataset):

    def __init__(self, embeds: pd.DataFrame, labels: dict[int, int],
                 max_t: int = NUM_TIMESTEPS):
        self.max_t = max_t
        embed_cols = [c for c in embeds.columns if c.startswith("e")]
        self.embed_dim = len(embed_cols)

        self.by_node = {int(nid): g.sort_values("t")
                        for nid, g in embeds.groupby("node_id")}
        self.node_ids = [n for n in self.by_node if n in labels
                         and labels[n] != LABEL_UNKNOWN]
        self.labels = labels
        self.embed_cols = embed_cols

    def __len__(self) -> int:
        return len(self.node_ids)

    def __getitem__(self, i: int):
        nid = self.node_ids[i]
        g = self.by_node[nid]
        seq = torch.zeros(self.max_t, self.embed_dim, dtype=torch.float32)
        seq[g["t"].values - 1] = torch.tensor(
            g[self.embed_cols].values, dtype=torch.float32
        )
        length = int(g["t"].max())
        return seq, length, int(self.labels[nid])


def split_by_last_timestep(ds: NodeSequenceDataset, train_range, val_range, test_range):
    train_idx, val_idx, test_idx = [], [], []
    for i, nid in enumerate(ds.node_ids):
        last_t = int(ds.by_node[nid]["t"].max())
        if last_t in train_range:
            train_idx.append(i)
        elif last_t in val_range:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--embeddings", default="artefacts/embeddings.parquet")
    ap.add_argument("--gcn", default="artefacts/gcn_subnet.pt")
    ap.add_argument("--out", default="artefacts/gru_head.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    log.info("Loading labels and embeddings")
    data = load_elliptic(args.data_dir, cache_path="artefacts/graph.pkl")
    label_map = {int(nid): int(lbl)
                 for nid, lbl in zip(data.node_id.tolist(), data.y.tolist())}
    embeds = pd.read_parquet(args.embeddings)

    ds = NodeSequenceDataset(embeds, label_map)
    log.info("Labelled nodes available: %d", len(ds))
    train_range, val_range, test_range = time_split()
    tr, va, _ = split_by_last_timestep(ds, train_range, val_range, test_range)

    def loader(idx, shuffle: bool):
        sub = torch.utils.data.Subset(ds, idx)
        return DataLoader(sub, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0)

    train_dl, val_dl = loader(tr, True), loader(va, False)

    device = torch.device(args.device)
    model = GcnGruHybrid(gcn_weights_path=args.gcn).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info("Trainable params: %d (GRU + head only)",
             sum(p.numel() for p in trainable))

    opt = Adam(trainable, lr=args.lr)

    train_labels = [label_map[ds.node_ids[i]] for i in tr]
    n_ill = sum(1 for lbl in train_labels if lbl == LABEL_ILLICIT)
    n_lic = sum(1 for lbl in train_labels if lbl == LABEL_LICIT)
    cw = torch.tensor([1.0, n_lic / max(n_ill, 1)], device=device)
    log.info("Class weights: %s", cw.tolist())

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for seq, _length, y in train_dl:
            seq, y = seq.to(device), y.to(device)
            opt.zero_grad()
            logits = model(seq)
            loss = F.cross_entropy(logits, y, weight=cw)
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * seq.size(0)
        epoch_loss /= max(len(tr), 1)

        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for seq, _l, y in val_dl:
                logits = model(seq.to(device))
                preds.append(logits.argmax(-1).cpu().numpy())
                truth.append(y.numpy())
        preds = np.concatenate(preds) if preds else np.array([])
        truth = np.concatenate(truth) if truth else np.array([])
        val_f1 = (f1_score(truth, preds, pos_label=LABEL_ILLICIT, zero_division=0)
                  if len(preds) else 0.0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()
                          if not k.startswith("gcn.")}

        if epoch == 1 or epoch % 5 == 0:
            log.info("epoch %3d  loss=%.4f  val_f1=%.4f  best=%.4f",
                     epoch, epoch_loss, val_f1, best_val_f1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "val_f1": best_val_f1,
                "config": vars(args)}, out)
    log.info("Saved GRU head (val F1=%.4f) to %s", best_val_f1, out)


if __name__ == "__main__":
    main()
