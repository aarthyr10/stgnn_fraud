from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam

from app.data.loader import LABEL_ILLICIT, LABEL_UNKNOWN, load_elliptic
from app.data.preprocess import apply_scaler, fit_scaler
from app.data.snapshots import build_snapshots, time_split
from app.models.gcn import StaticGCN

log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--out", default="artefacts/gcn_subnet.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    log.info("Loading dataset")
    data = load_elliptic(args.data_dir, cache_path="artefacts/graph.pkl")
    snaps = build_snapshots(data)
    train_range, val_range, _ = time_split()
    scaler = fit_scaler(snaps, train_range)
    snaps = apply_scaler(snaps, scaler)

    device = torch.device(args.device)
    model = StaticGCN().to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_labels = torch.cat([s.y for s in snaps if s.t in train_range])
    labelled = train_labels[train_labels != LABEL_UNKNOWN]
    n_illicit = int((labelled == LABEL_ILLICIT).sum())
    n_licit = int((labelled == 0).sum())
    class_weight = torch.tensor([1.0, n_licit / max(n_illicit, 1)], device=device)
    log.info("Class weights (licit, illicit): %s", class_weight.tolist())

    best_val_f1, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for s in snaps:
            if s.t not in train_range:
                continue
            x = s.x.to(device)
            ei = s.edge_index.to(device)
            y = s.y.to(device)
            opt.zero_grad()
            logits = model(x, ei)
            loss = F.cross_entropy(logits, y, weight=class_weight,
                                   ignore_index=LABEL_UNKNOWN)
            loss.backward()
            opt.step()
            epoch_loss += float(loss)

        model.eval()
        all_preds, all_truth = [], []
        with torch.no_grad():
            for s in snaps:
                if s.t not in val_range:
                    continue
                logits = model(s.x.to(device), s.edge_index.to(device))
                preds = logits.argmax(-1).cpu()
                mask = (s.y != LABEL_UNKNOWN)
                all_preds.append(preds[mask])
                all_truth.append(s.y[mask])
        if all_preds:
            preds = torch.cat(all_preds).numpy()
            truth = torch.cat(all_truth).numpy()
            val_f1 = f1_score(truth, preds, pos_label=LABEL_ILLICIT,
                              zero_division=0)
        else:
            val_f1 = 0.0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            log.info("epoch %3d  loss=%.4f  val_f1=%.4f  best=%.4f",
                     epoch, epoch_loss, val_f1, best_val_f1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "val_f1": best_val_f1,
                "seed": args.seed, "config": vars(args)}, out)
    log.info("Saved best checkpoint (val F1=%.4f) to %s", best_val_f1, out)


if __name__ == "__main__":
    main()
