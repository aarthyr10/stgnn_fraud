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
from app.models.tgat import TGAT

log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--out", default="artefacts/tgat.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    data = load_elliptic(args.data_dir, cache_path="artefacts/graph.pkl")
    snaps = build_snapshots(data)
    train_range, val_range, _ = time_split()
    scaler = fit_scaler(snaps, train_range)
    snaps = apply_scaler(snaps, scaler)

    device = torch.device(args.device)
    model = TGAT().to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_y = torch.cat([s.y for s in snaps if s.t in train_range])
    n_ill = int((train_y == LABEL_ILLICIT).sum())
    n_lic = int((train_y == 0).sum())
    cw = torch.tensor([1.0, n_lic / max(n_ill, 1)], device=device)
    log.info("Class weights (licit, illicit): %s", cw.tolist())

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for s in snaps:
            if s.t not in train_range or s.x.size(0) == 0:
                continue
            x = s.x.to(device)
            ei = s.edge_index.to(device)
            y = s.y.to(device)
            t = torch.full((x.size(0),), s.t, dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(x, ei, t)
            loss = F.cross_entropy(logits, y, weight=cw,
                                   ignore_index=LABEL_UNKNOWN)
            loss.backward()
            opt.step()
            epoch_loss += float(loss)

        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for s in snaps:
                if s.t not in val_range or s.x.size(0) == 0:
                    continue
                t = torch.full((s.x.size(0),), s.t, dtype=torch.long, device=device)
                logits = model(s.x.to(device), s.edge_index.to(device), t)
                p = logits.argmax(-1).cpu()
                mask = (s.y != LABEL_UNKNOWN)
                preds.append(p[mask])
                truth.append(s.y[mask])
        if preds:
            preds = torch.cat(preds).numpy()
            truth = torch.cat(truth).numpy()
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
                "config": vars(args)}, out)
    log.info("Saved TGAT (val F1=%.4f) to %s", best_val_f1, out)


if __name__ == "__main__":
    main()
