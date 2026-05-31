from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from app.data.loader import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN
from app.data.snapshots import Snapshot, time_split
from app.models.gcn_gru import GcnGruHybrid

log = logging.getLogger(__name__)


def _build_node_index(
    snaps: List[Snapshot],
    node_ids: torch.Tensor,
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, int]]:
    appearances: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    labels: Dict[int, int] = {}
    train_range, _, _ = time_split()
    train_t = set(train_range)

    for s in snaps:
        if s.x.size(0) == 0:
            continue
        ids = node_ids[s.global_idx].cpu().numpy()
        y = s.y.cpu().numpy()
        for local_idx, (nid, lab) in enumerate(zip(ids, y)):
            nid = int(nid)
            appearances[nid].append((int(s.t), local_idx))
            if s.t in train_t and lab != LABEL_UNKNOWN and nid not in labels:
                labels[nid] = int(lab)

    for nid in appearances:
        appearances[nid].sort()
    return appearances, labels


def _epoch(
    model: GcnGruHybrid,
    snaps: List[Snapshot],
    appearances: Dict[int, List[Tuple[int, int]]],
    train_nids: List[int],
    labels: Dict[int, int],
    class_weight: torch.Tensor,
    opt: torch.optim.Optimizer,
    batch_size: int,
    train_t: set,
    rng: np.random.Generator,
    grad_clip: float = 1.0,
) -> Tuple[float, int]:
    model.train()
    perm = rng.permutation(len(train_nids))
    epoch_loss = 0.0
    n_used = 0

    for start in range(0, len(perm), batch_size):
        batch_idx = perm[start: start + batch_size]
        batch_nids = [train_nids[i] for i in batch_idx]

        ts_in_batch = sorted({
            t for nid in batch_nids
            for (t, _) in appearances[nid]
            if t in train_t
        })
        if not ts_in_batch:
            continue

        embeds_per_t: Dict[int, torch.Tensor] = {}
        for t in ts_in_batch:
            s = snaps[t - 1]

            embeds_per_t[t] = model.gcn.encode(s.x, s.edge_index)

        by_len: Dict[int, List[Tuple[torch.Tensor, int]]] = defaultdict(list)
        for nid in batch_nids:
            seq_rows: List[torch.Tensor] = []
            for (t, local_idx) in appearances[nid]:
                if t not in train_t:
                    continue
                seq_rows.append(embeds_per_t[t][local_idx])
            if not seq_rows:
                continue
            seq = torch.stack(seq_rows, dim=0)
            by_len[seq.size(0)].append((seq, labels[nid]))

        if not by_len:
            continue

        opt.zero_grad()
        total = torch.zeros((), device=seq.device)
        n_batch = 0
        for length, items in by_len.items():
            seqs = torch.stack([s for s, _ in items], dim=0)
            y = torch.tensor([lab for _, lab in items], dtype=torch.long,
                             device=seqs.device)
            logits = model(seqs)
            loss = F.cross_entropy(logits, y, weight=class_weight)
            total = total + loss * seqs.size(0)
            n_batch += seqs.size(0)
        loss = total / max(n_batch, 1)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        epoch_loss += loss.detach().item() * n_batch
        n_used += n_batch

    return epoch_loss, n_used


def _val_f1(
    model: GcnGruHybrid,
    snaps: List[Snapshot],
    appearances: Dict[int, List[Tuple[int, int]]],
    labels: Dict[int, int],
    val_t: set,
    train_t: set,
) -> float:
    model.eval()
    preds, truth = [], []

    embeds_per_t: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for t in sorted(train_t):
            s = snaps[t - 1]
            if s.x.size(0) == 0:
                continue
            embeds_per_t[t] = model.gcn.encode(s.x, s.edge_index)

        val_nids = []
        for nid, appears in appearances.items():
            if nid not in labels:
                continue
            train_appears = [(t, idx) for t, idx in appears if t in train_t]
            if not train_appears:
                continue
            if train_appears[-1][0] in val_t:
                val_nids.append((nid, train_appears))
        if not val_nids:
            return 0.0

        by_len: Dict[int, List[Tuple[torch.Tensor, int]]] = defaultdict(list)
        for nid, appears in val_nids:
            seq_rows = [embeds_per_t[t][idx] for t, idx in appears
                        if t in embeds_per_t]
            if not seq_rows:
                continue
            seq = torch.stack(seq_rows, dim=0)
            by_len[seq.size(0)].append((seq, labels[nid]))

        for length, items in by_len.items():
            seqs = torch.stack([s for s, _ in items], dim=0)
            ys = [lab for _, lab in items]
            logits = model(seqs)
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            truth.extend(ys)

    if not preds:
        return 0.0
    return float(f1_score(
        truth, preds, pos_label=LABEL_ILLICIT, zero_division=0,
    ))


def _assert_gcn_grads_flow(model: GcnGruHybrid) -> None:
    gcn_has_grad = False
    for name, p in model.gcn.named_parameters():
        if p.grad is not None and float(p.grad.abs().sum()) > 0:
            gcn_has_grad = True
            break
    if not gcn_has_grad:
        raise RuntimeError(
            "Joint training is misconfigured: no GCN parameter has a "
            "non-zero gradient after the first backward pass. Check "
            "GcnGruHybrid(freeze_gcn=False) and that requires_grad is "
            "True on conv1/conv2."
        )


def train_joint(
    snaps: List[Snapshot],
    node_ids: torch.Tensor,
    out_gcn: Path,
    out_gru: Path,
    *,
    epochs: int = 150,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    gru_hidden: int = 128,
    gru_layers: int = 2,
    dropout: float = 0.2,
    seed: int = 42,
    gcn_init_path: Path | None = None,
    grad_clip: float = 1.0,
    log_every: int = 5,
    early_stop: bool = False,
    val_timesteps: Tuple[int, ...] = (30, 31, 32, 33, 34),
    patience: int = 15,
    val_every: int = 2,
) -> GcnGruHybrid:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if early_stop:
        if gru_hidden == 128:
            gru_hidden = 64
            log.info("early_stop: reducing gru_hidden 128 -> 64")
        if abs(weight_decay - 1e-4) < 1e-9:
            weight_decay = 5e-4
            log.info("early_stop: bumping weight_decay 1e-4 -> 5e-4")
        if epochs == 150:
            epochs = 200
            log.info("early_stop: epoch cap 150 -> 200 "
                     "(early stop with patience=%d will halt sooner)",
                     patience)

    model = GcnGruHybrid(
        gcn_weights_path=str(gcn_init_path)
            if gcn_init_path and gcn_init_path.exists() else None,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        dropout=dropout,
        freeze_gcn=False,
    )

    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    train_range, _, _ = time_split()
    train_t = set(train_range)

    log.info("Building node-appearance index...")
    appearances, labels = _build_node_index(snaps, node_ids)

    train_nids = sorted([
        nid for nid in labels
        if any(t in train_t for (t, _) in appearances.get(nid, []))
    ])
    if not train_nids:
        raise RuntimeError("No labelled training nodes found.")

    n_il = sum(1 for nid in train_nids if labels[nid] == LABEL_ILLICIT)
    n_li = sum(1 for nid in train_nids if labels[nid] == LABEL_LICIT)
    n_total = max(n_il + n_li, 1)
    w_li = n_total / (2.0 * max(n_li, 1))
    w_il = n_total / (2.0 * max(n_il, 1))
    class_weight = torch.tensor([w_li, w_il], dtype=torch.float32)
    log.info(
        "Joint train: %d labelled nodes (%d illicit, %d licit) | "
        "cw=[licit=%.3f, illicit=%.3f] | epochs=%d lr=%.4f "
        "gru_hidden=%d batch=%d",
        len(train_nids), n_il, n_li, w_li, w_il,
        epochs, lr, gru_hidden, batch_size,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    val_t = set(val_timesteps) if early_stop else set()
    if early_stop:

        before = len(train_nids)
        train_nids = [
            nid for nid in train_nids
            if next((t for (t, _) in reversed(appearances[nid])
                     if t in train_t), None) not in val_t
        ]
        log.info("early_stop: held out %d nodes for validation "
                 "(t=%s), %d remain for training",
                 before - len(train_nids), sorted(val_t),
                 len(train_nids))

    grad_checked = False
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        epoch_loss, n = _epoch(
            model, snaps, appearances, train_nids, labels,
            class_weight, opt, batch_size, train_t, rng,
            grad_clip=grad_clip,
        )
        if not grad_checked:
            _assert_gcn_grads_flow(model)
            grad_checked = True
            log.info("✓ GCN gradients confirmed flowing.")

        log_this = (epoch == 1 or epoch % log_every == 0
                    or epoch == epochs)

        if early_stop and (epoch % val_every == 0 or epoch == 1):
            val_f1 = _val_f1(model, snaps, appearances, labels,
                             val_t, train_t)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch

                best_state = copy.deepcopy(model.state_dict())
                epochs_since_best = 0
                marker = " *best*"
            else:
                epochs_since_best += val_every
                marker = ""
            if log_this or marker:
                avg = epoch_loss / max(n, 1)
                log.info(
                    "  epoch %3d/%d  loss=%.4f  val_F1=%.4f%s  "
                    "patience=%d/%d",
                    epoch, epochs, avg, val_f1, marker,
                    epochs_since_best, patience,
                )
            if epochs_since_best >= patience:
                log.info("EARLY STOP at epoch %d "
                         "(best val_F1=%.4f at epoch %d, "
                         "no improvement for %d epochs)",
                         epoch, best_val_f1, best_epoch,
                         epochs_since_best)
                break
        elif log_this:
            avg = epoch_loss / max(n, 1)
            log.info("  epoch %3d/%d  loss=%.4f  examples=%d",
                     epoch, epochs, avg, n)

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
        log.info("Restored best val_F1=%.4f checkpoint from epoch %d",
                 best_val_f1, best_epoch)

    out_gcn.parent.mkdir(parents=True, exist_ok=True)
    gcn_state = {k: v.detach().cpu().clone()
                 for k, v in model.gcn.state_dict().items()}
    torch.save({
        "model": gcn_state, "seed": seed,
        "config": {"joint": True, "epochs": epochs, "lr": lr,
                   "gru_hidden": gru_hidden},
    }, out_gcn)

    head_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()
                  if not k.startswith("gcn.")}
    torch.save({
        "model": head_state, "seed": seed,
        "config": {"joint": True, "epochs": epochs, "lr": lr,
                   "gru_hidden": gru_hidden},
    }, out_gru)
    log.info("Joint training complete. GCN -> %s, GRU head -> %s",
             out_gcn, out_gru)

    model.eval()
    return model
