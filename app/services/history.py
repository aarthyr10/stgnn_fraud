from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

CONDITIONS = (
    "gcn_only",
    "gcn_gru_none", "gcn_gru_batch", "gcn_gru_online",
    "rf_none", "rf_batch", "rf_online",
)

METRIC_KEYS = (
    "pr_auc",
    "recall_at_5pct_fpr",
    "f1_illicit",
    "f1_post_shutdown",
    "spearman_rho_prior",
    "spearman_rho_prior_full",
)


def _safe(value, default=None):
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:
            return default
        return f
    except (TypeError, ValueError):
        return default


def flatten_metrics(metrics: dict) -> dict:
    out: dict[str, float | None] = {}
    for cond in CONDITIONS:
        m = metrics.get(cond, {}) if isinstance(metrics, dict) else {}
        if not isinstance(m, dict):
            m = {}
        for key in METRIC_KEYS:
            out[f"{cond}.{key}"] = _safe(m.get(key))
    return out


def append_run(
    history_path: str | Path,
    params: dict,
    metrics: dict,
    note: str = "",
) -> dict:
    record = {
        "ts": int(time.time()),
        "note": note,
        "params": {
            "alpha": _safe(params.get("alpha"), 0.0),
            "beta": _safe(params.get("beta"), 0.0),
            "em_iter": int(params.get("em_iter", 0) or 0),
            "seed": int(params.get("seed", 0) or 0),
        },
        "metrics": flatten_metrics(metrics),
    }
    p = Path(history_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_history(history_path: str | Path) -> list[dict]:
    p = Path(history_path)
    if not p.exists():
        return []
    records: list[dict] = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def history_rows(records: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in records:
        metrics = r.get("metrics", {}) or {}
        params = r.get("params", {}) or {}
        rows.append({
            "ts": int(r.get("ts", 0) or 0),
            "note": r.get("note", "") or "",
            "alpha": _safe(params.get("alpha"), 0.0),
            "beta": _safe(params.get("beta"), 0.0),
            "em_iter": int(params.get("em_iter", 0) or 0),
            "seed": int(params.get("seed", 0) or 0),
            "C1 PR-AUC":  metrics.get("gcn_gru_none.pr_auc"),
            "C2 PR-AUC":  metrics.get("gcn_gru_batch.pr_auc"),
            "C3 PR-AUC":  metrics.get("gcn_gru_online.pr_auc"),
            "RF PR-AUC":  metrics.get("rf_none.pr_auc"),
            "RF+ PR-AUC": metrics.get("rf_online.pr_auc"),
            "GCN-only F1": metrics.get("gcn_only.f1_illicit"),
            "C1 F1":       metrics.get("gcn_gru_none.f1_illicit"),
            "C3 rho":     metrics.get("gcn_gru_online.spearman_rho_prior"),
            "C3 rho_full": metrics.get(
                "gcn_gru_online.spearman_rho_prior_full",
            ),
            "RF+ rho":    metrics.get("rf_online.spearman_rho_prior"),
            "C2 F1@t>=43":  metrics.get("gcn_gru_batch.f1_post_shutdown"),
            "C3 F1@t>=43":  metrics.get("gcn_gru_online.f1_post_shutdown"),
            "RF+ F1@t>=43": metrics.get("rf_online.f1_post_shutdown"),
            "RF F1":        metrics.get("rf_none.f1_illicit"),
        })
    return rows
