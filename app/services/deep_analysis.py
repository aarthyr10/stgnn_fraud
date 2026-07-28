"""Deeper analysis helpers for the prior-shift / stacking study.

Everything here is deliberately model-agnostic: it takes label vectors and
illicit-probability vectors and returns statistics.  Grouped into

* point metrics and oracle / deployable thresholds
* uncertainty: percentile bootstrap, paired bootstrap, across-seed spread
* calibration: ECE, MCE, Brier, reliability bins
* decision economics: cost curves under asymmetric FN/FP costs
* drift: per-week population stability and KS against the training window
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.data.loader import LABEL_ILLICIT, LABEL_LICIT

EPS = 1e-12


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------
def f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    if y.size == 0:
        return float("nan")
    return float(f1_score(y, (p >= thr).astype(np.int64),
                          pos_label=LABEL_ILLICIT, zero_division=0))


def oracle_f1(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Max F1 over all thresholds and the threshold that attains it.

    This uses the evaluated window's own labels, so it is an upper bound,
    not a deployable operating point.

    ``precision_recall_curve`` returns ``len(thresholds) + 1`` points: entry
    ``i`` corresponds to ``thresholds[i]`` for ``i < len(thresholds)``, and
    the final entry (precision 1, recall 0) has no threshold.  The argmax is
    therefore taken over ``f1[:-1]`` and indexes ``thresholds`` directly --
    ``thresholds[idx - 1]`` would return the neighbouring operating point.
    """
    if y.size == 0 or (y == LABEL_ILLICIT).sum() == 0:
        return 0.0, 0.5
    prec, rec, thr = precision_recall_curve(y, p, pos_label=LABEL_ILLICIT)
    if thr.size == 0:
        return 0.0, 0.5
    f1 = 2 * prec * rec / np.clip(prec + rec, EPS, None)
    idx = int(np.nanargmax(f1[:-1]))
    return float(f1[idx]), float(thr[min(idx, thr.size - 1)])


def threshold_from(y: np.ndarray, p: np.ndarray) -> float:
    """The F1-maximising threshold fitted on a *held-out* window.

    Identical arithmetic to :func:`oracle_f1`, but named for the deployable
    use: fit on validation, apply to test.
    """
    return oracle_f1(y, p)[1]


def recall_at_fpr(y: np.ndarray, p: np.ndarray, target: float = 0.05) -> float:
    if y.size == 0 or (y == LABEL_ILLICIT).sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p, pos_label=LABEL_ILLICIT)
    idx = max(0, min(int(np.searchsorted(fpr, target, side="right")) - 1,
                     len(tpr) - 1))
    return float(tpr[idx])


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or (y == LABEL_ILLICIT).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def prior_matched_predictions(
    p_per_t: Dict[int, np.ndarray], y_per_t: Dict[int, np.ndarray],
    q_per_t: Dict[int, float], timesteps: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Deployable rule: flag the top ``q_t`` fraction of each week.

    Uses only the tracked rate, never test labels, so it is the honest
    counterpart to :func:`oracle_f1`.
    """
    preds, ys = [], []
    for t in timesteps:
        y = y_per_t.get(t)
        p = p_per_t.get(t)
        if y is None or p is None or y.size == 0:
            continue
        keep = (y == LABEL_ILLICIT) | (y == LABEL_LICIT)
        if not keep.any():
            continue
        s, yt = p[keep], y[keep]
        q = q_per_t.get(t, float("nan"))
        q = 0.0 if q != q else float(np.clip(q, 0.0, 1.0))
        if q <= 0.0:
            pr = np.zeros_like(yt)
        elif q >= 1.0:
            pr = np.ones_like(yt)
        else:
            pr = (s >= float(np.quantile(s, 1.0 - q))).astype(np.int64)
        preds.append(pr)
        ys.append(yt)
    if not ys:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(preds), np.concatenate(ys)


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------
def bootstrap_ci(
    y: np.ndarray, p: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap over nodes, stratified by class."""
    if y.size == 0:
        return {"point": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == LABEL_ILLICIT)
    neg = np.flatnonzero(y != LABEL_ILLICIT)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=pos.size, replace=True),
            rng.choice(neg, size=neg.size, replace=True),
        ])
        vals[b] = metric(y[idx], p[idx])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(metric(y, p)), "lo": float(lo), "hi": float(hi),
            "n": int(y.size), "boot_median": float(np.nanmedian(vals))}


def paired_bootstrap(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap the paired difference metric(A) - metric(B) on the same
    resampled nodes, and report a two-sided bootstrap p-value."""
    if y.size == 0:
        return {"diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == LABEL_ILLICIT)
    neg = np.flatnonzero(y != LABEL_ILLICIT)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=pos.size, replace=True),
            rng.choice(neg, size=neg.size, replace=True),
        ])
        diffs[b] = metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    frac = float(np.mean(diffs <= 0)) if np.nanmean(diffs) > 0 \
        else float(np.mean(diffs >= 0))
    return {
        "diff": float(metric(y, p_a) - metric(y, p_b)),
        "lo": float(lo), "hi": float(hi),
        "p_value": float(min(1.0, 2 * frac)),
        "boot_median": float(np.nanmedian(diffs)),
    }


def seed_summary(values: Sequence[float], *, alpha: float = 0.05,
                 n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Median with a bootstrap CI of the median across seeds."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"median": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_seeds": 0}
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"median": float(np.median(v)), "lo": float(lo), "hi": float(hi),
            "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "std": float(v.std(ddof=1))
            if v.size > 1 else 0.0, "n_seeds": int(v.size)}


def paired_seed_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Sign test plus median paired difference across seeds."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return {"median_diff": float("nan"), "wins": 0, "n": 0,
                "sign_p": float("nan")}
    d = a - b
    wins = int((d > 0).sum())
    n = int((d != 0).sum())
    from math import comb
    if n == 0:
        p = 1.0
    else:
        k = min(wins, n - wins)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"median_diff": float(np.median(d)), "wins": wins,
            "n": int(a.size), "sign_p": float(p)}


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    ece: float
    mce: float
    brier: float
    bin_edges: List[float]
    bin_conf: List[float]
    bin_acc: List[float]
    bin_count: List[int]

    def to_dict(self) -> dict:
        return asdict(self)


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Calibration:
    if y.size == 0:
        return Calibration(float("nan"), float("nan"), float("nan"),
                           [], [], [], [])
    yb = (y == LABEL_ILLICIT).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        cnt.append(c)
        if c == 0:
            conf.append(float("nan"))
            acc.append(float("nan"))
            continue
        cb, ab = float(p[m].mean()), float(yb[m].mean())
        conf.append(cb)
        acc.append(ab)
        gap = abs(ab - cb)
        ece += c / y.size * gap
        mce = max(mce, gap)
    return Calibration(
        ece=float(ece), mce=float(mce),
        brier=float(brier_score_loss(yb, np.clip(p, 0, 1))),
        bin_edges=edges.tolist(), bin_conf=conf, bin_acc=acc, bin_count=cnt,
    )


# --------------------------------------------------------------------------
# decision economics
# --------------------------------------------------------------------------
def cost_curve(
    y: np.ndarray, p: np.ndarray,
    cost_ratios: Sequence[float] = (1, 5, 10, 25, 50, 100),
    n_thresholds: int = 101,
) -> Dict[str, dict]:
    """Expected cost per node against threshold for several FN:FP ratios."""
    out: Dict[str, dict] = {}
    if y.size == 0:
        return out
    yb = (y == LABEL_ILLICIT)
    thrs = np.linspace(0.0, 1.0, n_thresholds)
    fp = np.array([int((~yb & (p >= t)).sum()) for t in thrs])
    fn = np.array([int((yb & (p < t)).sum()) for t in thrs])
    for r in cost_ratios:
        cost = (fp + r * fn) / y.size
        best = int(np.argmin(cost))
        out[str(r)] = {
            "thresholds": thrs.tolist(), "cost": cost.tolist(),
            "best_threshold": float(thrs[best]),
            "best_cost": float(cost[best]),
            "cost_at_0p5": float(cost[int(n_thresholds // 2)]),
        }
    return out


def confusion_at(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    yb = (y == LABEL_ILLICIT).astype(np.int64)
    tp = int(((pred == 1) & (yb == 1)).sum())
    fp = int(((pred == 1) & (yb == 0)).sum())
    fn = int(((pred == 0) & (yb == 1)).sum())
    tn = int(((pred == 0) & (yb == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision_score(yb, pred, zero_division=0)),
        "recall": float(recall_score(yb, pred, zero_division=0)),
        "f1": float(f1_score(yb, pred, zero_division=0)),
        "alerts": int(tp + fp),
        "alert_rate": float((tp + fp) / max(len(y), 1)),
    }


# --------------------------------------------------------------------------
# per-week reporting
# --------------------------------------------------------------------------
def per_week_report(
    p_per_t: Dict[int, np.ndarray], y_per_t: Dict[int, np.ndarray],
    timesteps: Sequence[int], thr: float,
) -> List[dict]:
    rows = []
    for t in timesteps:
        y = y_per_t.get(t)
        p = p_per_t.get(t)
        if y is None or p is None or y.size == 0:
            continue
        keep = (y == LABEL_ILLICIT) | (y == LABEL_LICIT)
        if not keep.any():
            continue
        yt, pt = y[keep], p[keep]
        row = {"t": int(t), "n_labelled": int(keep.sum()),
               "prevalence": float((yt == LABEL_ILLICIT).mean()),
               "pr_auc": pr_auc(yt, pt), "roc_auc": roc_auc(yt, pt)}
        row.update({f"{k}": v for k, v in confusion_at(yt, pt, thr).items()})
        row["oracle_f1"] = oracle_f1(yt, pt)[0]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------
def population_stability_index(
    ref: np.ndarray, cur: np.ndarray, n_bins: int = 10,
) -> float:
    """PSI of a single feature between a reference and a current window."""
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return 0.0
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    r = np.clip(r / max(r.sum(), 1), 1e-6, None)
    c = np.clip(c / max(c.sum(), 1), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def weekly_drift(
    ref_X: np.ndarray, X_per_t: Dict[int, np.ndarray],
    timesteps: Sequence[int], *, n_features: int = 30,
) -> List[dict]:
    """Mean PSI over the first ``n_features`` columns, per week."""
    cols = min(n_features, ref_X.shape[1] if ref_X.ndim == 2 else 0)
    rows = []
    for t in timesteps:
        X = X_per_t.get(t)
        if X is None or X.shape[0] == 0:
            continue
        psis = [population_stability_index(ref_X[:, j], X[:, j])
                for j in range(1, cols)]
        rows.append({"t": int(t), "mean_psi": float(np.nanmean(psis)),
                     "max_psi": float(np.nanmax(psis)),
                     "n_nodes": int(X.shape[0])})
    return rows
