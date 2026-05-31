from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

from app.data.loader import LABEL_ILLICIT, LABEL_LICIT


def _apply_correction(p_y_given_x: np.ndarray, q: np.ndarray,
                      p_train: np.ndarray) -> np.ndarray:
    ratio = q / np.clip(p_train, 1e-8, None)
    weighted = p_y_given_x * ratio[None, :]
    norm = weighted.sum(axis=1, keepdims=True)
    return weighted / np.clip(norm, 1e-12, None)


def _em_step(p_y_given_x: np.ndarray, q: np.ndarray,
             p_train: np.ndarray,
             alpha: float = 0.0, beta: float = 0.0) -> np.ndarray:
    corrected = _apply_correction(p_y_given_x, q, p_train)
    n_illicit = corrected[:, LABEL_ILLICIT].sum()
    n_total = float(corrected.shape[0])

    q_illicit = (n_illicit + alpha) / (n_total + alpha + beta)
    q_illicit = float(np.clip(q_illicit, 1e-6, 1 - 1e-6))
    return np.array([1.0 - q_illicit, q_illicit])


@dataclass
class BatchPriorResult:
    q: np.ndarray
    corrected: np.ndarray
    history: List[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0


def saerens_em_batch(
    p_y_given_x: np.ndarray,
    p_train: np.ndarray,
    *,
    init_q: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-5,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> BatchPriorResult:
    q = (init_q.copy() if init_q is not None else p_train.copy())
    q = q / q.sum()
    history: List[float] = [float(q[LABEL_ILLICIT])]
    converged = False
    iterations = 0
    for it in range(max_iter):
        q_new = _em_step(p_y_given_x, q, p_train, alpha=alpha, beta=beta)
        history.append(float(q_new[LABEL_ILLICIT]))
        iterations = it + 1
        if np.abs(q_new - q).max() < tol:
            q = q_new
            converged = True
            break
        q = q_new
    corrected = _apply_correction(p_y_given_x, q, p_train)
    return BatchPriorResult(
        q=q, corrected=corrected, history=history,
        converged=converged, iterations=iterations,
    )


@dataclass
class OnlineStepResult:
    t: int
    q: np.ndarray
    q_init: np.ndarray
    corrected: np.ndarray
    history: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


@dataclass
class OnlineTrackerResult:
    per_step: Dict[int, OnlineStepResult] = field(default_factory=dict)

    timesteps: List[int] = field(default_factory=list)
    estimated_q_illicit: np.ndarray = field(
        default_factory=lambda: np.zeros(0))

    def stack(self) -> None:
        self.timesteps = sorted(self.per_step.keys())
        self.estimated_q_illicit = np.array([
            float(self.per_step[t].q[LABEL_ILLICIT])
            for t in self.timesteps
        ])


def online_per_timestep_tracker(
    p_y_per_t: Dict[int, np.ndarray],
    p_train: np.ndarray,
    *,
    alpha: float = 0.2,
    beta: float = 1.8,
    max_iter: int = 12,
    tol: float = 1e-5,
    init_q: np.ndarray | None = None,
    timesteps: Iterable[int] | None = None,
    init_mode: str = "prev",
    blend: float = 0.5,
    floor: float = 0.0,
) -> OnlineTrackerResult:
    if timesteps is None:
        timesteps = sorted(p_y_per_t.keys())
    timesteps = list(timesteps)

    out = OnlineTrackerResult()
    p_train = p_train / p_train.sum()
    q_prev = (init_q.copy() if init_q is not None else p_train.copy())
    q_prev = q_prev / q_prev.sum()

    for t in timesteps:
        p = p_y_per_t[t]
        if p.size == 0:
            continue

        if init_mode == "prior":
            q_init_step = p_train.copy()
        elif init_mode == "blend":
            q_init_step = blend * q_prev + (1.0 - blend) * p_train
            q_init_step = q_init_step / q_init_step.sum()
        else:
            q_init_step = q_prev.copy()

        if floor > 0.0 and q_init_step[LABEL_ILLICIT] < floor:
            q_init_step = np.array([1.0 - floor, floor])

        q = q_init_step.copy()
        history: List[float] = [float(q[LABEL_ILLICIT])]
        converged = False
        iters = 0
        for it in range(max_iter):
            q_new = _em_step(p, q, p_train, alpha=alpha, beta=beta)
            history.append(float(q_new[LABEL_ILLICIT]))
            iters = it + 1
            if np.abs(q_new - q).max() < tol:
                q = q_new
                converged = True
                break
            q = q_new
        corrected = _apply_correction(p, q, p_train)
        out.per_step[t] = OnlineStepResult(
            t=t, q=q, q_init=q_init_step.copy(),
            corrected=corrected, history=history,
            iterations=iters, converged=converged,
        )
        q_prev = q

    out.stack()
    return out


def correct_single_posterior(
    p_y_given_x: np.ndarray,
    q_y: np.ndarray,
    p_train: np.ndarray,
) -> np.ndarray:
    return _apply_correction(p_y_given_x[None, :], q_y, p_train)[0]


def compute_true_prior_per_timestep(
    labels_per_t: Dict[int, np.ndarray],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for t, y in labels_per_t.items():
        mask = (y == LABEL_ILLICIT) | (y == LABEL_LICIT)
        if not mask.any():
            out[t] = float("nan")
            continue
        out[t] = float((y[mask] == LABEL_ILLICIT).mean())
    return out


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]

    def rank(x: np.ndarray) -> np.ndarray:
        order = x.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))

        for v in np.unique(x):
            idx = np.where(x == v)[0]
            if idx.size > 1:
                ranks[idx] = ranks[idx].mean()
        return ranks

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
