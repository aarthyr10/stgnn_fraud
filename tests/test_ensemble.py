"""Tests for the stacking ensemble, the causal enrichment blocks and the
statistics helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from app.services import deep_analysis as da
from app.services.enrichment import delta_features, wavelet_features
from app.services.ensemble import (
    LEAK_AWARE_SPLIT,
    LEGACY_SPLIT,
    BaseScores,
    error_correlation,
    fit_stack,
)


# ---------------------------------------------------------------- splits
def test_legacy_split_has_no_validation_window():
    assert LEGACY_SPLIT.val == []
    assert not LEGACY_SPLIT.has_validation
    assert LEGACY_SPLIT.test[0] == 35 and LEGACY_SPLIT.test[-1] == 49


def test_leak_aware_split_windows_do_not_overlap():
    s = LEAK_AWARE_SPLIT
    assert s.train[-1] == 34 and s.val == list(range(35, 42))
    assert s.test == list(range(42, 50))
    assert not (set(s.train) & set(s.val)) and not (set(s.val) & set(s.test))
    assert s.post_shutdown == list(range(43, 50))


def test_stack_refuses_to_fit_without_a_validation_window():
    labels = {t: np.array([0, 1]) for t in range(1, 50)}
    scores = {"a": {t: np.array([0.1, 0.9]) for t in range(1, 50)}}
    base = BaseScores(names=["a"], scores=scores, labels=labels)
    with pytest.raises(ValueError, match="validation window"):
        fit_stack(base, LEGACY_SPLIT, "logistic_regression", 0)


# ------------------------------------------------------------- thresholds
def test_oracle_f1_threshold_actually_attains_the_reported_f1():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(500) < 0.12).astype(np.int64)
        p = np.clip(0.25 * y + rng.random(500) * 0.75, 0, 1)
        value, thr = da.oracle_f1(y, p)
        attained = f1_score(y, (p >= thr).astype(int), zero_division=0)
        assert attained == pytest.approx(value, abs=1e-9)


def test_oracle_f1_is_the_maximum_over_all_thresholds():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.2).astype(np.int64)
    p = np.clip(0.3 * y + rng.random(300) * 0.7, 0, 1)
    value, _ = da.oracle_f1(y, p)
    brute = max(f1_score(y, (p >= c).astype(int), zero_division=0)
                for c in np.unique(p))
    assert value == pytest.approx(brute, abs=1e-9)


def test_oracle_f1_handles_a_degenerate_window():
    y = np.zeros(10, dtype=np.int64)
    assert da.oracle_f1(y, np.linspace(0, 1, 10)) == (0.0, 0.5)


# ------------------------------------------------------- causal enrichment
def _week_series(n_weeks=49, per_week=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(1, n_weeks + 1):
        for _ in range(per_week):
            rows.append({"t": t, **{f"f{i}": rng.normal(t * 0.1, 1.0)
                                    for i in range(10)}})
    return pd.DataFrame(rows)


def test_causal_wavelet_cannot_see_the_future():
    """Perturbing the test period must not change any earlier week."""
    cols = [f"f{i}" for i in range(10)]
    frame = _week_series()
    ref, _ = wavelet_features(frame, cols, causal=True)
    bumped = frame.copy()
    bumped.loc[bumped.t >= 42, cols] *= 5.0
    alt, _ = wavelet_features(bumped, cols, causal=True)
    before = ref[ref.t < 42].drop(columns="t").to_numpy()
    after = alt[alt.t < 42].drop(columns="t").to_numpy()
    assert np.allclose(before, after)


def test_non_causal_wavelet_does_leak():
    """The variant kept for comparison must reproduce the original leak."""
    cols = [f"f{i}" for i in range(10)]
    frame = _week_series()
    ref, _ = wavelet_features(frame, cols, causal=False)
    bumped = frame.copy()
    bumped.loc[bumped.t >= 42, cols] *= 5.0
    alt, _ = wavelet_features(bumped, cols, causal=False)
    before = ref[ref.t < 42].drop(columns="t").to_numpy()
    after = alt[alt.t < 42].drop(columns="t").to_numpy()
    assert not np.allclose(before, after)


def test_delta_features_use_only_the_previous_week():
    cols = [f"f{i}" for i in range(10)]
    frame = _week_series()
    ref, names = delta_features(frame, cols)
    bumped = frame.copy()
    bumped.loc[bumped.t >= 42, cols] *= 5.0
    alt, _ = delta_features(bumped, cols)
    mask = (frame.t < 42).to_numpy()
    assert np.allclose(ref.to_numpy()[mask], alt.to_numpy()[mask])
    assert names == [f"d_f{i}" for i in range(10)]


# ------------------------------------------------------------- statistics
def test_error_correlation_is_one_for_identical_learners():
    labels = {t: np.array([0, 1, 0, 1]) for t in (42, 43)}
    s = {t: np.array([0.1, 0.9, 0.8, 0.2]) for t in (42, 43)}
    base = BaseScores(names=["a", "b"], scores={"a": s, "b": dict(s)},
                      labels=labels)

    C, names, disagree = error_correlation(base, [42, 43])
    assert names == ["a", "b"]
    assert C[0, 1] == pytest.approx(1.0)
    assert disagree == pytest.approx(0.0)


def test_paired_seed_test_reports_wins_and_a_symmetric_p_value():
    out = da.paired_seed_test([1, 2, 3, 4], [0, 1, 2, 3])
    assert out["wins"] == 4 and out["n"] == 4
    assert out["median_diff"] == pytest.approx(1.0)
    assert out["sign_p"] == pytest.approx(0.125)
    tie = da.paired_seed_test([1, 2], [2, 1])
    assert tie["sign_p"] == pytest.approx(1.0)


def test_seed_summary_reports_the_spread_not_just_the_median():
    out = da.seed_summary([0.1, 0.2, 0.3, 0.4, 0.5])
    assert out["median"] == pytest.approx(0.3)
    assert out["min"] == pytest.approx(0.1) and out["max"] == pytest.approx(0.5)
    assert out["n_seeds"] == 5 and out["lo"] <= out["median"] <= out["hi"]


def test_calibration_is_perfect_for_a_perfectly_calibrated_score():
    rng = np.random.default_rng(3)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(np.int64)
    cal = da.calibration(y, p, n_bins=10)
    assert cal.ece < 0.02
    assert cal.brier == pytest.approx(np.mean((p - y) ** 2), abs=1e-9)


def test_prior_matched_predictions_flag_the_tracked_fraction():
    p = {42: np.linspace(0, 1, 100)}
    y = {42: np.array([1] * 10 + [0] * 90)}
    pred, yy = da.prior_matched_predictions(p, y, {42: 0.2}, [42])
    assert pred.sum() == pytest.approx(20, abs=1)
    assert yy.size == 100


def test_cost_curve_prefers_a_lower_threshold_as_misses_get_costlier():
    rng = np.random.default_rng(4)
    y = (rng.random(4000) < 0.05).astype(np.int64)
    p = np.clip(0.4 * y + rng.random(4000) * 0.6, 0, 1)
    cc = da.cost_curve(y, p, cost_ratios=(1, 100))
    assert cc["100"]["best_threshold"] <= cc["1"]["best_threshold"]
