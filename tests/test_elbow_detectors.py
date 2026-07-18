"""Unit tests for the three elbow detectors in analysis.elbow_topk.

The detectors are pure functions of a 1-D array, so they are tested here
without touching H5Dataset or any embedding file. Reference curves:

- `harmonic_curve(N)` = 1 / (1 + arange(N)) -- convex-decreasing, dominates
  early. Provides analytically tractable cumulative-sum expectations.
- `linear_curve(N)` -- kneedle chord matches the curve exactly, so
  perpendicular distance is 0 everywhere; used as a degenerate-input
  smoke test.
- `constant_curve(N)` -- max == min, exercises the zero-range branch in
  every detector.
- `sharp_elbow_curve(...)` -- piecewise-linear with an obvious knee at a
  known index; kneedle should land on the corner.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.elbow_topk import (
    detect_cumulative,
    detect_kneedle,
    detect_second_deriv,
)


def harmonic_curve(n: int) -> np.ndarray:
    return 1.0 / (1.0 + np.arange(n, dtype=np.float64))


def linear_curve(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, dtype=np.float64)


def constant_curve(n: int, value: float = 1.0) -> np.ndarray:
    return np.full(n, value, dtype=np.float64)


def sharp_elbow_curve(head_len: int = 10, tail_len: int = 40) -> np.ndarray:
    head = np.linspace(10.0, 5.0, head_len)
    tail = np.linspace(5.0, 0.5, tail_len)
    return np.concatenate([head, tail])


class TestDetectKneedle:
    def test_returns_1_indexed_position(self):
        y = harmonic_curve(100)
        k = detect_kneedle(y)
        assert 1 <= k <= len(y)

    def test_harmonic_curve_early_elbow(self):
        # Convex-decreasing curve: the kneedle should land in the front
        # quarter, well below any "middle of curve" fallback.
        y = harmonic_curve(100)
        assert detect_kneedle(y) < 25

    def test_sharp_elbow_lands_on_corner(self):
        # Piecewise-linear with a corner at index 9 (1-indexed k=10).
        y = sharp_elbow_curve(head_len=10, tail_len=40)
        k = detect_kneedle(y)
        assert abs(k - 10) <= 1

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_linear_curve_does_not_crash(self, n):
        # Perpendicular distance to the endpoint chord is ~0 everywhere;
        # argmax is ill-defined but must return a valid 1-indexed k.
        y = linear_curve(n)
        k = detect_kneedle(y)
        assert 1 <= k <= n

    def test_constant_curve_returns_n(self):
        y = constant_curve(20)
        assert detect_kneedle(y) == len(y)

    def test_singleton(self):
        assert detect_kneedle(np.array([1.0])) == 1

    def test_empty(self):
        assert detect_kneedle(np.array([])) == 0


class TestDetectCumulative:
    def test_harmonic_curve_half_threshold(self):
        # cumsum of 1/(1+i) reaches 0.5 * H_100 at k=8 (checked numerically).
        y = harmonic_curve(100)
        assert detect_cumulative(y, 0.5) == 8

    def test_harmonic_curve_0_9_threshold(self):
        y = harmonic_curve(100)
        assert detect_cumulative(y, 0.9) == 60

    def test_threshold_1_returns_n(self):
        y = harmonic_curve(50)
        assert detect_cumulative(y, 1.0) == len(y)

    def test_threshold_above_1_returns_n(self):
        # Unreachable threshold falls back to n rather than an OOB index.
        y = harmonic_curve(50)
        assert detect_cumulative(y, 1.5) == len(y)

    def test_threshold_0_returns_1(self):
        y = harmonic_curve(50)
        assert detect_cumulative(y, 0.0) == 1

    def test_constant_curve(self):
        # Uniform mass: k / N >= threshold -> k = ceil(threshold * N).
        y = constant_curve(10)
        assert detect_cumulative(y, 0.5) == 5
        assert detect_cumulative(y, 0.75) == 8

    def test_all_zero_curve_returns_n(self):
        assert detect_cumulative(np.zeros(10), 0.5) == 10

    def test_empty(self):
        assert detect_cumulative(np.array([]), 0.5) == 0


class TestDetectSecondDeriv:
    def test_harmonic_curve_unsmoothed(self):
        # y'' = 2 / (1 + i)^3 is maximal at i = 0 -> +2 offset gives k=2.
        y = harmonic_curve(100)
        assert detect_second_deriv(y, smoothing_sigma=0.0) == 2

    def test_harmonic_curve_smoothed_stays_early(self):
        # With sigma=2 smoothing, the peak shifts slightly right but stays
        # in the front decile of a 100-point convex curve.
        y = harmonic_curve(100)
        k = detect_second_deriv(y, smoothing_sigma=2.0)
        assert 1 < k < 15

    def test_sharp_elbow_near_corner(self):
        # The second derivative peaks near the corner of the piecewise curve.
        y = sharp_elbow_curve(head_len=10, tail_len=40)
        k = detect_second_deriv(y, smoothing_sigma=1.0)
        assert abs(k - 10) <= 3

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_linear_curve_does_not_crash(self, n):
        y = linear_curve(n)
        k = detect_second_deriv(y)
        assert 1 <= k <= n

    def test_constant_curve_does_not_crash(self):
        y = constant_curve(20)
        k = detect_second_deriv(y)
        assert 1 <= k <= len(y)

    def test_short_curve(self):
        assert detect_second_deriv(np.array([1.0, 0.5])) == 2

    def test_empty(self):
        assert detect_second_deriv(np.array([])) == 0
