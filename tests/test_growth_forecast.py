"""Tests for Growth forecast and Warnings (mycodemo.growth.forecast), using made-up curves."""

import numpy as np
import pytest

from mycodemo.growth.forecast import (
    day_reaching,
    forecast,
    forecast_by_photo,
    forecast_warnings,
    logistic,
)

T = np.arange(0, 15, 1.0)
K, T50 = 0.8, 6.0
C = logistic(T, K, T50)


def test_logistic_midpoint():
    assert logistic(T50, K, T50) == pytest.approx(0.5)


def test_day_reaching_matches_formula():
    # 90% day of a logistic: t50 + ln(9) / k
    assert day_reaching(logistic, [K, T50], 0.90) == pytest.approx(T50 + np.log(9) / K, abs=0.01)


def test_day_reaching_never():
    assert np.isnan(day_reaching(lambda t, a: 0.5 * np.ones_like(t), [1.0], 0.90))


def test_forecast_recovers_true_days():
    r = forecast(T, C)
    assert r["day90"] == pytest.approx(T50 + np.log(9) / K, abs=0.1)
    assert r["day95"] == pytest.approx(T50 + np.log(19) / K, abs=0.1)
    assert r["day95"] > r["day90"]


def test_forecast_too_few_points_fails_gracefully():
    r = forecast([1.0], [0.1])
    assert r["params"] is None and np.isnan(r["day90"])


def test_forecast_by_photo_has_one_row_per_photo():
    h = forecast_by_photo(T, C)
    assert list(h["n_photos"]) == list(range(2, len(T) + 1))


def test_warning_few_photos():
    r = forecast(T[:2], C[:2])
    assert any("photo" in w for w in forecast_warnings(T[:2], C[:2], r))


def test_warning_never_reaches_90():
    c = np.full(len(T), 0.05)  # dish that stays almost empty
    r = forecast(T, c)
    w = forecast_warnings(T, c, r)
    assert w  # something is flagged


def test_no_warning_for_good_series():
    assert forecast_warnings(T, C, forecast(T, C)) == []
