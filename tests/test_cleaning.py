"""Unit tests for the shared cleaning policy in src/cleaning.py."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning import calendar_align, winsorize


def test_winsorize_clips_only_the_tails():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=10_000), name="X")
    out = winsorize(s.to_frame(), 0.01, 0.99)["X"]
    assert out.min() >= s.quantile(0.01) - 1e-12
    assert out.max() <= s.quantile(0.99) + 1e-12
    # interior points untouched
    interior = (s > s.quantile(0.01)) & (s < s.quantile(0.99))
    pd.testing.assert_series_equal(s[interior], out[interior], check_names=False)


def test_winsorize_is_per_column_not_cross_sectional():
    # A wild column must not widen the clip bounds of the tame one.
    df = pd.DataFrame(
        {
            "tame": [0.0] * 98 + [-9.0, 9.0],
            "wild": [0.0] * 98 + [-900.0, 900.0],
        }
    )
    out = winsorize(df, 0.01, 0.99)
    assert abs(out["tame"]).max() < 1.0, "tame column was clipped by wild column's scale"
    assert abs(out["wild"]).max() < 900.0


def test_winsorize_preserves_shape_and_index():
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(50, 3)), columns=list("ABC"))
    out = winsorize(df, 0.05, 0.95)
    assert out.shape == df.shape
    pd.testing.assert_index_equal(out.index, df.index)


def test_calendar_align_respects_ffill_limit():
    idx = pd.bdate_range("2024-01-01", periods=10)

    # 3-day gap, limit 5: every NaN is within reach of the last valid value.
    prices = pd.DataFrame({"A": np.arange(10.0)}, index=idx)
    prices.loc[idx[3]:idx[5], "A"] = np.nan
    macro = pd.DataFrame({"VIX": np.arange(10.0)}, index=idx)
    p, _ = calendar_align(prices, macro, ffill_limit=5)
    assert p["A"].notna().all(), "gap within the ffill limit should be filled"
    assert len(p) == 10

    # 8-day interior gap, limit 5: the unfillable rows stay all-NaN and are
    # therefore dropped by the row filter - no magic price creation, and no
    # silently mispriced bars carried into the feature matrix.
    prices2 = pd.DataFrame({"A": np.arange(10.0)}, index=idx)
    prices2.loc[idx[1]:idx[8], "A"] = np.nan
    p2, _ = calendar_align(prices2, macro, ffill_limit=5)
    assert p2["A"].notna().all()
    assert len(p2) == 7
    assert list(p2.index[[0, -1]]) == [idx[0], idx[9]]


def test_calendar_align_does_not_mutate_inputs():
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=idx)
    macro = pd.DataFrame({"V": [1.0, 2.0, np.nan, np.nan, 5.0]}, index=idx)
    snapshot = prices.copy()
    calendar_align(prices, macro, ffill_limit=1)
    pd.testing.assert_frame_equal(prices, snapshot)
