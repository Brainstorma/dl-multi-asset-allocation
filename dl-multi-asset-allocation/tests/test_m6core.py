from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import m6core as m


def test_holding_returns_lag_one_starts_next_close():
    idx = pd.bdate_range("2026-01-02", periods=12)
    px = pd.DataFrame({"A": np.arange(100.0, 112.0)}, index=idx)
    dates = pd.DatetimeIndex([idx[0], idx[5]])
    out = m.holding_returns(px, dates, lag=1)
    assert np.isclose(out.iloc[0, 0], px.iloc[6, 0] / px.iloc[1, 0] - 1)


def test_turnover_is_one_way():
    a = np.array([0.6, 0.3, 0.1])
    b = np.array([0.4, 0.5, 0.1])
    assert np.isclose(m.turnover(a, b), 0.2)


def test_extension_grid_is_frozen_at_21():
    grid = m.extension_grid()
    assert len(grid) == 21
    assert len({x.name for x in grid}) == 21


def test_summary_calmar_identity():
    idx = pd.date_range("2020-01-03", periods=100, freq="W-FRI")
    r = np.r_[np.full(50, 0.002), np.full(50, -0.001)]
    ts = pd.DataFrame({"net": r, "excess": r, "turnover": 0.0, "cash": 0.0,
                       "n_holdings": 3, "turnover_cap_binds": False,
                       "vol_ceiling_binds": False, "status": "optimal"}, index=idx)
    s = m.summarise(ts, "x")
    assert np.isclose(s["calmar"], s["ann_return"] / abs(s["max_dd"]))


def test_holding_returns_changes_under_lag_one():
    idx = pd.bdate_range("2020-01-01", periods=20)
    prices = pd.DataFrame({"A": np.arange(1.0, 21.0), "B": np.arange(21.0, 41.0)}, index=idx)
    dates = pd.DatetimeIndex([idx[4], idx[9], idx[14]])
    same_close = m.holding_returns(prices, dates, lag=0)
    lag_one = m.holding_returns(prices, dates, lag=1)
    assert same_close.shape == lag_one.shape
    assert not np.allclose(same_close.to_numpy(), lag_one.to_numpy(), equal_nan=True)
