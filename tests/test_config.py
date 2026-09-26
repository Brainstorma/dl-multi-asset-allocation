"""Structural validation of the single source of truth: config/config.yaml.

These tests are the executable form of the protocol pre-commitment: if a
split date, seed, or limit drifts, CI fails before any experiment runs.
"""

from datetime import date

import pytest

from src.config import CONFIG_PATH, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_config_file_found():
    assert CONFIG_PATH.exists(), "config/config.yaml missing from repo root"


def test_seeds_pinned(cfg):
    assert set(cfg["seeds"]) == {"numpy", "random", "tensorflow"}
    assert all(isinstance(v, int) for v in cfg["seeds"].values())


def test_universe_has_no_duplicate_tickers(cfg):
    tickers = cfg["universe"]["etfs"] + cfg["universe"]["equity_sleeve"]
    assert len(tickers) == len(set(tickers)), "ticker appears in both sleeves"


def test_splits_are_ordered_and_non_overlapping(cfg):
    s = cfg["splits"]

    def d(key):
        return date.fromisoformat(s[key])

    assert d("train_start") <= d("train_end") < d("valid_start")
    assert d("valid_start") <= d("valid_end") < d("test_start")


def test_cleaning_windows_positive(cfg):
    assert cfg["cleaning"]["ffill_limit"] >= 1
    assert cfg["cleaning"]["max_missing_days"] >= 0
    lo, hi = cfg["cleaning"]["winsor"]
    assert 0 <= lo < 0.5 < hi <= 1


def test_allocation_limits_coherent(cfg):
    alloc = cfg["allocation"]
    n = len(cfg["universe"]["etfs"]) + len(cfg["universe"]["equity_sleeve"])
    assert 0 < alloc["w_max"] <= 1
    assert alloc["w_max"] * n >= 1, "bounds make it impossible to invest fully"
    assert 0 < alloc["turnover_cap"] <= 2
    assert alloc["gamma"] > 0
    assert 0 <= alloc["lambda_shrink_blend"] <= 1


def test_costs_nonnegative(cfg):
    assert cfg["backtest"]["cost_per_side"] >= 0
