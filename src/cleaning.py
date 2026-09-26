"""Cleaning policy shared by the feature pipeline and its tests.

The winsorization rule pre-committed in the M2 problem statement is applied
*per asset, per feature block*: clipping cross-sectionally at a single global
quantile would let one volatile name (e.g. XOM around an earnings gap) drag
the clip bounds of every quiet sector ETF toward its own tails.
"""

from __future__ import annotations

import pandas as pd


def winsorize(frame: pd.DataFrame, q_lo: float, q_hi: float) -> pd.DataFrame:
    """Clip each column of ``frame`` independently to its [q_lo, q_hi] quantiles."""
    lo = frame.quantile(q_lo)
    hi = frame.quantile(q_hi)
    return frame.clip(lower=lo, upper=hi, axis=1)


def calendar_align(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    ffill_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort on the index and forward-fill gaps up to ``ffill_limit`` days.

    Returns copies; the raw frames on disk are never mutated in place.
    """
    prices = prices.sort_index().ffill(limit=ffill_limit).dropna(how="all")
    macro = macro.sort_index().ffill(limit=ffill_limit)
    return prices, macro
