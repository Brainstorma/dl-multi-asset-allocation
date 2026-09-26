"""Cleaning policy shared by the feature pipeline and its tests.

The winsorization rule pre-committed in the M2 problem statement is applied
*per asset, per feature block*: clipping cross-sectionally at a single global
quantile would let one volatile name (e.g. XOM around an earnings gap) drag
the clip bounds of every quiet sector ETF toward its own tails.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


def winsorize(frame: pd.DataFrame, q_lo: float, q_hi: float) -> pd.DataFrame:
    """Clip each column of ``frame`` independently to its [q_lo, q_hi] quantiles."""
    lo = frame.quantile(q_lo)
    hi = frame.quantile(q_hi)
    return frame.clip(lower=lo, upper=hi, axis=1)


def winsorize_with_bounds(
    frame: pd.DataFrame,
    q_lo: float,
    q_hi: float,
    fit: "pd.Series | slice | None" = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Clip every column to quantiles estimated on a *fit* subset only.

    ``fit`` selects the rows whose distribution defines the bounds (typically
    a boolean mask for the training block). Passing an out-of-sample subset
    is the point: computing quantiles over the full sample leaks future
    tails into features used for fitting. Returns ``(clipped, lo, hi)`` so
    callers can record or reuse the fitted bounds.
    """
    ref = frame if fit is None else frame.loc[fit]
    lo = ref.quantile(q_lo)
    hi = ref.quantile(q_hi)
    return frame.clip(lower=lo, upper=hi, axis=1), lo, hi


def set_seed(seed: int) -> None:
    """Fix python/``numpy`` generators from a single pinned value.

    Called by notebooks 01-02 (and anything downstream) before any data is
    touched, so the reproducibility pre-commitment in ``config/config.yaml``
    is enforced by one shared code path rather than copy-pasted cell snippets.
    TensorFlow's graph-level seed is set separately in notebook 03, which is
    the first stage that builds models.
    """
    random.seed(seed)
    np.random.seed(seed)


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
