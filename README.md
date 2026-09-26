# Deep Learning for Dynamic Multi-Asset Portfolio Allocation

MScFE 690 Capstone - Track 8, Topic 2. Team: Joseph Amoako-Asare (901774),
Isaac Asirifi (881355).

## Research question

Do portfolio weights driven by LSTM return and covariance forecasts earn higher
out-of-sample risk-adjusted returns than classical allocation rules (equal-weight,
mean-variance with Ledoit-Wolf shrinkage, risk parity) on a universe of liquid US
sector ETFs and large-cap equities?

## Repository structure

```
config/config.yaml          seeds, universe, split dates, limits, cost model
src/                        shared library used by the notebooks AND the tests
  config.py                   path-safe YAML loader (single source of truth)
  cleaning.py                   winsorization + calendar-alignment policy
notebooks/01_data_pull.ipynb    scripted yfinance + FRED pull, data-quality gate
notebooks/02_features.ipynb     feature matrix, leakage guards, date splits
notebooks/03_models.ipynb       [M3] LSTM return/covariance models, walk-forward
notebooks/04_allocate.ipynb     [M4] constrained optimizer, hybrid variant
notebooks/05_backtest.ipynb     [M5] backtest engine, metrics, sub-periods
notebooks/06_report.ipynb       [M6] exhibits for the final report
tests/                      pytest suite (config validation, cleaning unit tests,
                            notebook static checks) - runs in CI on every push
data/raw/, data/processed/      CSV/parquet store (never hand-edited, gitignored;
                                regenerate with notebook 01 -> 02)
results/                    RESULTS_MANIFEST exports
```

## Pipeline

Run the notebooks in numeric order from within `notebooks/` (Jupyter Lab sets
the working directory there; all paths resolve through `src.config.project_root()`
so the same code also works from the repo root):

1. **01_data_pull** - downloads adjusted closes (10 sector SPDRs + SPY + the
   frozen equity sleeve) and five FRED macro series into `data/raw/`, then
   applies a hard data-quality gate (asserts before anything downstream runs).
2. **02_features** - builds returns / rolling vol / momentum, joins a
   point-in-time macro block, winsorizes tails per column, writes the parquet
   feature store plus a machine-readable cleaning log, and cuts the dated
   train/validation/test blocks.
3. **03-06** - models, allocation, backtest, report (M3-M6 milestones).

## Leakage guards (enforced in code, not just prose)

- **Point-in-time macro data.** FRED series carry publication lags (CPIAUCSL
  by weeks); notebook 02 shifts the whole macro block by a configurable
  business-day lag (`features.macro_lag_bdays`) before joining to prices.
- **T+1 convention.** Signals at day *t* are tradeable at *t+1*; the supervised
  target is the next-day return, so no row peeks at its own close-to-close move.
- **Per-column winsorization.** One volatile name cannot drag the clip bounds
  of every quiet sector ETF toward its own tails.
- **Bounded forward-fill.** Gaps longer than `ffill_limit` stay NaN and are
  dropped rather than silently invented.

## Protocol pre-commitment

- Fixed seeds for numpy / random / tensorflow from `config/config.yaml`.
- Train/validation/test splits are date-based and set before any test-block run.
- Every number cited in the report must be reproducible from a notebook cell
  recorded in `results/RESULTS_MANIFEST`.
- Transaction-cost assumptions favor the benchmarks when in doubt.
- The protocol itself is now executable: `tests/test_config.py` fails CI if a
  seed unpins, a split overlaps, or an allocation limit goes incoherent.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                              # guardrail suite, ~1s, no network needed
jupyter lab                         # then run notebooks/01 -> 02 -> ...
```

CI (`.github/workflows/ci.yml`) installs the full pinned environment and runs
`pytest` on every push and pull request.

## Open items

- Frozen equity sleeve: AAPL, JPM, JNJ, XOM (committed at M2; MSFT dropped -
  overlap with AAPL duplicates XLK exposure).
- Per-series macro release lags to replace the single conservative default at
  M3 (VIXCLS/DFF need far less lag than CPIAUCSL).
- GitHub repository URL to be added when the repo goes live.
