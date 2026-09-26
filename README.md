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
config/config.yaml     seeds, universe, split dates, limits, cost model
notebooks/01_data_pull.ipynb    scripted yfinance + FRED pull, data-quality gate
notebooks/02_features.ipynb     feature matrix, outlier screen, date splits
notebooks/03_models.ipynb       [M3] LSTM return/covariance models, walk-forward
notebooks/04_allocate.ipynb     [M4] constrained optimizer, hybrid variant
notebooks/05_backtest.ipynb     [M5] backtest engine, metrics, sub-periods
notebooks/06_report.ipynb       [M6] exhibits for the final report
data/raw/, data/processed/      CSV/parquet store (never hand-edited)
results/                        RESULTS_MANIFEST exports
```

## Protocol pre-commitment

- Fixed seeds for numpy / random / tensorflow from `config/config.yaml`.
- Train/validation/test splits are date-based and set before any test-block run.
- Every number cited in the report must be reproducible from a notebook cell
  recorded in `results/RESULTS_MANIFEST`.
- Transaction-cost assumptions favor the benchmarks when in doubt.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Open items

- Frozen equity sleeve: AAPL, JPM, JNJ, XOM (committed at M2; MSFT dropped - overlap with AAPL duplicates XLK exposure).
- GitHub repository URL to be added when the repo goes live.
