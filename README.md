# Deep Learning for Dynamic Multi-Asset Portfolio Allocation

MScFE 690 Capstone, Group 17863: Joseph Amoako-Asare and Isaac Asirifi.

## Research question

We test whether LSTM return and covariance forecasts improve a constrained allocation across nine US sector ETFs after a one-trading-day execution lag, position and turnover limits, a 10% volatility ceiling and transaction costs of 10 basis points per side.

Model settings are selected on the 2004–2006 validation folds and frozen before the 2007–2025 test. An untouched January–September 2026 block provides a final directional check.

## Final results

| Item | Result |
|---|---:|
| Full-design net Sharpe | 0.305 |
| Equal-weight net Sharpe | 0.528 |
| Shrinkage minimum-variance net Sharpe | 0.522 |
| Full-design maximum drawdown | -28.8% |
| Full-design weekly turnover | 17.8% |
| Pooled forecast hit rate | 52.4% |
| Empirical always-up base rate | 56.1% |
| Validation-selected extension | Rebalance every four weeks |
| Selected-extension test Sharpe | 0.260 |
| Full-design 2026 hold-out Sharpe | 0.509 |

H1, H2 and H3 are rejected. The return forecast falls short of its empirical base rate and has no useful cross-sectional ranking signal. Validation assigns a shrinkage weight of one, so the neural covariance head does not enter headline portfolio decisions. DCC improves Frobenius covariance loss but raises turnover and reduces net Sharpe relative to shrinkage minimum variance. The central result is that forecast skill, risk-model accuracy and implementation cost must be evaluated as one decision chain.

## Repository layout

- `notebooks/Final_Capstone_Executed.ipynb`: executed end-to-end notebook with 18 code cells, seven figures and 22 final result tables.
- `src/m6core.py`: reusable data, forecasting, allocation, testing and export functions.
- `config/m6.yaml`: frozen study and execution controls for the final run.
- `tests/`: deterministic tests that require no data download.
- `figures/`: architecture, timing and five result figures.
- `outputs/current_lag1/`: 24-sheet results workbook, run log and 22 source CSV tables.
- `docs/`: final report, one-page Solution Design Document, reproducibility notes and run guide.
- `scripts/verify_repo.py`: cross-file verification of notebook, figures, workbook, CSV tables and report structure.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab notebooks/Final_Capstone_Executed.ipynb
```

Run all notebook cells with:

```python
QUICK_CHECK = False
PRIMARY_EXECUTION_LAG = 1
RUN_TAG = "lag1_v3_inline"
```

A clean full run needs network access to Yahoo Finance and FRED. The repository already includes the executed notebook and final exported evidence so the reported results can be inspected without repeating the download and training stages.

## Local verification

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
python scripts/verify_repo.py
```

The verifier checks code syntax, notebook execution state, inline figures, workbook-to-CSV equality, frozen choices, timing conventions, report structure and the absence of internal process material.

## Reproducibility controls

- Expanding training uses only data available before each forecast year.
- The final five targets are purged at every annual refit boundary.
- Extension settings are selected on 2004–2006 only.
- Test returns begin one trading day after the decision close.
- All choices are frozen before test and hold-out evaluation.
- All 21 extension variants are reported.
- Hypothesis-test checkpoints are versioned by `RUN_TAG`.
- Every strategy receives identical constraints and costs.

## Authors

- Joseph Amoako-Asare: research design, validation, covariance modelling, analysis and writing.
- Isaac Asirifi: data pipeline, training loop, allocation and backtest code, testing and reproducibility.

## Disclaimer

This repository is academic research, not investment advice. Results are historical and model-dependent.
