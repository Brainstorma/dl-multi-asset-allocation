# Deep Learning for Dynamic Multi-Asset Portfolio Allocation

MScFE 690 Capstone, Group 17863: Joseph Amoako-Asare and Isaac Asirifi.

This repository evaluates whether LSTM return and covariance forecasts improve a constrained allocation across nine US sector ETFs after position limits, a 10% volatility ceiling, turnover limits and transaction costs. The Milestone 6 release contains the final draft report, the fully executed `lag1_v3_inline` notebook, reusable `m6core`, deterministic tests, configuration, the latest run tables and all figures.

## Milestone 6 result

The primary convention forms forecasts with information available at close *t* and executes at close *t+1*.

| Item | Lag-one result |
|---|---:|
| Full-design net Sharpe | 0.305 |
| Equal-weight net Sharpe | 0.528 |
| Shrinkage minimum-variance net Sharpe | 0.522 |
| Full-design maximum drawdown | -28.8% |
| Validation-selected extension | Rebalance every four weeks |
| Selected-extension test Sharpe | 0.260 |
| Full-design 2026 hold-out Sharpe | 0.509 |

H1, H2 and H3 are rejected. The full design trails the named benchmarks, its drawdown advantage disappears against risk-matched portfolios, and the stress-week coefficient is not positive and significant. Same-close results remain only in the labelled timing-sensitivity sheet.

## Final run provenance

The canonical notebook in `notebooks/` is the executed Colab notebook from 25 October 2026. It has 18 executed code cells, no error outputs, seven inline PNG outputs and 22 final result tables. The matching workbook identifies the run as `lag1_v3_inline`, records a one-trading-day execution lag and covers prices through 1 October 2026. Its 22 result sheets agree with the original CSV exports in the newest full-run folder.

## Repository layout

- `notebooks/`: fully executed revised Colab notebook.
- `src/`: reusable pipeline module.
- `config/`: frozen study and execution controls.
- `tests/`: deterministic tests that require no data download.
- `figures/`: the seven PNGs displayed by the executed notebook.
- `outputs/current_lag1/`: final results workbook, run log and 22 original CSV tables.
- `docs/`: final M6 report, PDF, audit, result status and reproducibility notes.
- `scripts/`: repository, delivery and release verification.

The separate TEAM_ONLY package contains the original newest-run archive, executed HTML, configuration, verification, assistance and REAP evidence. It is excluded from the GitHub release and academic SUBMISSION package.

## Reproduce

Open `notebooks/M6_FULL_RUN_Group17863_REVISED_COLAB.ipynb` in Google Colab and run all cells with `QUICK_CHECK = False`, `PRIMARY_EXECUTION_LAG = 1` and `RUN_TAG = "lag1_v3_inline"`. The notebook writes `M6_RESULTS_17863.xlsx`, `m6_run_log.md`, `figures.zip`, seven PNGs and 22 CSV result tables.

## Local verification

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
python scripts/verify_repo.py
python scripts/build_release.py
python scripts/verify_delivery.py
```

A clean full run additionally needs the packages in `requirements.txt` and network access to Yahoo Finance and FRED.

## Reproducibility controls

- Expanding training uses only data available before each forecast year.
- The final five targets are purged at every annual boundary.
- Extension settings are selected on 2004-2006 only.
- Test returns start one trading day after the decision close.
- Choices are frozen before test and hold-out evaluation.
- All 21 validation variants are reported.
- Hypothesis-test checkpoints are versioned by `RUN_TAG` to prevent cross-run cache reuse.

## Disclaimer

This is academic research, not investment advice. Results are historical and model-dependent.
