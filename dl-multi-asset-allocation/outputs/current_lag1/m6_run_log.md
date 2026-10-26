# M6 run log

- group: 17863
- milestone: M6
- mode: full
- started: 2026-10-25 23:53:45.614630
- finished: 2026-10-25 23:55:41.138006
- gpu: ['/physical_device:GPU:0']
- tensorflow: 2.21.0
- cvxpy: True
- execution_lag_days: 1
- run_tag: lag1_v3_inline
- price_end: 2026-10-01

| stage | seconds |
|---|---|
| 01_data | 0.0 |
| 02_return_forecasts_seed42 | 0.0 |
| 03_cov_forecasts | 0.0 |
| 02_return_forecasts_seed7 | 0.0 |
| 02_return_forecasts_seed2026 | 0.0 |
| 04_cov_lw | 0.0 |
| 04_cov_dcc | 0.0 |
| 05_backtest_lag1_v3_inline | 28.7 |
| 05_backtest_same_close_diagnostic_v2 | 0.0 |
| 05_backtest_0bps_lag1_v3_inline | 9.8 |
| 05_backtest_seed7_lag1_v3_inline | 4.0 |
| 05_backtest_seed2026_lag1_v3_inline | 4.2 |
| 06_tests_lag1_v3_inline | 0.6 |
| 07_extensions_lag1_v3_inline | 49.6 |
| 08_holdout_lag1_v3_inline | 1.0 |