# M6 Colab run guide

## Verified final execution

The release uses the full Colab run completed on 25 October 2026 with:

```python
QUICK_CHECK = False
PRIMARY_EXECUTION_LAG = 1
RUN_TAG = "lag1_v3_inline"
RUN_SEED_ROBUSTNESS = True
```

The run completed all 18 code cells without an error output and wrote the final workbook, run log, figures archive, seven PNGs and 22 CSV tables. The executed notebook displays all seven figures and all 22 final tables inline.

## Re-run procedure

1. Open `notebooks/M6_FULL_RUN_Group17863_REVISED_COLAB.ipynb` in Google Colab.
2. Mount the project Drive location used by the notebook.
3. Confirm the controls above.
4. Select **Runtime > Run all**.
5. Keep the output folder, executed notebook and generated files together.

## Completion checks

1. Every code cell has an execution count and no code cell ends with an error output.
2. The run log records `mode: full`, `run_tag: lag1_v3_inline` and `execution_lag_days: 1`.
3. The workbook contains 24 sheets, including `RUN_LOG`, `metrics_test`, `execution_timing_compare`, `validation_leaderboard` and `holdout_2026`.
4. `metrics_test` has 992 weeks for every strategy; `holdout_2026` has 39.
5. Seven PNG outputs and 22 final result tables appear inline.
6. The full-design lag-one and same-close Sharpe ratios differ in `execution_timing_compare`.

Do not replace this release with a quick-check run. Quick-check outputs test execution only and are not report evidence.
