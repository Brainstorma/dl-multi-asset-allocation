# Reproducibility

The final evidence chain uses run tag `lag1_v3_inline` and a one-trading-day execution lag. The executed notebook contains 18 executed code cells, seven inline PNG figures, 42 HTML outputs and no error output. Its final export cell displays the 22 result tables contained in `outputs/current_lag1/M6_RESULTS_17863.xlsx`.

## Frozen boundaries

- Expanding training starts in March 1999.
- Validation folds are 2004, 2005 and 2006.
- The primary test contains 992 weekly observations from 2007 through 2025.
- The untouched hold-out contains 39 weekly observations from January through September 2026.
- Extension parameters are selected on validation only.
- Execution occurs one trading day after the decision close.

## Evidence map

| Evidence | Location |
|---|---|
| Executed computation | `notebooks/Final_Capstone_Executed.ipynb` |
| Reusable study logic | `src/m6core.py` |
| Frozen settings | `config/m6.yaml` |
| Consolidated results | `outputs/current_lag1/M6_RESULTS_17863.xlsx` |
| Original result tables | `outputs/current_lag1/tables/` |
| Run status | `outputs/current_lag1/m6_run_log.md` |
| Figures | `figures/` |
| Final report | `docs/MScFE_690_Final_Capstone_Report_Group17863.pdf` |

Run `python scripts/verify_repo.py` after extraction. It compares all 22 workbook result sheets with their CSV sources, verifies the notebook figures against the PNG files and checks the frozen result identities used in the report.
