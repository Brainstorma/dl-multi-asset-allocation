# Run Guide

## Review the completed run

1. Open `notebooks/Final_Capstone_Executed.ipynb`.
2. Confirm `QUICK_CHECK = False`, `PRIMARY_EXECUTION_LAG = 1` and `RUN_TAG = "lag1_v3_inline"`.
3. Inspect the final 22 tables and seven figures.
4. Open `outputs/current_lag1/M6_RESULTS_17863.xlsx` for the complete result register.

## Repeat the full study

```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/Final_Capstone_Executed.ipynb
```

Run all cells in order. A clean run requires network access to Yahoo Finance and FRED and a TensorFlow-capable environment. Completed checkpoints reduce repeat time, but a clean retraining may take materially longer than the archived continuation run.

## Verify the repository

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
python scripts/verify_repo.py
```
