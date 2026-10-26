# Reproducibility

## Canonical evidence chain

The primary source is `CAPSTONE_M6_RUN-20261026T000130Z-1-001.zip`, the newest full-run folder. It contains 61 files: 28 checkpoints, 22 CSV result tables, seven figures, the workbook, run log, stage log and `figures.zip`. The standalone workbook, run log and figures archive are byte-identical to the copies inside that folder.

The canonical notebook is the executed Colab file from the same run. All 18 code cells have execution counts, there are no error outputs, and its seven inline PNG outputs are pixel-identical to the seven PNGs in the run folder. Its 22 final displayed tables agree with the workbook and the original CSV exports. The executed HTML shows the same seven images and 22 final result tables.

The run started at 2026-10-25 23:53:45.614630 and finished at 23:55:41.138006, an elapsed 115.523 seconds. Data, return forecasts and covariance forecasts were loaded from checkpoints; the lag-one backtests, tests, extensions and hold-out were recomputed under `RUN_TAG = "lag1_v3_inline"`.

## Frozen choices

- Assets: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY.
- Validation years: 2004, 2005 and 2006.
- Test: January 2007 to December 2025.
- Hold-out: January to September 2026.
- Forecast horizon and input window: 5 and 60 trading days.
- Costs: 10 bps per side.
- Execution: close *t+1* after observing close *t*.
- Extension winner: rebalance every four weeks.

## Deterministic verification

```bash
python -m pytest -q
python scripts/verify_repo.py
python scripts/verify_delivery.py
```

These checks validate source syntax, full-run provenance, lag sensitivity, workbook schema and arithmetic, notebook execution and inline outputs, figure identity, report structure, audit presence and archive integrity. A clean model refit also depends on downloaded market data and the package versions recorded by the notebook and run log.
