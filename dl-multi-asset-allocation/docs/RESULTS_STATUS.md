# Results status

- **Primary source:** `CAPSTONE_M6_RUN-20261026T000130Z-1-001.zip`, modified 26 October 2026 at 00:01.
- **Source execution:** full Colab run from 23:53:45 to 23:55:41 on 25 October 2026.
- **Run tag:** `lag1_v3_inline`.
- **Primary convention:** one-trading-day execution lag across validation, test and hold-out.
- **Diagnostic convention:** same-close results remain only in `metrics_same_close_diagnostic` and `execution_timing_compare`.
- **Headline test block:** 992 weekly rebalances from January 2007 to December 2025.
- **Hold-out:** 39 weekly rebalances from January to September 2026.
- **Selected extension:** `RB_every4`, frozen from the 2004-2006 validation folds before test evaluation.
- **Report status:** final M6 draft completed from the newest full-run evidence.

The final executed notebook has 18 executed code cells, no error outputs, seven inline PNG outputs and 22 final HTML tables. A 91-check consistency pass confirmed that the workbook agrees with all 22 original CSV tables and that the notebook and executed HTML contain the same seven figures and 22 final result tables. The workbook's headline numbers are unchanged from the corrected lag-one evidence used in the report.

Release gates passed: Capstone results check, number trace, submission lint, DOCX accessibility and visual rendering, repository tests, package integrity and download publication.
