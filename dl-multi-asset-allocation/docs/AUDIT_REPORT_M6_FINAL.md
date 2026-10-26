# M6 final audit - Group 17863

**Project:** Deep Learning for Dynamic Multi-Asset Portfolio Allocation  
**Members:** Joseph Amoako-Asare and Isaac Asirifi  
**Audit date:** 26 October 2026

## Executive verdict

**PASS for Milestone 6 delivery.** The release now uses the newest full-run folder, `CAPSTONE_M6_RUN-20261026T000130Z-1-001.zip`, together with the matching executed notebook, executed HTML, workbook, run log and figures. The earlier requirement to rerun `lag1_v3_inline` is closed. The final run confirms the corrected lag-one presentation layer without changing the report's quantitative findings.

The run archive, standalone artifacts, executed notebook and HTML were checked as one evidence chain. A 91-check consistency program reported 91 passed and zero failed. The results gate, report number trace, submission lint, accessibility audit, visual render, repository tests and archive checks pass.

## 1. Primary-source control

| Source | Verification | Result |
|---|---|---|
| Newest full-run folder | 61 files: 28 checkpoints, 22 CSV tables, seven PNGs, workbook, run log, stage log and figures archive | Pass |
| Standalone workbook | Byte-identical to the workbook inside the newest run folder | Pass |
| Standalone run log | Byte-identical to the log inside the newest run folder | Pass |
| Standalone `figures.zip` | Byte-identical to the archive inside the newest run folder | Pass |
| Executed notebook | 18/18 code cells executed; zero error outputs | Pass |
| Executed HTML | Local Chromium render shows 18 code cells, seven images, 42 tables and zero notebook error outputs | Pass |
| Older runs | Not used for the final result register or figures | Pass |

The run began at 2026-10-25 23:53:45.614630 and ended at 23:55:41.138006, an elapsed 115.523 seconds. Data, return-forecast and covariance checkpoints were reused; the lag-one backtests, hypothesis tests, extensions and hold-out were recomputed under `RUN_TAG = "lag1_v3_inline"`.

## 2. Cross-file consistency

| Check | Scope | Result |
|---|---|---|
| Workbook versus original CSVs | All 22 result sheets against all 22 CSV exports | 22/22 pass |
| Notebook figures | Seven inline PNG outputs against the seven run-folder PNGs | 7/7 exact byte or pixel match |
| Executed-HTML figures | Seven embedded PNGs against the seven run-folder PNGs | 7/7 exact byte or pixel match |
| Notebook final tables | 22 displayed result tables against workbook sheets | 22/22 pass |
| Executed-HTML final tables | Final 22 displayed tables against workbook sheets | 22/22 pass |
| Workbook structure | 24 sheets, no formulas, no cross-sheet links or VBA | Pass |
| Source module | Notebook's `m6core.py` writefile cell against repository `src/m6core.py` | Exact match |

The executed HTML requests local RequireJS and MathJax resources and an external analytics script that are unavailable in the offline sandbox. These generated console messages do not affect the embedded figures, tables or notebook text, all of which rendered and were independently matched to the primary run files.

## 3. Results gate and headline register

| Item | Final value |
|---|---:|
| Full-design net Sharpe, 2007-2025 | 0.304979 |
| Equal-weight net Sharpe | 0.528081 |
| Shrinkage GMV net Sharpe | 0.521721 |
| Full-design maximum drawdown | -28.778% |
| Selected validation extension | `RB_every4` |
| Selected-extension test Sharpe | 0.260006 |
| Full-design 2026 hold-out Sharpe | 0.508658 |
| Test observations | 992 weeks |
| Hold-out observations | 39 weeks |

The results checker returned `RESULTS OK` with zero errors. H1, H2 and H3 remain rejected:

- **H1:** the full design trails equal weight, shrinkage mean-variance and ERC on net Sharpe; Holm-adjusted p-values do not support superiority.
- **H2:** its maximum drawdown is worse than shrinkage GMV, ERC and risk-matched equal weight.
- **H3:** the stress coefficient is not positive and significant against equal weight or risk-matched equal weight.

The latest run changes provenance, not interpretation. Its result tables agree with the corrected lag-one workbook used in the prior report build; the only workbook changes from that interim copy are run metadata, the populated `RUN_LOG` sheet and the frozen-choice timestamp.

## 4. Report audit

| Check | Result |
|---|---|
| DOCX structure | 16 pages, 12 tables, five embedded result figures |
| Report body | 5,235 words in paragraph text; all required sections developed |
| Number trace | `TRACE PASS`; zero untraced report numbers |
| Submission lint | Zero errors and zero warnings |
| Accessibility | Zero high, medium or low findings |
| Visual QA | All 16 DOCX-rendered pages inspected; no clipping, overlap, broken glyphs or table overflow |
| PDF preflight | 16 pages, openable, unencrypted, tagged, not scanned |
| Prose integrity | No invisible/confusable Unicode, no formulaic opening transitions and zero em dashes; document sentence-length CV 1.742 and paragraph-length CV 0.511 |

The final report replaces the five embedded charts with the PNGs from the newest run and records the completed Colab execution. Its result values did not require revision because they already matched the corrected lag-one evidence.

## 5. Repository and package audit

| Gate | Result |
|---|---|
| Repository unit tests | 5/5 passed |
| Repository verifier | Pass: executed notebook, seven figures, 42 HTML outputs, 24-sheet workbook, 22 CSVs and report structure |
| Capstone Pro tests | 21/21 passed |
| REAP self-test and tests | 6/6 and 13/13 passed |
| AutoResearch Pro tests | 22/22 passed |
| Skill installation | All 12 attached bundles materialized; every available SHA-256 manifest verified |
| SUBMISSION separation | Report, PDF, executed notebook, figures, README, contributors and requirements only |
| TEAM_ONLY completeness | Audit, status, v13 config, decisions, assistance log, REAP report, primary run archive, executed HTML and evidence |
| GitHub release separation | No TEAM_ONLY directory or primary run archive in the public repository package |
| ZIP integrity | No corrupt members, duplicate paths, absolute paths or traversal entries |
| Manifest | SHA-256 and byte-size manifest generated and verified for all four release archives |

## 6. Rubric coverage

| Rubric area | Available | Projection | Basis |
|---|---:|---:|---|
| Developed draft project | 60 | 58 | Full report structure, current traced results, progress and remaining work |
| Source-code development | 50 | 48 | Fully executed notebook, reusable source, config, tests, outputs and reproducible release |
| Writing and presentation | 20 | 19 | Formal organization, five clear figures, consistent tables and verified PDF |
| **Total** | **130** | **125** | Internal estimate, not a promised grade |

## 7. Remaining actions for M7

1. Push the repository to the team's private GitHub project and grant the instructor read-only access.
2. Review M6 feedback and decide whether the secondary-universe result belongs in the final report.
3. Confirm final AI-use declaration wording and reference formatting against the current course instructions.
4. Tag the final repository release and prepare the individual presentations.

## Final publication status

**PASS.** The final M6 report, PDF, audit, GitHub repository package, SUBMISSION package, TEAM_ONLY package and complete archive are suitable for delivery from the newest full-run evidence set.
