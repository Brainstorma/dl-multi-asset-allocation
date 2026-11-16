#!/usr/bin/env python3
from pathlib import Path
import ast, base64, json, math, re
import pandas as pd
from openpyxl import load_workbook
from docx import Document

root = Path(__file__).resolve().parents[1]
required = [
    "README.md", "LICENSE", "CONTRIBUTORS.md", "requirements.txt", "requirements-ci.txt",
    "config/m6.yaml", "src/m6core.py", "notebooks/Final_Capstone_Executed.ipynb",
    "outputs/current_lag1/M6_RESULTS_17863.xlsx", "outputs/current_lag1/m6_run_log.md",
    "docs/MScFE_690_Final_Capstone_Report_Group17863.docx",
    "docs/MScFE_690_Final_Capstone_Report_Group17863.pdf",
    "docs/Solution_Design_Document_Group17863.docx",
    "docs/Solution_Design_Document_Group17863.pdf",
    "docs/REPRODUCIBILITY.md", "docs/RUN_GUIDE.md",
]
missing = [x for x in required if not (root / x).exists()]
assert not missing, f"missing: {missing}"
ast.parse((root / "src/m6core.py").read_text(encoding="utf-8"))

nb = json.loads((root / "notebooks/Final_Capstone_Executed.ipynb").read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
assert len(code_cells) == 18 and all(c.get("execution_count") is not None for c in code_cells)
imgs = [(i, o["data"]["image/png"]) for i, c in enumerate(nb["cells"]) for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
html_total = sum("text/html" in o.get("data", {}) for c in nb["cells"] for o in c.get("outputs", []))
errs = sum(o.get("output_type") == "error" for c in nb["cells"] for o in c.get("outputs", []))
assert (len(imgs), html_total, errs) == (7, 42, 0), (len(imgs), html_total, errs)
assert sum("text/html" in o.get("data", {}) for o in nb["cells"][30].get("outputs", [])) == 22
src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
assert 'RUN_TAG = "lag1_v3_inline"' in src and 'store.stage(f"06_tests_{RUN_TAG}"' in src

for i, c in enumerate(code_cells):
    s = "".join(c.get("source", []))
    if s.startswith("%%writefile"):
        candidate = "\n".join(s.splitlines()[1:]) + "\n"
        if s.splitlines()[0].endswith("m6core.py"):
            assert candidate == (root / "src/m6core.py").read_text(encoding="utf-8")
        s = candidate
    if any(line.lstrip().startswith(("!", "%")) for line in s.splitlines()):
        continue
    ast.parse(s, filename=f"code-cell-{i}")

figure_names = ["architecture_diagram.png", "timing_diagram.png", "fig1_sharpe_by_strategy.png", "fig2_calmar_by_strategy.png", "fig3_drawdowns.png", "fig4_stress_windows.png", "fig5_validation_grid.png"]
for (_, encoded), name in zip(imgs, figure_names):
    if isinstance(encoded, list): encoded = "".join(encoded)
    assert base64.b64decode(encoded) == (root / "figures" / name).read_bytes(), name

book = root / "outputs/current_lag1/M6_RESULTS_17863.xlsx"
wb = load_workbook(book, read_only=True, data_only=True)
assert len(wb.sheetnames) == 24
assert {"README", "RUN_LOG", "metrics_test", "validation_leaderboard", "holdout_2026", "execution_timing_compare"} <= set(wb.sheetnames)
readme = dict(wb["README"].values)
assert readme["mode"] == "full" and readme["execution_lag_days"] == 1 and readme["run_tag"] == "lag1_v3_inline"
assert wb["RUN_LOG"].max_row == 16
result_sheets = [s for s in wb.sheetnames if s not in {"README", "RUN_LOG"}]
csv_dir = root / "outputs/current_lag1/tables"
assert len(result_sheets) == 22
assert {p.stem for p in csv_dir.glob("*.csv") if not p.stem.startswith("report_")} == set(result_sheets)
for sheet in result_sheets:
    x = pd.read_excel(book, sheet); y = pd.read_csv(csv_dir / f"{sheet}.csv")
    pd.testing.assert_frame_equal(x, y, check_dtype=False, rtol=1e-12, atol=1e-12)

metrics = pd.read_excel(book, "metrics_test")
assert len(metrics) == 10 and (metrics["weeks"] == 992).all()
assert ((metrics["ann_return"] / metrics["max_dd"].abs() - metrics["calmar"]).abs() < 1e-10).all()
assert (metrics["sharpe"].rank(ascending=False, method="min").astype(int) == metrics["sharpe_rank"]).all()
for col in metrics.select_dtypes("number"):
    assert metrics[col].map(lambda x: math.isfinite(float(x))).all(), col
lb = pd.read_excel(book, "validation_leaderboard")
assert len(lb) == 21 and set(lb["selection_direction"]) == {"max"}
assert lb.loc[lb["selected_overall"].astype(bool), "mean_val_sharpe"].iloc[0] == lb["mean_val_sharpe"].max()
assert (pd.read_excel(book, "holdout_2026")["weeks"] == 39).all()
timing = pd.read_excel(book, "execution_timing_compare")
assert (timing["sharpe_lag1"].round(12) != timing["sharpe_same_close"].round(12)).all()

report = Document(root / "docs/MScFE_690_Final_Capstone_Report_Group17863.docx")
report_text = "\n".join(p.text for p in report.paragraphs)
assert len(report_text.split()) > 5000 and len(report.tables) == 12 and len(report.inline_shapes) == 5
assert "Final Project, MScFE 690 Capstone, Module 7" in report_text
assert "6. Decision Attribution and Practical Guidance" in report_text
solution = Document(root / "docs/Solution_Design_Document_Group17863.docx")
assert len(solution.tables) == 2 and "Solution Design Document" in "\n".join(p.text for p in solution.paragraphs)

print(f"REPO_OK files={sum(1 for p in root.rglob('*') if p.is_file())} tests=external executed_code_cells={len(code_cells)} notebook_images={len(imgs)} notebook_tables={html_total} workbook_sheets={len(wb.sheetnames)} result_csvs={len(result_sheets)} report_words={len(report_text.split())}")
