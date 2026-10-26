#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, zipfile

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
expected = [
    "M6_SUBMISSION_17863_FINAL.zip",
    "M6_TEAM_ONLY_17863_FINAL.zip",
    "M6_GITHUB_REPO_17863_FINAL.zip",
    "M6_COMPLETE_Group17863_FINAL.zip",
]
for name in expected:
    p = dist / name
    assert p.exists() and p.stat().st_size > 0, name
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None, name
        names = z.namelist()
        assert len(names) == len(set(names)), f"duplicate archive paths in {name}"
        assert not any(Path(n).is_absolute() or ".." in Path(n).parts for n in names)

with zipfile.ZipFile(dist / expected[0]) as z:
    names = set(z.namelist())
    assert "MScFE_690_M6_Draft_Project_Group17863_FINAL.pdf" in names
    assert "M6_FULL_RUN_Group17863_REVISED_COLAB.ipynb" in names
    assert not any("AUDIT" in n or "team_only" in n.lower() or "M6_RESULTS" in n for n in names)
    nb = json.loads(z.read("M6_FULL_RUN_Group17863_REVISED_COLAB.ipynb"))
    code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    imgs = sum("image/png" in o.get("data", {}) for c in nb["cells"] for o in c.get("outputs", []))
    errs = sum(o.get("output_type") == "error" for c in nb["cells"] for o in c.get("outputs", []))
    assert len(code) == 18 and all(c.get("execution_count") is not None for c in code)
    assert imgs == 7 and errs == 0

with zipfile.ZipFile(dist / expected[1]) as z:
    names = set(z.namelist())
    assert "AUDIT_REPORT_M6_FINAL.md" in names
    assert "evidence/current_lag1/M6_RESULTS_17863.xlsx" in names
    assert "team_only/REAP_GATE_REPORT_M6.md" in names
    assert "team_only/00_TEAM_CONFIG_v13.md" in names
    assert "team_only/evidence/latest_colab/CAPSTONE_M6_RUN-20261026T000130Z-1-001.zip" in names
    assert "team_only/evidence/latest_colab/M6_FULL_RUN_Group17863_REVISED_COLAB_Executed.html" in names

with zipfile.ZipFile(dist / expected[2]) as z:
    names = set(z.namelist())
    assert "dl-multi-asset-allocation/README.md" in names
    assert "dl-multi-asset-allocation/outputs/current_lag1/M6_RESULTS_17863.xlsx" in names
    assert "dl-multi-asset-allocation/notebooks/M6_FULL_RUN_Group17863_REVISED_COLAB.ipynb" in names
    assert not any("/team_only/" in n for n in names)
    assert not any(n.endswith("CAPSTONE_M6_RUN-20261026T000130Z-1-001.zip") for n in names)

with zipfile.ZipFile(dist / expected[3]) as z:
    names = set(z.namelist())
    assert any(n.startswith("SUBMISSION/") for n in names)
    assert any(n.startswith("TEAM_ONLY/") for n in names)
    assert any(n.startswith("GITHUB_REPO/dl-multi-asset-allocation/") for n in names)
    for name in expected[:3]:
        assert name in names

manifest = json.loads((dist / "SHA256SUMS.json").read_text(encoding="utf-8"))
assert {x["file"] for x in manifest} == set(expected)
for row in manifest:
    p = dist / row["file"]
    assert p.stat().st_size == row["bytes"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"]
print("DELIVERY_OK", " ".join(expected))
