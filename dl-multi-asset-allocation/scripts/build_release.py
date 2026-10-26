#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, shutil, zipfile

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
if dist.exists():
    shutil.rmtree(dist)
dist.mkdir()
submission = dist / "SUBMISSION"
team = dist / "TEAM_ONLY"
github_parent = dist / "GITHUB_REPO"
github = github_parent / "dl-multi-asset-allocation"
for p in (submission, team, github):
    p.mkdir(parents=True, exist_ok=True)

# SUBMISSION: project content only.
for src in [
    root / "docs/MScFE_690_M6_Draft_Project_Group17863_FINAL.docx",
    root / "docs/MScFE_690_M6_Draft_Project_Group17863_FINAL.pdf",
    root / "notebooks/M6_FULL_RUN_Group17863_REVISED_COLAB.ipynb",
    root / "README.md", root / "requirements.txt", root / "CONTRIBUTORS.md"
]:
    shutil.copy2(src, submission / src.name)
shutil.copytree(root / "figures", submission / "figures")

# TEAM_ONLY: process, audit and complete evidence.
shutil.copytree(root / "team_only", team / "team_only")
shutil.copy2(root / "docs/AUDIT_REPORT_M6_FINAL.md", team / "AUDIT_REPORT_M6_FINAL.md")
shutil.copy2(root / "docs/RESULTS_STATUS.md", team / "RESULTS_STATUS.md")
shutil.copytree(root / "outputs/current_lag1", team / "evidence/current_lag1")

# GitHub-ready tree excludes team-only material and release products.
skip_roots = {"team_only", "dist"}
for src in sorted(root.rglob("*")):
    rel = src.relative_to(root)
    if rel.parts and rel.parts[0] in skip_roots:
        continue
    if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in rel.parts):
        continue
    dst = github / rel
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

# Remove audit/process references from the public release if they point to team-only paths.

def zip_dir(src: Path, target: Path, arc_prefix: str = ""):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, Path(arc_prefix) / f.relative_to(src))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    return target

submission_zip = zip_dir(submission, dist / "M6_SUBMISSION_17863_FINAL.zip")
team_zip = zip_dir(team, dist / "M6_TEAM_ONLY_17863_FINAL.zip")
repo_zip = zip_dir(github, dist / "M6_GITHUB_REPO_17863_FINAL.zip", "dl-multi-asset-allocation")

complete = dist / "M6_COMPLETE_Group17863_FINAL.zip"
with zipfile.ZipFile(complete, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for folder in (submission, team, github_parent):
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(dist))
    for f in (submission_zip, team_zip, repo_zip):
        z.write(f, f.name)
    for f in [
        root / "docs/MScFE_690_M6_Draft_Project_Group17863_FINAL.docx",
        root / "docs/MScFE_690_M6_Draft_Project_Group17863_FINAL.pdf",
        root / "notebooks/M6_FULL_RUN_Group17863_REVISED_COLAB.ipynb",
        root / "outputs/current_lag1/M6_RESULTS_17863.xlsx",
        root / "docs/AUDIT_REPORT_M6_FINAL.md"
    ]:
        z.write(f, f.name)
with zipfile.ZipFile(complete) as z:
    assert z.testzip() is None

artifacts = [submission_zip, team_zip, repo_zip, complete]
manifest = []
for p in artifacts:
    manifest.append({"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
(dist / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(dist / "SHA256SUMS.txt").write_text("\n".join(f"{x['sha256']}  {x['file']}" for x in manifest) + "\n", encoding="utf-8")
for row in manifest:
    print(row["file"], row["bytes"], row["sha256"])
