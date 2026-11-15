"""
Script to create an ultra-lightweight, clean Source Code & Reproduction Package for Collaborator P.
Excludes:
- .venv, .venv-moirai (virtual environments)
- __pycache__, .pytest_cache, .git
- Large 25MB camera-ready PDF
- Heavy 300 DPI figures
- Raw Binance zip archives and heavy cache .npz files

Includes:
- src/ (Core algorithms, adapters, queues, metrics)
- scripts/ (Reproduction pipeline run_all_phases.py, etc.)
- tests/ (All 40 pytest unit test suites)
- Clean processed benchmark data (SPY & BTC CSV)
- paper_tables/ (CSV tables)
- Master Reports (Word .docx, HTML, Markdown)
- pyproject.toml, requirements, README.md, reproduce bat/py
"""

import os
from pathlib import Path
import zipfile

ROOT = Path("D:/01_AI_Research_Projects/cita2027-phase2")
PKG_NAME = "CITA2027_Phase2_SourceCode_Lightweight"
ZIP_OUT = ROOT / f"{PKG_NAME}.zip"

# Include list of essential files and directories
INCLUDE_DIRS = ["src", "scripts", "tests", "requirements", "configs"]
INCLUDE_FILES = [
    "pyproject.toml",
    "README.md",
    "reproduce_all.py",
    "reproduce_all.bat",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "BAO_CAO_TONG_HOP_TOAN_DIEN_CITA_2027.docx",
    "BAO_CAO_TONG_HOP_TOAN_DIEN_CITA_2027.html",
    "BAO_CAO_TONG_HOP_TOAN_DIEN_CITA_2027.md",
    "BAO_CAO_CHI_TIET_CAU_HINH_VA_DANH_GIA_H.docx",
    "BAO_CAO_CHI_TIET_CAU_HINH_VA_DANH_GIA_H.md",
    "MINH_CHUNG_THUC_NGHIEM_DANH_GIA_CHIEU_DAI_H.docx",
    "MINH_CHUNG_THUC_NGHIEM_DANH_GIA_CHIEU_DAI_H.md",
    "# Research Handoff - Collaborator P (2).txt"
]

def should_exclude(path_obj: Path) -> bool:
    parts = path_obj.parts
    # Exclude virtualenvs, caches, git
    if any(p in [".venv", ".venv-moirai", "__pycache__", ".pytest_cache", ".git", ".vscode", "cache"] for p in parts):
        return True
    # Exclude binaries, large pdfs, large archives
    if path_obj.suffix.lower() in [".pyc", ".pyo", ".zip", ".tar", ".gz", ".npz", ".exe"]:
        return True
    return False

def main():
    print(f"[*] Packaging Clean Lightweight Source Code into: {ZIP_OUT}")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. Add explicitly included single files
        for fname in INCLUDE_FILES:
            fpath = ROOT / fname
            if fpath.exists():
                arcname = f"{PKG_NAME}/{fname}"
                zipf.write(fpath, arcname)
                print(f"  [+] Added root file: {fname} ({fpath.stat().st_size / 1024:.1f} KB)")

        # 2. Add essential directories
        for dname in INCLUDE_DIRS:
            dpath = ROOT / dname
            if dpath.exists():
                for root, dirs, files in os.walk(dpath):
                    # Filter out directories in-place
                    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".pytest_cache", "cache"]]
                    for file in files:
                        fp = Path(root) / file
                        if not should_exclude(fp):
                            rel = fp.relative_to(ROOT)
                            arcname = f"{PKG_NAME}/{rel.as_posix()}"
                            zipf.write(fp, arcname)

        # 3. Add clean processed data CSVs for benchmark reproducibility
        data_processed = ROOT / "artifacts" / "processed"
        if data_processed.exists():
            for fp in data_processed.glob("*.csv"):
                rel = fp.relative_to(ROOT)
                arcname = f"{PKG_NAME}/{rel.as_posix()}"
                zipf.write(fp, arcname)
                print(f"  [+] Added benchmark dataset: {fp.name} ({fp.stat().st_size / 1024:.1f} KB)")
                
        btc_snaps = ROOT / "artifacts" / "btcusdt_data" / "snapshots"
        if btc_snaps.exists():
            for fp in btc_snaps.glob("*.csv"):
                rel = fp.relative_to(ROOT)
                arcname = f"{PKG_NAME}/{rel.as_posix()}"
                zipf.write(fp, arcname)
                print(f"  [+] Added BTC dataset snapshot: {fp.name} ({fp.stat().st_size / 1024:.1f} KB)")

        # 4. Add result CSV tables
        tab_dir = ROOT / "paper_tables"
        if tab_dir.exists():
            for fp in tab_dir.glob("*.csv"):
                rel = fp.relative_to(ROOT)
                arcname = f"{PKG_NAME}/{rel.as_posix()}"
                zipf.write(fp, arcname)

    size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Clean Lightweight Code Zip created successfully!")
    print(f"  - File : {ZIP_OUT}")
    print(f"  - Size : {size_mb:.2f} MB (Ultra-lightweight!)")

if __name__ == "__main__":
    main()
