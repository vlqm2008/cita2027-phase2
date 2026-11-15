"""
Build complete project package WITH ALL DATA included:
- src/
- scripts/
- tests/
- data/ (ALL raw datasets)
- artifacts/ (ALL processed data, snapshots, manifests, calibration, checkpoints, adaptation data)
- configs/
- requirements/ + requirements.txt
- pyproject.toml
- README.md
- BAO_CAO_RUT_GON_CITA_2027.docx
- BAO_CAO_TONG_HOP_TOAN_DIEN_CITA_2027.docx
- HUONG_DAN_CHAY.txt
"""

import os
from pathlib import Path
import zipfile

ROOT = Path("D:/01_AI_Research_Projects/cita2027-phase2")
ZIP_OUT = ROOT / "CITA2027_Full_Project_With_Data.zip"

DIRS_TO_INCLUDE = [
    "src",
    "scripts",
    "tests",
    "data",
    "artifacts",
    "configs",
    "requirements",
    "paper_tables"
]

FILES_TO_INCLUDE = [
    "pyproject.toml",
    "README.md",
    "reproduce_all.py",
    "reproduce_all.bat",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "BAO_CAO_RUT_GON_CITA_2027.docx",
    "BAO_CAO_RUT_GON_CITA_2027.md",
    "BAO_CAO_TONG_HOP_TOAN_DIEN_CITA_2027.docx",
    "BAO_CAO_TONG_HOP_TOAN_DIEN_CITA_2027.md",
    "BAO_CAO_FJCAI_FULL_DATA.docx",
    "BAO_CAO_FJCAI_FULL_DATA.md",
]

def should_skip(path_obj: Path) -> bool:
    parts = path_obj.parts
    # Exclude virtual environments, git, pycache
    if any(p in [".venv", ".venv-moirai", "__pycache__", ".pytest_cache", ".git", ".vscode"] for p in parts):
        return True
    if path_obj.suffix.lower() in [".pyc", ".pyo"]:
        return True
    return False

def main():
    print(f"[*] Packaging Full Project with ALL DATA into: {ZIP_OUT}")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    count = 0
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # 1. Include root files
        for fname in FILES_TO_INCLUDE:
            fp = ROOT / fname
            if fp.exists():
                arcname = f"cita2027_project/{fname}"
                zipf.write(fp, arcname)
                count += 1

        # 2. Include all specified directories with all data
        for dname in DIRS_TO_INCLUDE:
            dp = ROOT / dname
            if dp.exists():
                for root, dirs, files in os.walk(dp):
                    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".pytest_cache"]]
                    for file in files:
                        fp = Path(root) / file
                        if not should_skip(fp):
                            rel = fp.relative_to(ROOT)
                            arcname = f"cita2027_project/{rel.as_posix()}"
                            zipf.write(fp, arcname)
                            count += 1

        # 3. Add requirements.txt at root if exists
        req_fp = ROOT / "requirements" / "requirements.txt"
        if req_fp.exists():
            zipf.write(req_fp, "cita2027_project/requirements.txt")

        # 4. Quickstart instructions
        guide = """# HƯỚNG DẪN CHẠY DỰ ÁN TRÊN MÁY KHÁC (ĐẦY ĐỦ DATA 100%)

Gói này đã chứa ĐẦY ĐỦ 100% DATA (cả raw data trong data/ và processed data trong artifacts/).
Bạn không cần tải thêm bất kỳ file dữ liệu nào từ bên ngoài.

3 BƯỚC THỰC THI:
1. Cài đặt môi trường:
   pip install -r requirements.txt

2. Chạy toàn bộ 10 Phase nghiên cứu (Tự động 100%):
   python scripts/run_all_phases.py

3. Chạy 40 Unit Tests (Xanh 100%):
   pytest tests/ -v
"""
        zipf.writestr("cita2027_project/HUONG_DAN_CHAY.txt", guide)

    size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Package created successfully!")
    print(f"  - Total files : {count}")
    print(f"  - Zip File    : {ZIP_OUT}")
    print(f"  - Total Size  : {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
