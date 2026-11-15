"""
Build complete, self-contained portable project zip for another machine.
Size target: ~1.5 - 2.5 MB.
Contains EVERYTHING needed to reproduce with 100% success on any machine:
- src/
- scripts/
- tests/
- Clean processed benchmark CSVs (SPY & BTC)
- requirements.txt, pyproject.toml
- README.md (3-step quickstart)
- BAO_CAO_RUT_GON_CITA_2027.docx
"""

from pathlib import Path
import zipfile

ROOT = Path("D:/01_AI_Research_Projects/cita2027-phase2")
ZIP_OUT = ROOT / "CITA2027_Portable_Project.zip"

def main():
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. src/
        for p in (ROOT / "src").rglob("*.py"):
            if "__pycache__" not in p.parts:
                zipf.write(p, f"cita2027/{p.relative_to(ROOT).as_posix()}")

        # 2. tests/
        for p in (ROOT / "tests").rglob("*.py"):
            if "__pycache__" not in p.parts:
                zipf.write(p, f"cita2027/{p.relative_to(ROOT).as_posix()}")

        # 3. scripts/
        for p in (ROOT / "scripts").glob("*.py"):
            zipf.write(p, f"cita2027/{p.relative_to(ROOT).as_posix()}")

        # 4. Clean processed datasets (SPY & BTC CSV)
        for p in (ROOT / "artifacts" / "processed").glob("*.csv"):
            zipf.write(p, f"cita2027/{p.relative_to(ROOT).as_posix()}")
        for p in (ROOT / "artifacts" / "btcusdt_data" / "snapshots").glob("*.csv"):
            zipf.write(p, f"cita2027/{p.relative_to(ROOT).as_posix()}")

        # 5. requirements, pyproject, readme
        for fname in ["pyproject.toml", "README.md", "BAO_CAO_RUT_GON_CITA_2027.docx"]:
            fp = ROOT / fname
            if fp.exists():
                zipf.write(fp, f"cita2027/{fname}")

        req = ROOT / "requirements" / "requirements.txt"
        if req.exists():
            zipf.write(req, "cita2027/requirements.txt")

        # Portable instructions
        quickstart = """# HƯỚNG DẪN CHẠY TRÊN MÁY KHÁC (3 BƯỚC)

1. Cài thư viện:
   pip install -r requirements.txt

2. Chạy toàn bộ 10 phase thực nghiệm (tự động 100%):
   python scripts/run_all_phases.py

3. Kiểm tra 40 unit tests (xanh 100%):
   pytest tests/ -v
"""
        zipf.writestr("cita2027/HUONG_DAN_CHAY.txt", quickstart)

    size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"[OK] Created CITA2027_Portable_Project.zip: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
