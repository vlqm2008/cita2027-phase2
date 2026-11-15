"""
Build ultra-tiny, pure code zip package (CITA2027_Code_Sieu_Nhe.zip).
Pure code only:
- src/
- tests/
- Essential scripts/
- pyproject.toml, requirements.txt, README.md, BAO_CAO_RUT_GON_CITA_2027.md
Zero media, zero docx, zero html, zero caches.
"""

from pathlib import Path
import zipfile

ROOT = Path("D:/01_AI_Research_Projects/cita2027-phase2")
ZIP_OUT = ROOT / "CITA2027_Code_Sieu_Nhe.zip"

INCLUDE_SCRIPTS = [
    "run_all_phases.py",
    "run_hypothesis_h1.py",
    "run_hypothesis_h2.py",
    "run_hypothesis_h3.py",
    "run_real_data_adaptation.py",
    "run_statistical_tests.py",
    "run_ablation_study.py",
    "plot_paper_figures.py",
    "run_advanced_benchmarks.py",
    "generate_comprehensive_paper_pdf.py",
    "prepare_fjcai_multiasset_data.py",
    "run_fjcai_complete_benchmark.py",
    "generate_fjcai_concise_docx_report.py"
]

def main():
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. src/
        for p in (ROOT / "src").rglob("*.py"):
            if "__pycache__" not in p.parts:
                zipf.write(p, f"code/{p.relative_to(ROOT).as_posix()}")

        # 2. tests/
        for p in (ROOT / "tests").rglob("*.py"):
            if "__pycache__" not in p.parts:
                zipf.write(p, f"code/{p.relative_to(ROOT).as_posix()}")

        # 3. essential scripts/
        for sname in INCLUDE_SCRIPTS:
            sp = ROOT / "scripts" / sname
            if sp.exists():
                zipf.write(sp, f"code/scripts/{sname}")

        # 4. config & readme & concise report
        for fname in ["pyproject.toml", "README.md", "BAO_CAO_RUT_GON_CITA_2027.md", "BAO_CAO_FJCAI_FULL_DATA.md"]:
            fp = ROOT / fname
            if fp.exists():
                zipf.write(fp, f"code/{fname}")

        # 4b. FJCAI replication tables
        for tp in (ROOT / "paper_tables").glob("table_fjcai_*.csv"):
            zipf.write(tp, f"code/paper_tables/{tp.name}")

        # 5. requirements
        req = ROOT / "requirements" / "requirements.txt"
        if req.exists():
            zipf.write(req, "code/requirements.txt")

    size_kb = ZIP_OUT.stat().st_size / 1024
    print(f"[OK] Created CITA2027_Code_Sieu_Nhe.zip: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")

if __name__ == "__main__":
    main()
