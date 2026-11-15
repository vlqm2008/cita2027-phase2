import zipfile
from pathlib import Path

zip_path = Path(r"D:\01_AI_Research_Projects\cita2027-phase2\CITA2027_Full_Project_With_Data.zip")
with zipfile.ZipFile(zip_path, "r") as z:
    all_files = z.namelist()
    new_scripts = [
        "inspect_datasets.py",
        "generate_horizon_empirical_evidence.py",
        "update_master_handoff_reports.py",
        "generate_one_master_report.py",
        "generate_concise_report.py",
        "build_clean_lightweight_code_zip.py",
        "build_ultra_tiny_code_zip.py",
        "build_standalone_portable_project_zip.py",
        "build_full_data_project_zip.py",
        "export_detailed_configuration_report.py"
    ]
    print(f"Total entries in zip: {len(all_files)}")
    print("Checking recently created scripts:")
    for s in new_scripts:
        found = any(s in f for f in all_files)
        print(f"  [{'OK' if found else 'MISSING'}] {s}")
