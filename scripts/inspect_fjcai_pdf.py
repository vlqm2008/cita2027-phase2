from pathlib import Path
import pypdf

pdf_path = Path(r"C:\Users\Admin\.gemini\antigravity\brain\6d964916-4283-4d59-a794-763b80b97e6c\.user_uploaded\media_1794758936099_a3680b35.pdf")

if pdf_path.exists():
    reader = pypdf.PdfReader(pdf_path)
    print(f"Num pages: {len(reader.pages)}")
    full_text = ""
    for i in range(min(5, len(reader.pages))):
        full_text += f"\n--- PAGE {i+1} ---\n" + reader.pages[i].extract_text()
    print(full_text[:3000])
else:
    print("PDF not found!")
