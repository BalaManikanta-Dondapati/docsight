# ingest/simple_extract.py
import fitz  # PyMuPDF
from pathlib import Path
import json

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text})
    return pages

def main():
    pdf_path = Path("data/raw/qatar_test_doc.pdf")
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found. Put the PDF in data/raw/")
        return
    pages = extract_text_by_page(pdf_path)
    out = Path("data/processed/qatar_pages_simple.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Extracted {len(pages)} pages. Saved to {out}")

if __name__ == "__main__":
    main()
