# ingest/enhanced_extract.py
import json
from pathlib import Path
from PIL import Image
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
import fitz  # pymupdf
import pandas as pd
from tqdm import tqdm

RAW_PDF = Path("data/raw/qatar_test_doc.pdf")
OUT_DIR = Path("data/processed")
TABLE_DIR = OUT_DIR / "tables"
OUT_JSON = OUT_DIR / "qatar_pages_enhanced.json"

def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = {}
    for i in range(len(doc)):
        page = doc[i]
        texts[i+1] = page.get_text("text")
    return texts

def extract_tables_pdfplumber(pdf_path):
    tables_by_page = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            page_tables = []
            for tidx, table in enumerate(tables):
                # convert nested list to DataFrame if possible
                if not table:
                    continue
                try:
                    df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                except Exception:
                    df = pd.DataFrame(table)
                csv_path = TABLE_DIR / f"table_page_{i+1}_{tidx}.csv"
                TABLE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                page_tables.append(str(csv_path))
            if page_tables:
                tables_by_page[i+1] = page_tables
    return tables_by_page

def ocr_pages_pdf2image(pdf_path, dpi=200, poppler_path=None):
    """
    Convert pages to images using pdf2image and OCR each page.
    If poppler_path is provided, pass it to convert_from_path to avoid needing PATH.
    """
    ocr_by_page = {}
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        poppler_path=poppler_path  # <--- important
    )
    for i, pil in enumerate(images):
        text = pytesseract.image_to_string(pil, lang="eng")
        img_out = OUT_DIR / f"page_render_{i+1}.png"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pil.save(img_out, format="PNG")
        ocr_by_page[i+1] = {"ocr_text": text, "rendered_image": str(img_out)}
    return ocr_by_page


def build_enhanced_json(pdf_path):
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    print("Extracting text with PyMuPDF...")
    text_by_page = extract_text_pymupdf(pdf_path)
    print("Extracting tables with pdfplumber...")
    tables_by_page = extract_tables_pdfplumber(pdf_path)
    print("Rendering pages and OCRing with pytesseract (pdf2image)...")
    POPPLER_BIN = r"C:\tools\poppler-xx\Library\bin"
    ocr_by_page = ocr_pages_pdf2image(pdf_path, dpi=200, poppler_path=POPPLER_BIN)


    pages = []
    num_pages = max(len(text_by_page), len(ocr_by_page))
    for p in range(1, num_pages+1):
        page_entry = {
            "doc_id": "qatar_test_doc",
            "page": p,
            "text": text_by_page.get(p, ""),
            "ocr_text": ocr_by_page.get(p, {}).get("ocr_text", ""),
            "rendered_image": ocr_by_page.get(p, {}).get("rendered_image", ""),
            "tables": tables_by_page.get(p, []),
        }
        pages.append(page_entry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Saved enhanced JSON to: {OUT_JSON}")
    print(f"Tables (if any) saved under: {TABLE_DIR}")
    return pages

if __name__ == "__main__":
    pages = build_enhanced_json(RAW_PDF)
    print(f"Processed {len(pages)} pages.")
