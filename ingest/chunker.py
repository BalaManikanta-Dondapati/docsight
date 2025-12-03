# ingest/chunker.py
import json
from pathlib import Path
from tqdm import tqdm

IN_JSON = Path("data/processed/qatar_pages_enhanced.json")
OUT_JSON = Path("data/processed/qatar_chunks.json")

def choose_text(page_obj):
    txt = page_obj.get("text", "") or page_obj.get("ocr_text", "")
    return txt.strip()

def make_chunks(text, chunk_size=1000, overlap=200):
    """
    Return list of (start_idx, end_idx, chunk_text).
    chunk_size and overlap are in characters.
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((start, min(end, length), chunk))
        if end >= length:
            break
        start = end - overlap  # overlap
    return chunks

def build_chunks(in_path=IN_JSON, out_path=OUT_JSON, chunk_size=1000, overlap=200):
    assert in_path.exists(), f"Input JSON not found: {in_path}"
    pages = json.load(open(in_path, "r", encoding="utf-8"))
    chunks_out = []
    for page in tqdm(pages, desc="Pages"):
        doc_id = page.get("doc_id", "qatar_test_doc")
        page_no = page.get("page", None)
        text = choose_text(page)
        # normalize whitespace
        text = " ".join(text.split())
        page_chunks = make_chunks(text, chunk_size=chunk_size, overlap=overlap)
        for i, (s, e, chunk_text) in enumerate(page_chunks, start=1):
            chunks_out.append({
                "doc_id": doc_id,
                "page": page_no,
                "chunk_id": f"{page_no}_{i}",
                "start_char": s,
                "end_char": e,
                "text": chunk_text
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks_out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks_out)} chunks to {out_path}")
    return chunks_out

if __name__ == "__main__":
    # change chunk_size/overlap here if you want different values
    chunks = build_chunks(chunk_size=1000, overlap=200)
    # print first 3 chunks for page 1 (if available)
    p1 = [c for c in chunks if c["page"] == 1][:3]
    print("\n=== first 3 chunks for page 1 ===")
    for c in p1:
        print(json.dumps(c, ensure_ascii=False)[:1000])

