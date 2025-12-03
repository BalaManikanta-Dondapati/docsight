# embeddings/build_faiss_index.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

CHUNKS_PATH = Path("data/processed/qatar_chunks.json")
META_OUT = Path("data/processed/chunk_metadata.json")
INDEX_OUT = Path("data/processed/faiss_index.idx")
EMB_DIM = 768  # all-mpnet-base-v2

def load_chunks(path):
    chunks = json.load(open(path, "r", encoding="utf-8"))
    return chunks

def embed_texts(texts, model_name="all-mpnet-base-v2", batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(embs)
    embeddings = np.vstack(embeddings).astype("float32")
    return embeddings

def build_faiss_index(embeddings, dim=EMB_DIM):
    # normalize for cosine-similarity if desired
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors -> cosine
    index.add(embeddings)
    return index

def main():
    assert CHUNKS_PATH.exists(), f"Chunks file not found: {CHUNKS_PATH}"
    chunks = load_chunks(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    print(f"Loaded {len(texts)} chunks. Computing embeddings with sentence-transformers...")
    embeddings = embed_texts(texts, batch_size=32)

    print("Building FAISS index (Inner Product on normalized vectors)...")
    index = build_faiss_index(embeddings, dim=embeddings.shape[1])

    print(f"Saving FAISS index to {INDEX_OUT} and metadata to {META_OUT}...")
    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_OUT))

    # Save metadata mapping: index -> chunk metadata
    meta = [{"id": i, "chunk_id": chunks[i]["chunk_id"], "doc_id":chunks[i]["doc_id"],
             "page": chunks[i]["page"], "start_char": chunks[i]["start_char"],
             "end_char": chunks[i]["end_char"]} for i in range(len(chunks))]
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done. You can run nearest-neighbor queries against the saved FAISS index.")

if __name__ == "__main__":
    main()
