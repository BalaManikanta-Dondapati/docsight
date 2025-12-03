# retrieval/query_faiss.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Paths (adjust if you used different names)
CHUNKS_PATH = Path("data/processed/qatar_chunks.json")
META_PATH = Path("data/processed/chunk_metadata.json")
INDEX_PATH = Path("data/processed/faiss_index.idx")

MODEL_NAME = "all-mpnet-base-v2"  # same model used for index
TOP_K = 5

def load_index(index_path):
    assert index_path.exists(), f"Index not found: {index_path}"
    index = faiss.read_index(str(index_path))
    return index

def load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta

def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

def embed_query(query, model):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)
    return q_emb

def query_index(query, top_k=TOP_K):
    # load resources
    index = load_index(INDEX_PATH)
    meta = load_meta(META_PATH)
    chunks = load_chunks(CHUNKS_PATH)
    model = SentenceTransformer(MODEL_NAME)

    q_emb = embed_query(query, model)

    # search
    D, I = index.search(q_emb, top_k)
    scores = D[0]
    idxs = I[0]

    results = []
    for score, idx in zip(scores, idxs):
        # guard: sometimes idx == -1 if not enough results
        if idx < 0:
            continue
        chunk_meta = meta[idx]
        chunk_obj = chunks[idx]
        results.append({
            "score": float(score),
            "index": int(idx),
            "chunk_id": chunk_meta.get("chunk_id"),
            "doc_id": chunk_meta.get("doc_id"),
            "page": chunk_meta.get("page"),
            "start_char": chunk_meta.get("start_char"),
            "end_char": chunk_meta.get("end_char"),
            "text_snippet": chunk_obj.get("text")[:800]  # first 800 chars for preview
        })
    return results

def pretty_print_results(results):
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, start=1):
        print(f"\n=== Result {i} | score: {r['score']:.4f} | page: {r['page']} | chunk_id: {r['chunk_id']} ===")
        print(r["text_snippet"])
        print("----")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--query", dest="query", required=False, default=None, help="Query string")
    parser.add_argument("--k", dest="k", type=int, default=5, help="Top-K results")
    args = parser.parse_args()

    if args.query is None:
        # interactive prompt if no query provided
        q = input("Enter your question: ").strip()
    else:
        q = args.query

    print(f"Query: {q}\nRetrieving top {args.k} chunks...")
    res = query_index(q, top_k=args.k)
    pretty_print_results(res)
