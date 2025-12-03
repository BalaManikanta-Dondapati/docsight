# rag/answer_with_rag.py
import os
import sys
import json
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from google import genai  # Gemini SDK

# ============================================================
# Paths & settings
# ============================================================
CHUNKS_PATH = Path("data/processed/qatar_chunks.json")
META_PATH = Path("data/processed/chunk_metadata.json")
INDEX_PATH = Path("data/processed/faiss_index.idx")
TABLE_CSV = Path("data/processed/qatar_table1_corrected.csv")  # corrected CSV created earlier

MODEL_NAME = "all-mpnet-base-v2"
TOP_K = 5

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ ERROR: Please set GEMINI_API_KEY environment variable.")
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


# ============================================================
# Load resources
# ============================================================
def load_resources():
    chunks = json.load(open(CHUNKS_PATH, "r", encoding="utf-8"))
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))
    embed_model = SentenceTransformer(MODEL_NAME)
    return chunks, meta, index, embed_model


# ============================================================
# Read deterministic table values (Table 1 -> CSV)
# ============================================================
def read_table1_csv(csv_path=TABLE_CSV):
    """Return a dict mapping year->value (as string) from the corrected CSV."""
    if not csv_path.exists():
        return {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = list(csv.reader(f))
    if len(r) < 2:
        return {}
    years = r[0][1:]
    vals = r[1][1:]
    mapping = {}
    for y, v in zip(years, vals):
        try:
            mapping[int(y)] = v.strip() if v.strip() != "" else None
        except:
            mapping[y] = v.strip()
    return mapping


# ============================================================
# Embedding / retrieval
# ============================================================
def embed_query(query, model):
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    return q

def retrieve_chunks(query, model, index, chunks, meta, top_k=TOP_K):
    q_emb = embed_query(query, model)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({
            "score": float(score),
            "chunk_id": meta[idx]["chunk_id"],
            "page": meta[idx]["page"],
            "text": chunks[idx]["text"]
        })
    return results


# ============================================================
# Prompt building (strict citation)
# ============================================================
def build_prompt(question, retrieved, table_map=None):
    # Build evidence snippets from retrieved chunks
    evidence_blocks = []
    for r in retrieved:
        snippet = r["text"].strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " ..."
        evidence_blocks.append(f"[page: {r['page']}] {snippet}")
    evidence = "\n\n".join(evidence_blocks)

    # Add deterministic table values as explicit evidence if available
    table_lines = ""
    if table_map:
        # only include years we have (e.g., 2024 and 2025)
        table_evidence = []
        for y in sorted(table_map.keys()):
            v = table_map[y]
            if v is not None and v != "":
                table_evidence.append(f"Table 1 (page 39): {y} = {v}%")
        if table_evidence:
            table_lines = "DETERMINISTIC TABLE VALUES (from Table 1, page 39):\n" + "\n".join(table_evidence) + "\n\n"

    prompt = f"""
You are a factual assistant. Use ONLY the evidence provided below to answer the user's question.
If the answer is not contained in the evidence, reply exactly: "I don't know based on the provided evidence."
Always cite pages in parentheses after factual sentences (e.g., (page 13)).

{table_lines}
EVIDENCE FROM DOCUMENTS:
{evidence}

QUESTION:
{question}

INSTRUCTIONS:
- Provide exactly three parts:
  1) Short answer: one sentence with numeric answer(s) and page citation(s).
  2) Explanation: up to two short sentences explaining drivers, each ended with page citation(s).
  3) Sources: bullet list of pages used.
- After every factual sentence, append the page citation in parentheses.
- Do not invent facts or add extra commentary.
"""
    return prompt



# ============================================================
# Gemini call
# ============================================================
def call_gemini(system_prompt, user_prompt):
    full_prompt = system_prompt + "\n\n" + user_prompt
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt
    )
    # unwrap common response shapes
    try:
        return response.text.strip()
    except Exception:
        try:
            return response.candidates[0].content[0].text.strip()
        except Exception:
            return str(response)


# ============================================================
# Combined answer (LLM + deterministic)
# ============================================================
def answer_question(question, top_k=TOP_K):
    # load resources
    chunks, meta, index, embed_model = load_resources()
    # retrieve
    retrieved = retrieve_chunks(question, embed_model, index, chunks, meta, top_k)
    # deterministic table values
    table_map = read_table1_csv()

    # Build LLM prompt and call Gemini if we have retrieved evidence
    llm_answer = None
    if retrieved:
        prompt = build_prompt(question, retrieved, table_map=read_table1_csv())
        system_prompt = "You answer strictly using provided evidence with page citations."
        llm_answer = call_gemini(system_prompt, prompt)

    # Build deterministic numeric summary (if question mentions 'growth' or 'GDP' or asks years)
    deterministic = {}
    # we fill deterministic for 2024 and 2025 specifically (useful for near-term GDP questions)
    for y in (2024, 2025):
        v = table_map.get(y)
        if v is not None and v != "":
            deterministic[y] = v

    return {
        "llm_answer": llm_answer,
        "retrieved": retrieved,
        "table_values": deterministic
    }


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", dest="q", default=None, help="Question to answer")
    parser.add_argument("--k", dest="k", type=int, default=5)
    args = parser.parse_args()

    if args.q is None:
        q = input("Enter your question: ").strip()
        if not q:
            print("No question provided. Exiting.")
            sys.exit(0)
    else:
        q = args.q

    out = answer_question(q, top_k=args.k)

    print("\n==================== FINAL (LLM) ANSWER ====================\n")
    if out["llm_answer"]:
        print(out["llm_answer"])
    else:
        print("No LLM answer (no retrieved evidence).")

    print("\n==================== DETERMINISTIC TABLE VALUES (Table 1, page 39) ====================\n")
    if out["table_values"]:
        for year, val in out["table_values"].items():
            print(f"{year}: {val}%")
    else:
        print("No deterministic table values found (check data/processed/qatar_table1_corrected.csv).")

    print("\n==================== RETRIEVED CHUNKS (pages) ====================\n")
    for r in out["retrieved"]:
        print(f"(page {r['page']}) {r['text'][:300]}...\n")
