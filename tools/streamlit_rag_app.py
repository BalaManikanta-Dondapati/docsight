# tools/streamlit_rag_app.py
"""
Streamlit RAG UI for the Qatar IMF Report (clean, robust, production-friendly).

Key improvements:
- Clean UI (compact sidebar; no big info/success banners)
- Safe SentenceTransformer loading forced to CPU with fallback
- FAISS retrieval
- Robust Gemini call attempts; GEMINI model & key override via env
- Robust CSV parsing for Table 1
"""
import os
import json
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import faiss

# Try import Google Gemini SDK (may be installed or not)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# sentence-transformers (may raise if not installed)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ============ CONFIG ============
# Base project directory (resolve relative paths reliably)
BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "qatar_chunks.json"
FAISS_PATH = BASE_DIR / "data" / "processed" / "faiss_index.idx"
TABLE_PATH = BASE_DIR / "data" / "processed" / "qatar_table1_corrected.csv"

# Embedding models
PREFERRED_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini defaults (override with env GEMINI_MODEL or GEMINI_API_KEY)
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ============ STREAMLIT SETUP ============
st.set_page_config(page_title="DocSight – Qatar IMF RAG", layout="wide")
st.title("📘 DocSight — Qatar IMF Report RAG")
st.write("Ask questions about the Qatar IMF Article IV Report (2025).")

# ============ Sidebar status (compact) ============
with st.sidebar:
    st.header("Status")
    st.write("Embedding model: —")
    st.write(f"FAISS index: {'found' if FAISS_PATH.exists() else 'missing'}")
    st.write(f"Table CSV: {'found' if TABLE_PATH.exists() else 'missing'}")
    st.write(f"Gemini SDK: {'installed' if genai is not None else 'missing'}")
    st.write("")
    st.caption("Tip: set GEMINI_API_KEY and GEMINI_MODEL env vars to enable LLM replies.")

# ============ Helpers: read files ============
def safe_read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return None


@st.cache_resource
def load_chunks_json():
    return safe_read_json(CHUNKS_PATH)


@st.cache_resource
def load_faiss_index():
    if not FAISS_PATH.exists():
        return None
    try:
        return faiss.read_index(str(FAISS_PATH))
    except Exception:
        return None


# ============ Robust CSV loader for Table 1 ============
@st.cache_resource
def load_table_df():
    """Try to normalize the CSV into a one-row wide style (Indicator + year columns)."""
    if not TABLE_PATH.exists():
        return None
    try:
        df = pd.read_csv(TABLE_PATH)
    except Exception:
        return None

    # Normalize column names to str
    df.columns = [str(c) for c in df.columns]
    cols_lower = [c.lower() for c in df.columns]

    # Long format pivot
    if "year" in cols_lower:
        year_col = df.columns[cols_lower.index("year")]
        val_cols = [c for c in df.columns if c != year_col]
        if val_cols:
            try:
                pivot = df[[year_col, val_cols[0]]].dropna()
                pivot = pivot.set_index(year_col).T
                pivot.columns = [str(c) for c in pivot.columns]
                pivot.insert(0, "Indicator", val_cols[0])
                return pivot.reset_index(drop=True)
            except Exception:
                pass

    # Wide format: find 'Real GDP' row
    first_col = df.columns[0]
    try:
        mask = df[first_col].astype(str).str.lower().str.contains("real gdp") | \
               df[first_col].astype(str).str.lower().str.contains("real_gdp") | \
               df[first_col].astype(str).str.lower().str.contains("real ")
        if mask.any():
            row = df[mask].iloc[0:1].copy()
            row = row.rename(columns={first_col: "Indicator"})
            return row.reset_index(drop=True)
    except Exception:
        pass

    if len(df) == 1:
        return df.reset_index(drop=True)

    return df


def table_lookup():
    """Return dict mapping year(int) -> value (string) for the Real GDP row."""
    df = load_table_df()
    if df is None:
        return {}

    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    mapping = {}

    if "indicator" in cols_lower:
        idx = cols_lower.index("indicator")
        indicator_col = cols[idx]
        # find real GDP row (use enumerate so iloc index is correct)
        real_row = None
        for row_i, val in enumerate(df[indicator_col].astype(str).tolist()):
            if "real gdp" in val.lower() or "real_gdp" in val.lower() or val.strip().lower() == "real":
                real_row = df.iloc[row_i]
                break
        if real_row is None:
            real_row = df.iloc[0]
        for c in df.columns:
            if c == indicator_col:
                continue
            key = str(c).strip()
            if key.isdigit() and len(key) == 4:
                v = real_row[c]
                mapping[int(key)] = (str(v).strip() if pd.notna(v) else None)
        return mapping

    # single-row with year-named columns
    if len(df) == 1:
        row = df.iloc[0]
        for c in df.columns:
            key = str(c).strip()
            if key.isdigit() and len(key) == 4:
                v = row[c]
                mapping[int(key)] = (str(v).strip() if pd.notna(v) else None)
        return mapping

    return {}


# ============ Embedding model loader (safe) ============
@st.cache_resource
def load_embed_model(force_cpu: bool = True):
    """
    Load the preferred embedding model. Force CPU to avoid meta-tensor errors on systems
    without proper GPU setup. Fallback to a smaller model on failure.
    Returns (model, model_name) or (None, None) on failure.
    """
    if SentenceTransformer is None:
        return None, None

    # Force CPU explicitly: this avoids meta-tensor / .to() issues when GPU not configured
    device = "cpu" if force_cpu else None
    try:
        if device:
            model = SentenceTransformer(PREFERRED_EMBED_MODEL, device=device)
        else:
            model = SentenceTransformer(PREFERRED_EMBED_MODEL)
        return model, PREFERRED_EMBED_MODEL
    except Exception as e_pref:
        try:
            if device:
                model = SentenceTransformer(FALLBACK_EMBED_MODEL, device=device)
            else:
                model = SentenceTransformer(FALLBACK_EMBED_MODEL)
            return model, FALLBACK_EMBED_MODEL
        except Exception:
            return None, None


# ============ Load resources (cached) ============
chunks = load_chunks_json()
faiss_index = load_faiss_index()
embed_model, embed_model_name = load_embed_model()
table_map = table_lookup()

# Update the sidebar with actual model name
with st.sidebar:
    try:
        st.write(f"Embedding model: {embed_model_name if embed_model_name else 'missing'}")
    except Exception:
        st.write("Embedding model: unknown")

# ============ Utility: embed and retrieve ============
def embed_query(query):
    if embed_model is None:
        raise RuntimeError("Embedding model not loaded")
    vec = embed_model.encode([query], convert_to_numpy=True)
    # normalize
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    vec = vec / norm
    return vec.astype("float32")


def retrieve_top_k(query, k=5):
    if faiss_index is None or chunks is None:
        return []
    try:
        qv = embed_query(query)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return []
    try:
        D, I = faiss_index.search(qv, k)
    except Exception as e:
        st.error(f"FAISS search failed: {e}")
        return []
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx].copy()
        c["score"] = float(score)
        results.append(c)
    return results


# ============ Gemini prompt + robust call ============
def build_prompt(question, retrieved, table_vals=None):
    evidence_blocks = []
    for r in retrieved:
        txt = r.get("text", "")
        short = txt if len(txt) <= 1200 else (txt[:1200] + " ...")
        evidence_blocks.append(f"[page: {r.get('page')}] {short}")
    evidence = "\n\n".join(evidence_blocks)

    table_lines = ""
    if table_vals:
        lines = [f"Table 1 (page 39): {y} = {v}%" for y, v in table_vals.items()]
        if lines:
            table_lines = "DETERMINISTIC TABLE VALUES:\n" + "\n".join(lines) + "\n\n"

    prompt = f"""
You are a factual assistant. Use only the evidence below to answer the user's question.
If the answer is not present in the evidence, respond exactly: "I don't know based on the provided evidence."
Cite pages in parentheses after factual sentences.

{table_lines}
EVIDENCE:
{evidence}

QUESTION:
{question}

INSTRUCTIONS:
1) Short answer: one sentence with numeric answer(s) and page citation(s).
2) Explanation: up to two short sentences (add page citations).
3) Sources: bullet list of pages used.
"""
    return prompt


def call_gemini(prompt, model_name: str = DEFAULT_GEMINI_MODEL):
    """
    Attempt several ways of calling the Gemini SDK to maximize compatibility across versions.
    Returns string output or None.
    """
    if genai is None:
        st.info("Gemini SDK not installed.")
        return None
    if not GEMINI_API_KEY:
        st.info("GEMINI_API_KEY not set.")
        return None

    # Ensure SDK configured (some versions require configure())
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

    errors = []
    # Attempt #1: GenerativeModel -> generate_content (new style)
    try:
        if hasattr(genai, "GenerativeModel"):
            model = genai.GenerativeModel(model_name)
            out = model.generate_content(prompt)
            if hasattr(out, "text") and out.text:
                return out.text.strip()
            if hasattr(out, "candidates") and out.candidates:
                cand = out.candidates[0]
                if hasattr(cand, "content"):
                    try:
                        return cand.content[0].text.strip()
                    except Exception:
                        pass
                if hasattr(cand, "text"):
                    return cand.text.strip()
            return str(out)
    except Exception as e:
        errors.append(f"GenModel error: {e}")

    # Attempt #2: generate_text (older style)
    try:
        if hasattr(genai, "generate_text"):
            out2 = genai.generate_text(model=model_name, prompt=prompt)
            if isinstance(out2, str):
                return out2.strip()
            if hasattr(out2, "text") and out2.text:
                return out2.text.strip()
            if hasattr(out2, "candidates") and out2.candidates:
                try:
                    return out2.candidates[0].text.strip()
                except Exception:
                    return str(out2)
    except Exception as e:
        errors.append(f"generate_text error: {e}")

    # Attempt #3: new `responses` wrapper
    try:
        if hasattr(genai, "responses"):
            out3 = genai.responses.generate(model=model_name, input=prompt)
            try:
                if hasattr(out3, "output"):
                    for item in getattr(out3, "output", []):
                        if isinstance(item, dict) and "content" in item:
                            for c in item["content"]:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    return c.get("text", "").strip()
                return str(out3)
            except Exception:
                return str(out3)
    except Exception as e:
        errors.append(f"responses.generate error: {e}")

    # If all attempts failed, show aggregated errors for debugging
    st.error("Gemini call failed. Attempts:\n" + "\n".join(errors))
    return None


# ============ UI: Query flow ============
query = st.text_input("Enter your question", value="What is Qatar's projected GDP growth for 2024-25?")
run = st.button("Run Query")

if run:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown("### 🔎 Retrieval")
        retrieved = retrieve_top_k(query, k=5)
        st.write(f"Retrieved {len(retrieved)} chunks.")

        st.markdown("### 🤖 LLM (Gemini) Answer (if configured)")
        prompt = build_prompt(query, retrieved, table_map)

        gemini_model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        result_text = None
        if genai is not None and GEMINI_API_KEY:
            with st.spinner("Calling Gemini..."):
                try:
                    result_text = call_gemini(prompt, model_name=gemini_model_name)
                except Exception as e:
                    st.error(f"Gemini call raised an exception: {e}\n{traceback.format_exc()}")
                    result_text = None
            if result_text:
                st.markdown(result_text)
            else:
                st.info("Gemini result not available. Check GEMINI_API_KEY, GEMINI_MODEL, or SDK version.")
        else:
            st.info("Gemini not configured or SDK not installed; deterministic answers only.")

        # Deterministic table values
        st.markdown("### 📊 Deterministic Table Values (Table 1, page 39)")
        if table_map:
            for y in sorted(table_map.keys()):
                st.write(f"**{y}: {table_map[y]}%**")
        else:
            st.write("No deterministic table values available (check CSV).")

        # Retrieved evidence
        st.markdown("### 📚 Retrieved Evidence (Top chunks)")
        if retrieved:
            for r in retrieved:
                with st.expander(f"Page {r.get('page')} | Score {r.get('score'):.4f}"):
                    st.write(r.get("text", ""))
        else:
            st.write("No evidence retrieved (check FAISS + chunks).")

        # Show table preview
        st.markdown("### 📁 Table CSV preview")
        df_table = load_table_df()
        if df_table is not None:
            try:
                st.dataframe(df_table.astype(str), use_container_width=True)
                csv_bytes = df_table.to_csv(index=False).encode("utf-8")
                st.download_button("Download Table CSV", csv_bytes, file_name="table1_preview.csv", mime="text/csv")
            except Exception:
                st.write("Could not display CSV preview.")
        else:
            st.write("No table CSV found.")
