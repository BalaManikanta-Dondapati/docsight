# DocSight — Qatar IMF Report RAG

**Project:** Multi-Modal RAG QA for the Qatar IMF Article IV Report (2025)  
**Author:** <Your Name> — <email>  
**Assignment:** Multi-Modal RAG QA System (Candidate Assignment)

## What this repo contains
- `tools/streamlit_rag_app.py` — Streamlit demo UI (ingestion already done; FAISS & chunks consumed)
- `data/processed/` — FAISS index, chunk JSON, and table CSV (not all may be included for size)
- `notebooks/` — (optional) preprocessing / chunking scripts
- `requirements.txt` — Python dependencies
- `report.pdf` — 2-page technical report (architecture + choices)
- `demo_video.mp4` — 3–5 minute screen recording

## Deliverables (per assignment)
- Codebase (this repo) — modular ingestion/retrieval/LLM
- Demo Application — Streamlit UI (`tools/streamlit_rag_app.py`)
- Technical Report — `report.pdf`
- Video Demonstration — `demo_video.mp4`

## Quickstart (local)
1. Create & activate venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows
Install:

bash
Copy code
pip install -r requirements.txt
Set environment variables (optional; required for Gemini LLM):

bash
Copy code
export GEMINI_API_KEY="your_key_here"
export GEMINI_MODEL="gemini-2.5-flash"   # recommended
On Windows (PowerShell):

powershell
Copy code
setx GEMINI_API_KEY "your_key_here"
setx GEMINI_MODEL "gemini-2.5-flash"
Run demo:

bash
Copy code
streamlit run tools/streamlit_rag_app.py
Open browser at http://localhost:8501.

Notes & tips
Gemini model: I recommend gemini-2.5-flash (set via GEMINI_MODEL env var). If Gemini SDK or key is missing the app still runs deterministically (retrieval + deterministic table outputs).

Embedding: sentence-transformers/all-mpnet-base-v2 is preferred; a smaller fallback (all-MiniLM-L6-v2) is used when needed. The code forces CPU by default to avoid meta-tensor errors on machines without GPU setup.

FAISS & chunks: keep data/processed/faiss_index.idx and data/processed/qatar_chunks.json alongside the code or provide instructions to re-run ingestion if you cannot include binary indexes.

CSV Table: data/processed/qatar_table1_corrected.csv used for deterministic table outputs and downloads.

Privacy / keys: do not commit GEMINI_API_KEY to GitHub. Use GitHub Secrets for CI or .env locally (and add to .gitignore).