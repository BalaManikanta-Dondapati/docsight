# tools/extract_table1_from_chunks.py
import json, re, csv
from pathlib import Path

CHUNKS = Path("data/processed/qatar_chunks.json")
OUT_CSV = Path("data/processed/qatar_table1_from_chunks.csv")

if not CHUNKS.exists():
    raise SystemExit("qatar_chunks.json not found. Run the chunker first (ingest/chunker.py).")

chunks = json.load(open(CHUNKS, "r", encoding="utf-8"))

# collect chunks from page 39 (some tools use page index 39)
page39_chunks = [c for c in chunks if c.get("page") == 39]
if not page39_chunks:
    # fallback: search for chunks that mention "Selected Macroeconomic Indicators" or "Table 1"
    page39_chunks = [c for c in chunks if ("Table 1" in c.get("text","") or "Selected Macroeconomic Indicators" in c.get("text","") or "Real GDP" in c.get("text",""))]

if not page39_chunks:
    print("No likely Table 1 chunks found in the chunk JSON. Try running the chunker again.")
    raise SystemExit(0)

# join chunks in order
page39_chunks = sorted(page39_chunks, key=lambda x: x.get("start_char", 0))
text = " ".join([c["text"] for c in page39_chunks])
text = " ".join(text.split())  # normalize whitespace

# Find year header (2020..2029) - flexible detection
years_match = re.search(r"(20\d{2})(?:\D+?(20\d{2})){2,}", text)
years_list = []
if years_match:
    # heuristically extract the sequence of years near the top of the chunk
    # attempt to find the long header like "2020 2021 2022 2023 2024 2025 2026 2027 2028 2029"
    header_match = re.search(r"(2020(?:\D+\d{4}){3,9})", text)
    if header_match:
        header_text = header_match.group(1)
        years_list = re.findall(r"20\d{2}", header_text)
else:
    # fallback search for any 2020..2029 years in text
    years_list = re.findall(r"20\d{2}", text)[:10]

if not years_list:
    print("Could not detect header years automatically. Showing page39 sample for manual check:\n")
    print(text[:2000])
    raise SystemExit(0)

# Now locate the Real GDP row: search for 'Real GDP' and then collect numbers to the right
row_regex = re.search(r"(Real\s+GDP[^\n\r\f]*)", text, re.IGNORECASE)
numbers = []
if row_regex:
    # take a window of text after the row label to capture numeric columns
    start = row_regex.start()
    window = text[start:start+1200]  # capture next part (heuristic)
    # extract numbers (floats) in the window
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", window)
else:
    # fallback: search for a line that starts with Real or 'Real GDP' anywhere
    # try splitting into tokens and scanning for 'Real' then numbers after it
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok.lower().startswith("real"):
            # collect the next 15 tokens and find numbers
            next_seq = " ".join(tokens[i:i+30])
            numbers = re.findall(r"-?\d+\.\d+|-?\d+", next_seq)
            if numbers:
                break

if not numbers:
    print("Could not automatically extract numeric values for Real GDP. Here's a larger sample for inspection:\n")
    print(text[:4000])
    raise SystemExit(0)

# Keep only as many numbers as years found
nums_clean = numbers[:len(years_list)]

# Prepare CSV rows: header years + values
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Indicator"] + years_list)
    writer.writerow(["Real GDP (percent, yoy)"] + nums_clean)

print(f"Detected years: {years_list}")
print(f"Detected Real GDP values: {nums_clean}")
print(f"Saved CSV to: {OUT_CSV}")
print("\nIf numbers look misaligned, paste the printed sample above and I will adjust the parser.")
