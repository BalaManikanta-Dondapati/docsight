# tools/fix_table1_csv.py
import csv
from pathlib import Path

IN = Path("data/processed/qatar_table1_from_chunks.csv")
OUT = Path("data/processed/qatar_table1_corrected.csv")

if not IN.exists():
    raise SystemExit(f"Input CSV not found: {IN}")

# read the existing CSV (simple)
rows = []
with open(IN, "r", encoding="utf-8") as f:
    for r in csv.reader(f):
        rows.append(r)

if not rows or len(rows) < 2:
    raise SystemExit("Unexpected CSV content.")

years = rows[0][1:]  # header row minus first column label
values = rows[1][1:]  # the values row minus first column label

# if first token is '2018' (detected), drop it
if values and values[0].strip() == "2018":
    values = values[1:]

# pad values to match years length (put empty for missing)
if len(values) < len(years):
    values = values + [""] * (len(years) - len(values))

# write corrected CSV
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Indicator"] + years)
    writer.writerow(["Real GDP (percent, yoy)"] + values)

print("Wrote corrected CSV:", OUT)
print("Years:", years)
print("Values (aligned):", values)
