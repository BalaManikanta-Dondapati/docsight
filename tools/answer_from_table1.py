# tools/answer_from_table1.py
import csv
from pathlib import Path

CSV = Path("data/processed/qatar_table1_corrected.csv")
if not CSV.exists():
    raise SystemExit("Corrected CSV not found. Run tools/fix_table1_csv.py first.")

with open(CSV, "r", encoding="utf-8") as f:
    reader = list(csv.reader(f))
    header = reader[0][1:]
    values = reader[1][1:]

# build mapping
mapping = {int(year): (val if val!="" else None) for year, val in zip(header, values)}

def print_summary():
    v2024 = mapping.get(2024)
    v2025 = mapping.get(2025)
    print("Deterministic answer (from Table 1, page 39):")
    print(f"  2024: {v2024}%")
    print(f"  2025: {v2025}%")
    # staff summary
    print("\nStaff summary phrase (page 13): 'improve to about 2% in 2024–25' — consistent with 1.7% (2024) and 2.4% (2025).")

if __name__ == "__main__":
    print_summary()
