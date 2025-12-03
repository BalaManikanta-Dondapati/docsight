# -*- coding: utf-8 -*-
import json

p = "data/processed/qatar_pages_enhanced.json"
j = json.load(open(p, "r", encoding="utf-8"))
page1 = j[0]

print("=== PAGE 1: text (first 300 chars) ===")
print(page1.get("text","")[:300].replace("\n"," "))

print("\n=== PAGE 1: ocr_text (first 300 chars) ===")
print(page1.get("ocr_text","")[:300].replace("\n"," "))