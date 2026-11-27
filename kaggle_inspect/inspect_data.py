"""Phase 1 — discover the real mount tree, then read any CSVs found.

Walks /kaggle/input but PRUNES at depth 4 so it never descends into the
image directories (which hold ~350k files). Prints directory structure and
reads any *.csv it finds near the top.
"""
import os
import pandas as pd

ROOT = "/kaggle/input"
MAX_DEPTH = 4
found_csvs = []

print("=== TREE (depth-limited) ===")
for dirpath, dirnames, filenames in os.walk(ROOT):
    depth = dirpath[len(ROOT):].count(os.sep)
    csvs = [f for f in filenames if f.lower().endswith(".csv")]
    for c in csvs:
        found_csvs.append(os.path.join(dirpath, c))
    shown_files = (csvs or filenames[:4])
    print(f"{'  ' * depth}{dirpath}  | dirs={dirnames[:8]} | files={shown_files[:6]}")
    if depth >= MAX_DEPTH:
        dirnames[:] = []            # prune: do not descend further

print("\n=== CSVs FOUND ===")
for p in found_csvs:
    print("  ", p, f"({os.path.getsize(p)//1024} KB)")

for p in found_csvs:
    print("\n" + "=" * 74)
    print("CSV:", p)
    try:
        df = pd.read_csv(p, nrows=50000)
    except Exception as e:
        print("  read error:", e)
        continue
    print("  columns:", list(df.columns))
    print("  sampled rows:", len(df))
    if "Finding Labels" in df.columns:
        vocab = set()
        for v in df["Finding Labels"].dropna():
            for f in str(v).split("|"):
                vocab.add(f.strip())
        print("  >> NIH Finding-Labels vocab:", sorted(vocab))
    chex = [c for c in ["Cardiomegaly", "Edema", "Consolidation", "Pneumonia",
                        "Atelectasis", "Pneumothorax", "Pleural Effusion",
                        "Lung Opacity", "No Finding"] if c in df.columns]
    if chex:
        print("  >> CheXpert-style label columns:", chex)
    for col in ["class_name", "label", "finding", "diagnosis"]:
        if col in df.columns:
            print(f"  >> '{col}' vocab:", sorted(map(str, df[col].dropna().unique())))
    for col in ["View Position", "Frontal/Lateral", "AP/PA", "view"]:
        if col in df.columns:
            print(f"  >> '{col}' counts:", df[col].value_counts().head(10).to_dict())
    print("  first row:", df.iloc[0].to_dict() if len(df) else None)

print("\n[inspect] done.")
