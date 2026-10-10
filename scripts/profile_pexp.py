import os
import pandas as pd
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(repo_root, "data", "Promise_Exp.csv")
if not os.path.exists(data_path) and os.path.exists(r"B:\Promise_Exp.csv"):
    data_path = r"B:\Promise_Exp.csv"
df = pd.read_csv(data_path)

print("="*60)
print("P-EXP DATASET DEEP PROFILE")
print("="*60)
print(f"Total Requirements: {len(df)}")
print(f"Total Projects: {df['File'].nunique()}")

print("\n--- Project Size Summary ---")
vc_file = df['File'].value_counts()
print(vc_file.describe())

print("\n--- Class Distribution ---")
vc_type = df['Type'].value_counts()
for t, cnt in vc_type.items():
    print(f"  {t:4s}: {cnt:4d} ({cnt/len(df)*100:5.2f}%)")

print("\n--- Project Composition (Top 10 Projects) ---")
for pid in vc_file.head(10).index:
    sub = df[df['File'] == pid]
    counts = dict(sub['Type'].value_counts())
    print(f"Project {pid:2d} (n={len(sub):2d}): {counts}")

print("\n--- Sample Requirements across Key Classes ---")
for t in ['F', 'SE', 'PE', 'US', 'FT', 'SC', 'A', 'MN', 'LF', 'O', 'L', 'PO']:
    samples = df[df['Type'] == t]['Requirement'].head(2).tolist()
    print(f"\n[{t}] (Total {vc_type.get(t, 0)}):")
    for s in samples:
        print(f"  * {s.strip()}")
