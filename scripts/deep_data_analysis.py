import pandas as pd
import numpy as np
import re
from collections import Counter

# Load P-EXP
df_pexp = pd.read_csv("B:/Promise/Paper2/data/p_exp.csv")

print("="*70)
print("1. PROMISE EXP (P-EXP) DATASET STRUCTURAL PROFILE")
print("="*70)
print(f"Total Rows: {len(df_pexp)}")
print(f"Columns: {list(df_pexp.columns)}")
print(f"Unique Projects (File): {df_pexp['File'].nunique()}")
print(f"Missing Values: {df_pexp.isnull().sum().to_dict()}")

# Project statistics
proj_sizes = df_pexp['File'].value_counts()
print(f"\nProject Size Summary:")
print(f"  Min requirements per project: {proj_sizes.min()}")
print(f"  25th percentile: {proj_sizes.quantile(0.25):.1f}")
print(f"  Median requirements: {proj_sizes.median():.1f}")
print(f"  Mean requirements: {proj_sizes.mean():.2f}")
print(f"  75th percentile: {proj_sizes.quantile(0.75):.1f}")
print(f"  Max requirements per project: {proj_sizes.max()}")

# Class Distribution
print(f"\nClass Distribution across all 969 requirements:")
class_counts = df_pexp['Type'].value_counts()
for c, cnt in class_counts.items():
    print(f"  {c:4s}: {cnt:4d} ({cnt/len(df_pexp)*100:5.2f}%)")

# Linguistic Analysis
print("\n" + "="*70)
print("2. LINGUISTIC & SPECIFICATION SYNTACTIC PROPERTIES")
print("="*70)

reqs = df_pexp['Requirement'].astype(str)

# Length statistics
lengths_words = reqs.apply(lambda x: len(x.split()))
lengths_chars = reqs.apply(lambda x: len(x))
print(f"Word Count per Requirement: Mean={lengths_words.mean():.1f}, Median={lengths_words.median():.1f}, Min={lengths_words.min()}, Max={lengths_words.max()}")
print(f"Char Count per Requirement: Mean={lengths_chars.mean():.1f}, Median={lengths_chars.median():.1f}, Min={lengths_chars.min()}, Max={lengths_chars.max()}")

# Modals analysis (RFC 2119 / ISO 29148 conformance)
modals = ['shall', 'must', 'should', 'will', 'may', 'can']
modal_stats = {m: reqs.str.contains(rf'\b{m}\b', case=False, regex=True).sum() for m in modals}
print("\nModal Verb Distribution (Normative Prescriptiveness):")
for m, cnt in modal_stats.items():
    print(f"  {m:6s}: {cnt:4d} ({cnt/len(df_pexp)*100:5.2f}%)")

# Ambiguity / Vague Adjectives
vague_terms = ['fast', 'intuitive', 'easy', 'appropriate', 'user-friendly', 'efficient', 'reliable', 'scalable', 'secure', 'normal', 'flexible', 'seamless']
vague_counts = {v: reqs.str.contains(rf'\b{v}\b', case=False, regex=True).sum() for v in vague_terms}
print("\nVague / Ambiguous Lexemes (ISO 29148 Non-verifiable descriptors):")
for v, cnt in sorted(vague_counts.items(), key=lambda x: -x[1]):
    if cnt > 0:
        print(f"  {v:14s}: {cnt:3d} ({cnt/len(df_pexp)*100:4.2f}%)")

# Quantitative Bounds (digits, %, seconds, minutes, ms, hours)
has_digit = reqs.str.contains(r'\d+', regex=True).sum()
has_pct = reqs.str.contains(r'%|percent', case=False, regex=True).sum()
has_temporal = reqs.str.contains(r'second|minute|hour|day|ms|sec|hr', case=False, regex=True).sum()
print(f"\nVerifiability & Quantification Indicators:")
print(f"  Contains numerical digits: {has_digit} ({has_digit/len(df_pexp)*100:.2f}%)")
print(f"  Contains percentage metric: {has_pct} ({has_pct/len(df_pexp)*100:.2f}%)")
print(f"  Contains temporal metric:   {has_temporal} ({has_temporal/len(df_pexp)*100:.2f}%)")

# Conditionality
has_condition = reqs.str.contains(r'\b(if|when|unless|whenever|in the event|where)\b', case=False, regex=True).sum()
print(f"  Contains conditional trigger: {has_condition} ({has_condition/len(df_pexp)*100:.2f}%)")

# Cross-Class Co-occurrence in Projects
print("\n" + "="*70)
print("3. CROSS-CLASS CO-OCCURRENCE WITHIN PROJECTS")
print("="*70)
proj_class_matrix = pd.crosstab(df_pexp['File'], df_pexp['Type'])
print("Correlation / Co-occurrence between Requirement Types across the 47 Projects:")
coocc_corr = proj_class_matrix.corr()
print(np.round(coocc_corr, 2).to_string())

# Find top pairwise correlated requirement types across projects
print("\nTop Positive Cross-Type Project Co-occurrences (Architectural Coupling):")
corr_pairs = []
cols = coocc_corr.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        corr_pairs.append((cols[i], cols[j], coocc_corr.iloc[i, j]))
corr_pairs.sort(key=lambda x: -x[2])
for c1, c2, r in corr_pairs[:12]:
    print(f"  {c1:4s} <---> {c2:4s}: r = {r:+.3f}")
