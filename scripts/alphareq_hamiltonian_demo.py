import os
import pandas as pd
import numpy as np
import scipy.linalg as la

# Load the Promise_Exp dataset
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(repo_root, "data", "Promise_Exp.csv")
if not os.path.exists(data_path) and os.path.exists(r"B:\Promise_Exp.csv"):
    data_path = r"B:\Promise_Exp.csv"
df = pd.read_csv(data_path)

print("="*75)
print("ALPHAREQ EMPIRICAL DEMONSTRATION: REQUIREMENTS HAMILTONIAN & FRUSTRATION")
print("="*75)

# 1. Compute empirical coupling tensor J_ij across requirement types
# We evaluate co-occurrence and semantic tension between classes
types = sorted(df['Type'].unique().tolist())
k = len(types)
print(f"Requirement Classes ({k}): {types}")

# Build project-by-type frequency matrix
proj_type = pd.crosstab(df['File'], df['Type'])

# Normalized frequency per project
proj_type_norm = proj_type.div(proj_type.sum(axis=1), axis=0)

# Correlation matrix as baseline coupling
R = proj_type.corr().fillna(0).values

# In Requirements Engineering domain physics:
# Functional (F) with any NFR adds implementation load (coupling J > 0 for structure)
# But conflicting NFRs (Security vs Performance, Usability vs Security) introduce negative (antiferromagnetic) tension!
# Let's define the theoretical-empirical interaction signs based on domain trade-off semantics:
# Tension pairs (SE vs PE, SE vs US, PE vs FT, etc.)
tension_penalties = {
    ('PE', 'SE'): -0.65, # Cryptographic security introduces latency
    ('SE', 'US'): -0.55, # Authentication/authorization hurdles reduce usability
    ('PE', 'FT'): -0.40, # Redundancy/replication increases synchronization delay
    ('SC', 'PE'): -0.35, # Distributed scalability introduces network overhead
    ('LF', 'PE'): -0.20, # Rich graphical rendering costs frame time
    ('MN', 'PE'): -0.30, # Heavy abstraction layers increase indirection
}

# Construct the Generalized Interaction Matrix J_class
J_class = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        if i == j:
            J_class[i, j] = 0.0
        else:
            t1, t2 = types[i], types[j]
            # Check if there is an inherent domain tension
            pair = tuple(sorted([t1, t2]))
            pair_rev = (t2, t1)
            penalty = 0.0
            if (t1, t2) in tension_penalties:
                penalty = tension_penalties[(t1, t2)]
            elif (t2, t1) in tension_penalties:
                penalty = tension_penalties[(t2, t1)]
            
            # Blend empirical co-occurrence with domain tension
            r_val = R[i, j]
            if penalty < 0:
                J_class[i, j] = penalty * (1.0 + abs(r_val))
            else:
                J_class[i, j] = 0.4 * r_val

print("\n--- Empirical-Semantic Coupling Matrix J (Classes) ---")
df_J = pd.DataFrame(np.round(J_class, 2), index=types, columns=types)
print(df_J.to_string())

# 2. Compute the Project-Level Hamiltonian, Frustration Index, and Phase State across all 47 projects
print("\n" + "="*75)
print("PROJECT-LEVEL STATISTICAL MECHANICS (HAMILTONIAN H, FRUSTRATION PHI, PHASE)")
print("="*75)

project_results = []
for pid in sorted(df['File'].unique()):
    p_df = df[df['File'] == pid]
    n_req = len(p_df)
    counts = p_df['Type'].value_counts()
    
    # Priority vector h (modals weighting)
    # Shall/must has high normative field (+1.0), should (+0.5), may (+0.2)
    req_texts = p_df['Requirement'].str.lower()
    normative_weights = req_texts.apply(lambda x: 1.0 if ('shall' in x or 'must' in x) else (0.5 if 'should' in x else 0.2)).values
    
    # Class composition vector
    v = np.zeros(k)
    for i, t in enumerate(types):
        v[i] = counts.get(t, 0)
    
    # Project-level interaction energy E_int = -0.5 * v^T * J * v
    # Total architectural tension / Hamiltonian
    E_int = 0.5 * np.dot(v, np.dot(J_class, v)) # Positive tension when J < 0
    H_norm = np.sum(normative_weights)
    
    # Frustration calculation: proportion of competing pairs
    competing_pairs = 0
    total_pairs = n_req * (n_req - 1) / 2 if n_req > 1 else 1
    
    # Number of SE, PE, US in project
    n_se = counts.get('SE', 0)
    n_pe = counts.get('PE', 0)
    n_us = counts.get('US', 0)
    n_sc = counts.get('SC', 0)
    n_ft = counts.get('FT', 0)
    
    frustrated_interactions = (n_se * n_pe) + (n_se * n_us) + (n_pe * n_ft) + (n_sc * n_pe)
    frustration_index = frustrated_interactions / total_pairs if total_pairs > 0 else 0.0
    
    # Phase Boundary Criterion:
    # Critical constraint density alpha = frustrated / n_req
    alpha = frustrated_interactions / n_req if n_req > 0 else 0
    # Critical threshold alpha_c = 1.8 (transition to non-ergodic spin-glass phase)
    phase = "Glassy/Frustrated (High Risk)" if alpha > 1.8 else ("Marginal (Moderate)" if alpha > 0.8 else "Paramagnetic (Low Risk)")
    
    project_results.append({
        'Project_ID': pid,
        'N_Reqs': n_req,
        'SE': n_se,
        'PE': n_pe,
        'US': n_us,
        'Frustrated_Pairs': frustrated_interactions,
        'Frustration_Index': frustration_index,
        'Alpha_Constraint_Density': alpha,
        'Phase_State': phase
    })

res_df = pd.DataFrame(project_results)
print(res_df.sort_values(by='Alpha_Constraint_Density', ascending=False).head(15).to_string(index=False))

print("\n--- Summary of Architectural Phase Distribution across 47 Projects ---")
print(res_df['Phase_State'].value_counts().to_string())
