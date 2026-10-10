import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset_loader import PExpDataLoader
from src.conflict_detector import ConflictDetector

def export_artifacts():
    os.makedirs("results", exist_ok=True)
    loader = PExpDataLoader()
    detector = ConflictDetector()

    # 1. Export Trade-off penalties lookup table (J_ij ground truth values)
    coupling_data = []
    for (t1, t2), weight in detector.penalties.items():
        coupling_data.append({
            "Type_1": t1,
            "Type_2": t2,
            "Coupling_J": weight,
            "Nature": "Antiferromagnetic (Conflict / Tension)"
        })
    for (t1, t2), weight in detector.bonuses.items():
        coupling_data.append({
            "Type_1": t1,
            "Type_2": t2,
            "Coupling_J": weight,
            "Nature": "Ferromagnetic (Synergy / Reinforcement)"
        })

    df_couplings = pd.DataFrame(coupling_data)
    df_couplings.to_csv("results/interaction_tensor_lookup_table.csv", index=False)
    print("[+] Saved: results/interaction_tensor_lookup_table.csv")

    # 2. Export Project 8 (Highest complexity: 92 reqs, 507 conflicts) sample J_ij matrix and metadata
    p8 = loader.get_project(8)
    J8, conflicts8 = detector.compute_coupling_tensor(p8)
    np.save("results/project_8_coupling_tensor_J.npy", J8)
    pd.DataFrame(J8).to_csv("results/project_8_coupling_tensor_J.csv", index=False, header=False)
    print(f"[+] Saved Project 8 coupling matrix J (shape: {J8.shape}) to results/project_8_coupling_tensor_J.csv")

    with open("results/project_8_conflicts.json", "w") as f:
        json.dump(conflicts8, f, indent=2)
    print(f"[+] Saved Project 8 conflicts list ({len(conflicts8)} items) to results/project_8_conflicts.json")

if __name__ == "__main__":
    export_artifacts()
