import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation_benchmark import MasterEvaluationBenchmark

def run_sensitivity():
    print("[*] Running Sensitivity Analysis across K in [2, 3, 4, 5, 6, 8]...")
    results = []
    for k in [2, 3, 4, 5, 6, 8]:
        bench = MasterEvaluationBenchmark(target_modules=k)
        df_k = bench.run_full_benchmark()
        results.append({
            "Target_K": k,
            "Mean_Cohesion_Q": round(float(df_k["AlphaReq_Cohesion"].mean()), 4),
            "Mean_Coupling_C": round(float(df_k["AlphaReq_Coupling"].mean()), 4),
            "Mean_CSR_pct": round(float(df_k["AlphaReq_CSR"].mean() * 100), 2),
            "Mean_Fitness": round(float(df_k["AlphaReq_Fitness"].mean()), 4)
        })

    df_sens = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS: ARCHITECTURAL FITNESS ACROSS K MODULES")
    print("="*60)
    print(df_sens.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    df_sens.to_csv("results/sensitivity_analysis_target_modules.csv", index=False)
    print("\n[+] Saved: results/sensitivity_analysis_target_modules.csv")

if __name__ == "__main__":
    run_sensitivity()
