import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

# Add src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from src.evaluation_benchmark import MasterEvaluationBenchmark
from src.generate_publication_figures import generate_all_figures
from src.utils import setup_logger, safe_save_dataframe, safe_save_json

logger = setup_logger("AlphaReq.CLI")

def parse_args():
    parser = argparse.ArgumentParser(
        description="AlphaReq: Automated Conflict Resolution and Autonomous Software Architecture Synthesis from Requirements."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="B:/Promise/Paper2/data/p_exp.csv",
        help="Path to the P-EXP dataset CSV file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.environ.get("TEMP", r"C:\Users\umert\AppData\Local\Temp\opencode"), "alphareq_output"),
        help="Directory to save benchmark summary CSVs and plots."
    )
    parser.add_argument(
        "--target-modules",
        type=int,
        default=4,
        help="Number of target microservices/modules for architectural partitioning."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generation of publication figures."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("=" * 75)
    logger.info("       ALPHAREQ MASTER PIPELINE: REPRODUCIBLE EXPERIMENTAL BENCHMARK")
    logger.info("=" * 75)
    logger.info(f"Target Corpus: {args.data_path}")
    logger.info(f"Output Directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Run Master Evaluation Benchmark
    benchmark = MasterEvaluationBenchmark(data_path=args.data_path, target_modules=args.target_modules)
    df_results = benchmark.run_full_benchmark()

    # Save CSV
    csv_target = os.path.join(args.output_dir, "benchmark_summary.csv")
    safe_save_dataframe(df_results, csv_target)

    # 2. Run Statistical Tests
    stats_tests = benchmark.compute_statistical_tests(df_results)
    json_target = os.path.join(args.output_dir, "statistical_tests.json")
    safe_save_json(stats_tests, json_target)

    # 3. Export Sample Docker Compose Blueprint for the most complex project
    if not df_results.empty:
        max_conflict_pid = int(df_results.sort_values(by='N_Conflicts_Identified', ascending=False).iloc[0]['Project_ID'])
        p_complex = benchmark.loader.get_project(max_conflict_pid)
        J_c, _ = benchmark.detector.compute_coupling_tensor(p_complex)
        h_c = p_complex['Normative_Weight'].values
        best_sigma_c, _, _ = benchmark.solver.solve_ground_state(J_c, h_c)
        sample_arch = benchmark.synthesizer.fold_architecture(p_complex, J_c, best_sigma_c)
        compose_path = os.path.join(args.output_dir, f"project_{max_conflict_pid}_docker_compose.yml")
        benchmark.synthesizer.export_docker_compose(sample_arch, compose_path)
        logger.info(f"Synthesized deployable Docker Compose blueprint for Project {max_conflict_pid}: {compose_path}")

    # 4. Print Macro Comparison Table
    logger.info("\n" + "=" * 80)
    logger.info("                     OVERALL PERFORMANCE COMPARISON TABLE")
    logger.info("=" * 80)

    macro_summary = {
        'Evaluation Metric': [
            'Conflict Detection Recall (%)',
            'Architectural Cohesion (Q)',
            'Inter-Service Coupling Density (C)',
            'Constraint Satisfaction Rate (CSR %)',
            'Composite Architectural Fitness'
        ],
        'Base Paper 1 (Ezzini 2022)': [
            f"{df_results['Ezzini_Conflict_Recall'].mean()*100:.2f}%",
            f"{df_results['Ezzini_Cohesion'].mean():.3f}",
            f"{df_results['Ezzini_Coupling'].mean():.3f}",
            f"{df_results['Ezzini_CSR'].mean()*100:.2f}%",
            f"{df_results['Ezzini_Fitness'].mean():.3f}"
        ],
        'Base Paper 2 (Al-Safadi 2023)': [
            f"{df_results['AlSafadi_Conflict_Recall'].mean()*100:.2f}%",
            f"{df_results['AlSafadi_Cohesion'].mean():.3f}",
            f"{df_results['AlSafadi_Coupling'].mean():.3f}",
            f"{df_results['AlSafadi_CSR'].mean()*100:.2f}%",
            f"{df_results['AlSafadi_Fitness'].mean():.3f}"
        ],
        'AlphaReq (This Paper)': [
            f"{df_results['AlphaReq_Conflict_Recall'].mean()*100:.2f}%",
            f"{df_results['AlphaReq_Cohesion'].mean():.3f}",
            f"{df_results['AlphaReq_Coupling'].mean():.3f}",
            f"{df_results['AlphaReq_CSR'].mean()*100:.2f}%",
            f"{df_results['AlphaReq_Fitness'].mean():.3f}"
        ],
        'Relative Gain vs Base Paper 2': [
            f"+{((df_results['AlphaReq_Conflict_Recall'].mean() - df_results['AlSafadi_Conflict_Recall'].mean())*100):.2f}%",
            f"+{((df_results['AlphaReq_Cohesion'].mean() - df_results['AlSafadi_Cohesion'].mean())/df_results['AlSafadi_Cohesion'].mean()*100):.2f}%",
            f"-{((df_results['AlSafadi_Coupling'].mean() - df_results['AlphaReq_Coupling'].mean())/df_results['AlSafadi_Coupling'].mean()*100):.2f}%",
            f"+{((df_results['AlphaReq_CSR'].mean() - df_results['AlSafadi_CSR'].mean())*100):.2f}%",
            f"+{((df_results['AlphaReq_Fitness'].mean() - df_results['AlSafadi_Fitness'].mean())/df_results['AlSafadi_Fitness'].mean()*100):.2f}%"
        ]
    }

    df_macro = pd.DataFrame(macro_summary)
    print(df_macro.to_string(index=False))

    logger.info("\n--- Statistical Significance (AlphaReq vs Al-Safadi et al. 2023) ---")
    for k, v in stats_tests.items():
        logger.info(f"  {k:22s}: p = {v['p_value']:.5e}, Cohen's d = {v['Cohens_d']:+.2f} (Significant: {v['Significant']})")

    # 4. Generate Publication Figures
    if not args.skip_plots:
        logger.info("Generating publication-grade figures (DPI 300)...")
        generate_all_figures(df_results, output_dir=plots_dir)

    logger.info("\n[+] AlphaReq Master Benchmark Completed Successfully!")

if __name__ == '__main__':
    main()
