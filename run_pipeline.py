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
    default_data_path = os.path.join(BASE_DIR, "data", "Promise_Exp.csv")
    if not os.path.exists(default_data_path) and os.path.exists(r"B:\Promise_Exp.csv"):
        default_data_path = r"B:\Promise_Exp.csv"

    parser.add_argument(
        "--data-path",
        type=str,
        default=default_data_path,
        help="Path to the Promise_Exp dataset CSV file (defaults to data/Promise_Exp.csv)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.environ.get("TEMP", r"C:\Users\umert\AppData\Local\Temp\opencode"), "alphareq_output"),
        help="Directory to save benchmark summary CSVs, JSON stats, and plots."
    )
    parser.add_argument(
        "--target-modules",
        type=int,
        default=4,
        help="Number of target microservices/modules for architectural partitioning."
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default="agglomerative",
        choices=["agglomerative", "kmeans", "spectral"],
        help="Clustering backbone algorithm for functional requirement partitioning."
    )
    parser.add_argument(
        "--run-multi-solvers",
        action="store_true",
        help="Run multi-algorithm optimization benchmark (SA, GA, Tabu Search, Hill Climbing, Random Search)."
    )
    parser.add_argument(
        "--run-multi-clustering",
        action="store_true",
        help="Run multi-clustering benchmark (Agglomerative, K-Means, Spectral)."
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run rigorous component ablation study."
    )
    parser.add_argument(
        "--run-kfold",
        action="store_true",
        help="Run 10-Fold Cross-Validation across 30 randomized splits (N=300 evaluations)."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)."
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=30,
        help="Number of randomized repetitions per fold (default: 30 repeats x 10 folds = 300 evaluations)."
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run primary benchmark, multi-solver benchmark, multi-clustering benchmark, ablation study, and 10-fold 30-repeat CV."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generation of publication figures."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("=" * 80)
    logger.info("       ALPHAREQ MASTER PIPELINE: REPRODUCIBLE EXPERIMENTAL BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Target Corpus: {args.data_path}")
    logger.info(f"Output Directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    benchmark = MasterEvaluationBenchmark(data_path=args.data_path, target_modules=args.target_modules)

    # -------------------------------------------------------------
    # 1. Primary Master Evaluation Benchmark
    # -------------------------------------------------------------
    logger.info("\n>>> [STAGE 1/4] EXECUTING PRIMARY BENCHMARK (AlphaReq vs Ezzini 2022 & Al-Safadi 2023)...")
    df_results = benchmark.run_full_benchmark()

    csv_target = os.path.join(args.output_dir, "benchmark_summary.csv")
    safe_save_dataframe(df_results, csv_target)

    stats_tests = benchmark.compute_statistical_tests(df_results)
    json_target = os.path.join(args.output_dir, "statistical_tests.json")
    safe_save_json(stats_tests, json_target)

    # Export Docker Compose Blueprint for Project 8
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

    # Print Macro Comparison Table
    logger.info("\n" + "=" * 90)
    logger.info("                     PRIMARY PERFORMANCE COMPARISON TABLE")
    logger.info("=" * 90)

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

    logger.info("\n--- Inferential Statistical Tests (AlphaReq vs Al-Safadi 2023) ---")
    for k, v in stats_tests.items():
        logger.info(f"  {k:22s}: p = {v['p_value']:.5e}, Cohen's d = {v['Cohens_d']:+.2f}, A12 = {v['Vargha_Delaney_A12']:.3f} (Significant: {v['Significant']})")

    # -------------------------------------------------------------
    # 2. Multi-Solver Optimization Benchmark
    # -------------------------------------------------------------
    if args.run_multi_solvers or args.run_all:
        logger.info("\n>>> [STAGE 2/4] EXECUTING MULTI-SOLVER OPTIMIZATION BENCHMARK (SA, GA, Tabu, HC, RS)...")
        df_solvers, solver_stats = benchmark.run_multi_solver_benchmark()
        safe_save_dataframe(df_solvers, os.path.join(args.output_dir, "multi_solver_benchmark.csv"))
        safe_save_json(solver_stats, os.path.join(args.output_dir, "multi_solver_stats.json"))

        logger.info("\n" + "=" * 90)
        logger.info("               MULTI-SOLVER OPTIMIZATION LEADERBOARD (47 PROJECTS)")
        logger.info("=" * 90)

        solver_names = ['Simulated_Annealing', 'Genetic_Algorithm', 'Tabu_Search', 'Hill_Climbing', 'Random_Search']
        solver_summary = []
        for s in solver_names:
            solver_summary.append({
                'Optimization Algorithm': s.replace("_", " "),
                'Mean Energy H(sigma)': f"{df_solvers[f'{s}_Energy'].mean():.2f}",
                'Mean Runtime (ms)': f"{df_solvers[f'{s}_Time_ms'].mean():.2f} ms",
                'Mean CSR (%)': f"{df_solvers[f'{s}_CSR'].mean()*100:.2f}%",
                'Mean Fitness': f"{df_solvers[f'{s}_Fitness'].mean():.3f}"
            })
        print(pd.DataFrame(solver_summary).to_string(index=False))

        logger.info(f"\nFriedman Omnibus Test: Chi2 = {solver_stats['Friedman_ChiSquare']:.3f}, p = {solver_stats['Friedman_p_value']:.5e} (Significant: {solver_stats['Friedman_Significant']})")
        logger.info("Pairwise Tests (Simulated Annealing vs Competitors):")
        for k, v in solver_stats['Pairwise_Tests'].items():
            logger.info(f"  {k:26s}: p = {v['p_value']:.5e}, Cohen's d = {v['Cohens_d']:+.2f}, A12 = {v['Vargha_Delaney_A12']:.3f}")

    # -------------------------------------------------------------
    # 3. Multi-Clustering Architecture Benchmark
    # -------------------------------------------------------------
    if args.run_multi_clustering or args.run_all:
        logger.info("\n>>> [STAGE 3/4] EXECUTING MULTI-CLUSTERING BENCHMARK (Agglomerative, K-Means, Spectral)...")
        df_cluster, cluster_stats = benchmark.run_multi_clustering_benchmark()
        safe_save_dataframe(df_cluster, os.path.join(args.output_dir, "multi_clustering_benchmark.csv"))
        safe_save_json(cluster_stats, os.path.join(args.output_dir, "multi_clustering_stats.json"))

        logger.info("\n" + "=" * 90)
        logger.info("               MULTI-CLUSTERING ARCHITECTURE LEADERBOARD (47 PROJECTS)")
        logger.info("=" * 90)

        cluster_summary = []
        for c in ['agglomerative', 'kmeans', 'spectral']:
            cluster_summary.append({
                'Clustering Backbone': c.capitalize(),
                'Cohesion (Q)': f"{df_cluster[f'{c}_Cohesion'].mean():.3f}",
                'Coupling Density (C)': f"{df_cluster[f'{c}_Coupling'].mean():.3f}",
                'Architecture Fitness': f"{df_cluster[f'{c}_Fitness'].mean():.3f}"
            })
        print(pd.DataFrame(cluster_summary).to_string(index=False))
        logger.info(f"Friedman Test on Cohesion: Chi2 = {cluster_stats['Friedman_Cohesion_ChiSquare']:.3f}, p = {cluster_stats['Friedman_Cohesion_p_value']:.5e}")

    # -------------------------------------------------------------
    # 4. Rigorous Component Ablation Study
    # -------------------------------------------------------------
    if args.run_ablation or args.run_all:
        logger.info("\n>>> [STAGE 4/4] EXECUTING COMPONENT ABLATION STUDY...")
        df_ablation, ablation_stats = benchmark.run_ablation_study()
        safe_save_dataframe(df_ablation, os.path.join(args.output_dir, "ablation_study.csv"))
        safe_save_json(ablation_stats, os.path.join(args.output_dir, "ablation_stats.json"))

        logger.info("\n" + "=" * 90)
        logger.info("                 COMPONENT ABLATION STUDY RESULTS (47 PROJECTS)")
        logger.info("=" * 90)

        variants = [
            ('Full AlphaReq (Complete Model)', 'Full_AlphaReq'),
            ('w/o Infrastructure Enclaves (Bundled NFRs)', 'WO_Enclaves'),
            ('w/o Tension Coupling Matrix (J = 0)', 'WO_Coupling_J'),
            ('w/o Normative Modal Weights (h = 1)', 'WO_Modals_h'),
            ('w/o Optimization Solver (sigma = 1 Fixed)', 'WO_Optimization')
        ]

        ablation_summary = []
        for label, var in variants:
            ablation_summary.append({
                'Ablated Model Configuration': label,
                'Mean CSR (%)': f"{df_ablation[f'{var}_CSR'].mean()*100:.2f}%",
                'Mean Fitness': f"{df_ablation[f'{var}_Fitness'].mean():.3f}",
                'Delta CSR vs Full': f"{(df_ablation[f'{var}_CSR'].mean() - df_ablation['Full_AlphaReq_CSR'].mean())*100:+.2f}%" if var != 'Full_AlphaReq' else "Ref"
            })
        print(pd.DataFrame(ablation_summary).to_string(index=False))

        logger.info("\nAblation Statistical Significance (Full AlphaReq vs Ablated Variants):")
        for k, v in ablation_stats.items():
            logger.info(f"  {k:24s}: p = {v['p_value']:.5e}, Cohen's d = {v['Cohens_d']:+.2f}, A12 = {v['Vargha_Delaney_A12']:.3f} (Significant: {v['Significant']})")

    # -------------------------------------------------------------
    # 5. 10-Fold Cross-Validation with 30 Randomized Splits
    # -------------------------------------------------------------
    df_cv = None
    if args.run_kfold or args.run_all:
        logger.info(f"\n>>> [STAGE 5/5] EXECUTING {args.n_splits}-FOLD CROSS-VALIDATION ACROSS {args.n_repeats} RANDOMIZED SPLITS (TOTAL {args.n_splits * args.n_repeats} EVALUATIONS)...")
        df_cv, cv_stats = benchmark.run_kfold_cross_validation(n_splits=args.n_splits, n_repeats=args.n_repeats)
        safe_save_dataframe(df_cv, os.path.join(args.output_dir, "kfold_30rep_benchmark.csv"))
        safe_save_json(cv_stats, os.path.join(args.output_dir, "kfold_30rep_stats.json"))

        logger.info("\n" + "=" * 100)
        logger.info(f"     10-FOLD CROSS-VALIDATION WITH 30 RANDOMIZED SPLITS (N={args.n_splits * args.n_repeats} EVALUATION FOLDS)")
        logger.info("=" * 100)

        cv_table_rows = []
        for m, m_data in cv_stats['metrics_summary'].items():
            is_pct = m in ['Conflict_Recall', 'CSR']
            fmt = (lambda v: f"{v*100:.2f}%") if is_pct else (lambda v: f"{v:.3f}")

            a_mean_std = f"{fmt(m_data['AlphaReq_Mean'])} +/- {fmt(m_data['AlphaReq_Std'])}"
            a_ci = f"[{fmt(m_data['AlphaReq_95CI'][0])}, {fmt(m_data['AlphaReq_95CI'][1])}]"
            als_mean_std = f"{fmt(m_data['AlSafadi_Mean'])} +/- {fmt(m_data['AlSafadi_Std'])}"
            ezz_mean_std = f"{fmt(m_data['Ezzini_Mean'])} +/- {fmt(m_data['Ezzini_Std'])}"

            cv_table_rows.append({
                'Evaluation Metric': m.replace("_", " "),
                'AlphaReq (Mean +/- Std)': a_mean_std,
                'AlphaReq 95% CI': a_ci,
                'Al-Safadi (Mean +/- Std)': als_mean_std,
                'Ezzini (Mean +/- Std)': ezz_mean_std,
                'vs Al-Safadi p-val': f"{m_data['vs_AlSafadi']['p_value']:.2e}",
                'Cohen d': f"{m_data['vs_AlSafadi']['Cohens_d']:+.2f}",
                'A12': f"{m_data['vs_AlSafadi']['Vargha_Delaney_A12']:.3f}"
            })
        print(pd.DataFrame(cv_table_rows).to_string(index=False))

    # -------------------------------------------------------------
    # 6. Generate Publication Figures (Up to 7 Figures at 300 DPI)
    # -------------------------------------------------------------
    if not args.skip_plots:
        logger.info("\nGenerating publication-grade figures (DPI 300)...")
        generate_all_figures(df_results, output_dir=plots_dir, df_cv=df_cv)

    logger.info("\n[+] AlphaReq Master Benchmark Execution Finished Successfully!")

if __name__ == '__main__':
    main()
