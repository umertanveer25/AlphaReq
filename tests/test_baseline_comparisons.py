import pytest
import pandas as pd
from src.baseline_comparisons import BasePaperModels
from src.evaluation_benchmark import MasterEvaluationBenchmark

def test_baselines(sample_requirements_df):
    models = BasePaperModels()
    res1 = models.evaluate_ezzini_baseline(sample_requirements_df, num_conflicts=2)
    assert res1['conflict_detection_recall'] == 0.0
    assert 0.0 <= res1['cohesion_modularity'] <= 1.0

    mock_conflicts = [
        {'req_i_text': 'fast response in 500 ms', 'req_j_text': 'secure encrypt login'}
    ]
    res2 = models.evaluate_alsafadi_baseline(sample_requirements_df, mock_conflicts)
    assert res2['conflict_detection_recall'] > 0.0
    assert 0.0 <= res2['coupling_density'] <= 1.0

def test_statistical_test_computation():
    # Synthetic dataframe matching benchmark structure (N=10 for Wilcoxon power)
    df_mock = pd.DataFrame({
        'AlphaReq_Cohesion': [0.75, 0.76, 0.74, 0.78, 0.72, 0.77, 0.73, 0.79, 0.75, 0.76],
        'AlSafadi_Cohesion': [0.60, 0.61, 0.59, 0.62, 0.58, 0.61, 0.59, 0.63, 0.60, 0.61],
        'AlphaReq_Coupling': [0.30, 0.32, 0.31, 0.29, 0.33, 0.30, 0.31, 0.28, 0.32, 0.30],
        'AlSafadi_Coupling': [0.65, 0.68, 0.67, 0.66, 0.69, 0.67, 0.68, 0.65, 0.66, 0.67],
        'AlphaReq_CSR': [0.95, 0.96, 0.97, 0.94, 0.98, 0.95, 0.96, 0.97, 0.94, 0.96],
        'AlSafadi_CSR': [0.68, 0.70, 0.69, 0.67, 0.71, 0.69, 0.68, 0.70, 0.67, 0.69],
        'AlphaReq_Fitness': [0.80, 0.81, 0.82, 0.79, 0.83, 0.81, 0.80, 0.82, 0.79, 0.81],
        'AlSafadi_Fitness': [0.62, 0.63, 0.64, 0.61, 0.65, 0.63, 0.62, 0.64, 0.61, 0.63],
    })
    stats = MasterEvaluationBenchmark.compute_statistical_tests(df_mock)
    assert 'Cohesion_vs_AlSafadi' in stats
    assert stats['Cohesion_vs_AlSafadi']['Significant'] is True
    assert stats['CSR_vs_AlSafadi']['Cohens_d'] > 0
    assert 'Vargha_Delaney_A12' in stats['CSR_vs_AlSafadi']

def test_vargha_delaney_a12():
    import numpy as np
    x = np.array([10, 11, 12, 13, 14])
    y = np.array([1, 2, 3, 4, 5])
    a12 = MasterEvaluationBenchmark.vargha_delaney_a12(x, y)
    assert a12 == 1.0  # Perfect superiority

    # Equal distributions
    a12_equal = MasterEvaluationBenchmark.vargha_delaney_a12(x, x)
    assert a12_equal == 0.5
