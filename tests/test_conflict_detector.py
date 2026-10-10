import pytest
import numpy as np
import pandas as pd
from src.conflict_detector import ConflictDetector

def test_boundary_empty_df(empty_df):
    detector = ConflictDetector()
    res = detector.evaluate_project_frustration(empty_df)
    assert res['n_requirements'] == 0
    assert res['num_conflicts'] == 0
    assert res['frustration_index'] == 0.0
    assert res['constraint_density_alpha'] == 0.0
    assert res['phase_state'] == "Paramagnetic (Low Risk)"

def test_boundary_single_df(single_requirement_df):
    detector = ConflictDetector()
    res = detector.evaluate_project_frustration(single_requirement_df)
    assert res['n_requirements'] == 1
    assert res['num_conflicts'] == 0
    assert res['total_possible_pairs'] == 0
    assert res['frustration_index'] == 0.0
    assert res['phase_state'] == "Paramagnetic (Low Risk)"

def test_conflict_detection_pairs(sample_requirements_df):
    detector = ConflictDetector()
    J, conflicts = detector.compute_coupling_tensor(sample_requirements_df)
    assert J.shape == (7, 7)
    # Check that J is symmetric
    assert np.allclose(J, J.T)
    # Check that SE vs PE is detected as conflicting
    # In sample_requirements_df, req 2 is PE (idx 2) and req 3 is SE (idx 3)
    assert J[2, 3] < -0.5
    assert len(conflicts) > 0
    severities = [c['severity'] for c in conflicts]
    assert 'Critical' in severities

def test_phase_state_classification():
    detector = ConflictDetector(critical_threshold=1.80)
    # Test high conflict df mock
    high_conflict_df = pd.DataFrame([
        {"Requirement": f"Req {i}", "Type": "SE" if i % 2 == 0 else "PE", "Clean_Requirement": f"Req {i}", "Normative_Weight": 1.0}
        for i in range(20)
    ])
    res = detector.evaluate_project_frustration(high_conflict_df)
    assert res['constraint_density_alpha'] > 1.80
    assert res['phase_state'] == "Glassy/Frustrated (High Risk)"

def test_custom_penalties_override(sample_requirements_df):
    custom = {('PE', 'SE'): -0.99}
    detector = ConflictDetector(custom_penalties=custom)
    J, conflicts = detector.compute_coupling_tensor(sample_requirements_df)
    # Check that custom penalty took effect
    assert J[2, 3] < -0.90
