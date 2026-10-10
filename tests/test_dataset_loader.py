import pytest
import os
import pandas as pd
from src.dataset_loader import PExpDataLoader

def test_normative_weight_computation():
    assert PExpDataLoader._compute_normative_weight("The system shall authenticate the user.") == 1.0
    assert PExpDataLoader._compute_normative_weight("The system must maintain audit logs.") == 1.0
    assert PExpDataLoader._compute_normative_weight("The system should cache frequent queries.") == 0.6
    assert PExpDataLoader._compute_normative_weight("The system will notify the administrator.") == 0.6
    assert PExpDataLoader._compute_normative_weight("The user may customize color themes.") == 0.3
    assert PExpDataLoader._compute_normative_weight("The operator can override defaults.") == 0.3
    assert PExpDataLoader._compute_normative_weight("General descriptive comment.") == 0.5
    assert PExpDataLoader._compute_normative_weight("") == 0.5

def test_clean_text():
    raw = "The system \n\t shall   refresh   display.  "
    cleaned = PExpDataLoader._clean_text(raw)
    assert cleaned == "The system shall refresh display."

def test_loader_with_real_dataset():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "data", "Promise_Exp.csv")
    if not os.path.exists(data_path) and os.path.exists(r"B:\Promise_Exp.csv"):
        data_path = r"B:\Promise_Exp.csv"
    if os.path.exists(data_path):
        loader = PExpDataLoader(data_path)
        summary = loader.get_dataset_summary()
        assert summary['total_requirements'] == 969
        assert summary['total_projects'] == 47
        assert 'F' in summary['class_distribution']
        assert 'SE' in summary['class_distribution']
        assert summary['normative_obligation_count'] > 700

        project_ids = loader.get_project_ids()
        assert len(project_ids) == 47
        p8 = loader.get_project(8)
        assert len(p8) == 92
