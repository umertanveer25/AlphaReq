import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_requirements_df():
    """
    Creates a representative mini-requirements dataset for testing.
    """
    data = [
        {"S.No": 1, "File": 101, "Requirement": "The system shall allow users to search products.", "Type": "F"},
        {"S.No": 2, "File": 101, "Requirement": "The system shall display search results in a grid.", "Type": "F"},
        {"S.No": 3, "File": 101, "Requirement": "The system shall respond fast within 500 milliseconds.", "Type": "PE"},
        {"S.No": 4, "File": 101, "Requirement": "The product must require biometric two-factor authentication.", "Type": "SE"},
        {"S.No": 5, "File": 101, "Requirement": "The interface shall be intuitive and easy to use.", "Type": "US"},
        {"S.No": 6, "File": 101, "Requirement": "The product should support 5000 simultaneous users.", "Type": "SC"},
        {"S.No": 7, "File": 101, "Requirement": "The system shall operate 24/7 with 99.9% uptime.", "Type": "A"},
    ]
    df = pd.DataFrame(data)
    df['Clean_Requirement'] = df['Requirement']
    df['Normative_Weight'] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.0]
    return df

@pytest.fixture
def single_requirement_df():
    data = [{"S.No": 1, "File": 102, "Requirement": "The system shall boot up.", "Type": "F"}]
    df = pd.DataFrame(data)
    df['Clean_Requirement'] = df['Requirement']
    df['Normative_Weight'] = [1.0]
    return df

@pytest.fixture
def empty_df():
    df = pd.DataFrame(columns=["S.No", "File", "Requirement", "Type", "Clean_Requirement", "Normative_Weight"])
    return df
