import pytest
import numpy as np
import pandas as pd
from src.energy_solver import HamiltonianEnergySolver

def test_compute_energy_exact():
    solver = HamiltonianEnergySolver()
    # 2-spin test
    sigma = np.array([1.0, 1.0])
    J = np.array([[0.0, -0.5], [-0.5, 0.0]])
    h = np.array([1.0, 1.0])
    # H = -0.5 * (1*-0.5*1 + 1*-0.5*1) - (1*1 + 1*1) = -0.5 * (-1.0) - 2.0 = 0.5 - 2.0 = -1.5
    energy = solver.compute_energy(sigma, J, h)
    assert np.isclose(energy, -1.5)

def test_simulated_annealing_ground_state():
    solver = HamiltonianEnergySolver(max_steps=200)
    # Frustrated antiferromagnetic triangle J_ij = -1
    J = np.array([
        [0.0, -1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [-1.0, -1.0, 0.0]
    ])
    h = np.zeros(3)
    best_sigma, best_energy, history = solver.solve_ground_state(J, h, random_seed=42)
    assert len(best_sigma) == 3
    assert set(np.unique(best_sigma)).issubset({-1.0, 1.0})
    assert len(history) > 1

def test_deterministic_seed():
    solver = HamiltonianEnergySolver(max_steps=100)
    J = np.random.randn(5, 5)
    J = 0.5 * (J + J.T)
    h = np.ones(5)
    s1, e1, _ = solver.solve_ground_state(J, h, random_seed=123)
    s2, e2, _ = solver.solve_ground_state(J, h, random_seed=123)
    assert np.allclose(s1, s2)
    assert np.isclose(e1, e2)

def test_analyze_resolution_strategy():
    solver = HamiltonianEnergySolver()
    df = pd.DataFrame([
        {'Type': 'PE', 'Clean_Requirement': 'Fast latency under 10ms'},
        {'Type': 'SE', 'Clean_Requirement': 'Enforce biometric OAuth2 token'}
    ])
    sigma = np.array([-1.0, -1.0])
    mitigations = solver.analyze_resolution_strategy(df, sigma)
    assert len(mitigations) == 2
    assert "Redis" in mitigations[0]['architectural_mitigation']
    assert "Gateway" in mitigations[1]['architectural_mitigation']
