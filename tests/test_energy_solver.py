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

def test_genetic_algorithm_solver():
    from src.energy_solver import GeneticAlgorithmSolver
    ga = GeneticAlgorithmSolver(population_size=10, generations=10)
    J = np.array([[0.0, -0.8], [-0.8, 0.0]])
    h = np.array([1.0, 1.0])
    sigma, energy, hist = ga.solve_ground_state(J, h, random_seed=42)
    assert len(sigma) == 2
    assert set(np.unique(sigma)).issubset({-1.0, 1.0})
    assert len(hist) > 0

def test_tabu_search_solver():
    from src.energy_solver import TabuSearchSolver
    ts = TabuSearchSolver(tabu_tenure=2, max_steps=15)
    J = np.array([[0.0, -0.8], [-0.8, 0.0]])
    h = np.array([1.0, 1.0])
    sigma, energy, hist = ts.solve_ground_state(J, h, random_seed=42)
    assert len(sigma) == 2
    assert set(np.unique(sigma)).issubset({-1.0, 1.0})

def test_greedy_hill_climbing_solver():
    from src.energy_solver import GreedyHillClimbingSolver
    hc = GreedyHillClimbingSolver(max_steps=15)
    J = np.array([[0.0, -0.8], [-0.8, 0.0]])
    h = np.array([1.0, 1.0])
    sigma, energy, hist = hc.solve_ground_state(J, h, random_seed=42)
    assert len(sigma) == 2
    assert set(np.unique(sigma)).issubset({-1.0, 1.0})

def test_random_search_solver():
    from src.energy_solver import RandomSearchSolver
    rs = RandomSearchSolver(samples=20)
    J = np.array([[0.0, -0.8], [-0.8, 0.0]])
    h = np.array([1.0, 1.0])
    sigma, energy, hist = rs.solve_ground_state(J, h, random_seed=42)
    assert len(sigma) == 2
    assert set(np.unique(sigma)).issubset({-1.0, 1.0})

def test_get_optimization_solver_factory():
    from src.energy_solver import (
        get_optimization_solver,
        SimulatedAnnealingSolver,
        GeneticAlgorithmSolver,
        TabuSearchSolver,
        GreedyHillClimbingSolver,
        RandomSearchSolver
    )
    assert isinstance(get_optimization_solver('sa'), SimulatedAnnealingSolver)
    assert isinstance(get_optimization_solver('ga'), GeneticAlgorithmSolver)
    assert isinstance(get_optimization_solver('tabu'), TabuSearchSolver)
    assert isinstance(get_optimization_solver('hc'), GreedyHillClimbingSolver)
    assert isinstance(get_optimization_solver('rs'), RandomSearchSolver)
