import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
from .utils import setup_logger

logger = setup_logger("AlphaReq.EnergySolver")

class BaseOptimizationSolver:
    """
    Abstract Base Class for Hamiltonian Discrete Energy Minimization Solvers.
    Minimizes: H(sigma) = - 0.5 * sigma^T * J * sigma - h^T * sigma
    where sigma in {-1, +1}^N.
    """

    @staticmethod
    def compute_energy(sigma: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
        if len(sigma) == 0:
            return 0.0
        return -0.5 * float(np.dot(sigma, np.dot(J, sigma))) - float(np.dot(h, sigma))

    @staticmethod
    def analyze_resolution_strategy(project_df: pd.DataFrame, best_sigma: np.ndarray) -> List[Dict]:
        mitigations = []
        if len(best_sigma) == 0:
            return mitigations

        for idx, (spin, req_type, req_text) in enumerate(zip(best_sigma, project_df['Type'], project_df['Clean_Requirement'])):
            if spin == -1.0:
                strategy = "Asynchronous Decoupling / Event Bus"
                if req_type == 'PE':
                    strategy = "Distributed Redis/Memcached In-Memory Cache"
                elif req_type == 'SE':
                    strategy = "Tokenized API Gateway & OAuth2 Stateless Edge Verification"
                elif req_type == 'US':
                    strategy = "Optimistic UI Update with Background Reconciliation"
                elif req_type == 'FT':
                    strategy = "Circuit Breaker Pattern with Graceful Fallback"

                mitigations.append({
                    'req_index': idx,
                    'type': req_type,
                    'text': req_text,
                    'architectural_mitigation': strategy
                })
        return mitigations


class HamiltonianEnergySolver(BaseOptimizationSolver):
    """
    Simulated Annealing & Energy Minimization Solver (Default Metropolis Monte Carlo).
    Maintains backward compatibility with all baseline and evaluation pipelines.
    """

    def __init__(self, initial_temp: float = 10.0, cooling_rate: float = 0.95, max_steps: int = 1000):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps
        self.name = "Simulated Annealing (SA)"

    def solve_ground_state(self, J: np.ndarray, h: np.ndarray, random_seed: int = 42) -> Tuple[np.ndarray, float, List[float]]:
        n = len(h)
        if n == 0:
            return np.array([]), 0.0, [0.0]
        if n == 1:
            best_sigma = np.array([1.0])
            best_energy = self.compute_energy(best_sigma, J, h)
            return best_sigma, best_energy, [best_energy]

        rng = np.random.RandomState(random_seed)
        current_sigma = np.ones(n, dtype=np.float64)
        current_energy = self.compute_energy(current_sigma, J, h)

        best_sigma = current_sigma.copy()
        best_energy = current_energy

        temp = self.initial_temp
        energy_history = [current_energy]

        for step in range(self.max_steps):
            flip_idx = rng.randint(0, n)
            candidate_sigma = current_sigma.copy()
            candidate_sigma[flip_idx] *= -1.0

            candidate_energy = self.compute_energy(candidate_sigma, J, h)
            delta_e = candidate_energy - current_energy

            if delta_e < 0 or rng.rand() < np.exp(-delta_e / max(temp, 1e-6)):
                current_sigma = candidate_sigma
                current_energy = candidate_energy

                if current_energy < best_energy:
                    best_sigma = current_sigma.copy()
                    best_energy = current_energy

            temp *= self.cooling_rate
            energy_history.append(current_energy)

        return best_sigma, best_energy, energy_history


class SimulatedAnnealingSolver(HamiltonianEnergySolver):
    """Explicit alias for Simulated Annealing Solver."""
    pass


class GeneticAlgorithmSolver(BaseOptimizationSolver):
    """
    Genetic Algorithm (GA) Metaheuristic Solver.
    Uses population-based evolutionary search with uniform crossover, bit-flip mutation, and elitism.
    """

    def __init__(self, population_size: int = 40, generations: int = 60, mutation_rate: float = 0.05, elitism: int = 2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.name = "Genetic Algorithm (GA)"

    def solve_ground_state(self, J: np.ndarray, h: np.ndarray, random_seed: int = 42) -> Tuple[np.ndarray, float, List[float]]:
        n = len(h)
        if n == 0:
            return np.array([]), 0.0, [0.0]
        if n == 1:
            best_sigma = np.array([1.0])
            return best_sigma, self.compute_energy(best_sigma, J, h), [0.0]

        rng = np.random.RandomState(random_seed)
        # Initialize population: one all-ones (normative baseline) + random +/- 1
        population = [np.ones(n, dtype=np.float64)]
        for _ in range(self.population_size - 1):
            ind = rng.choice([-1.0, 1.0], size=n)
            population.append(ind)

        best_sigma = population[0].copy()
        best_energy = self.compute_energy(best_sigma, J, h)
        energy_history = [best_energy]

        for gen in range(self.generations):
            # Evaluate fitness (lower energy = higher fitness)
            energies = [self.compute_energy(ind, J, h) for ind in population]
            # Track best
            min_idx = int(np.argmin(energies))
            if energies[min_idx] < best_energy:
                best_energy = energies[min_idx]
                best_sigma = population[min_idx].copy()
            energy_history.append(best_energy)

            # Sort population by energy
            sorted_indices = np.argsort(energies)
            new_population = [population[i].copy() for i in sorted_indices[:self.elitism]]

            # Tournament selection & crossover
            while len(new_population) < self.population_size:
                # Tournament size 3
                t1 = rng.choice(self.population_size, size=3)
                p1_idx = t1[np.argmin([energies[i] for i in t1])]
                t2 = rng.choice(self.population_size, size=3)
                p2_idx = t2[np.argmin([energies[i] for i in t2])]

                parent1 = population[p1_idx]
                parent2 = population[p2_idx]

                # Uniform crossover
                mask = rng.rand(n) < 0.5
                child = np.where(mask, parent1, parent2)

                # Mutation
                mutate_mask = rng.rand(n) < self.mutation_rate
                child[mutate_mask] *= -1.0

                new_population.append(child)

            population = new_population

        return best_sigma, best_energy, energy_history


class TabuSearchSolver(BaseOptimizationSolver):
    """
    Tabu Search (TS) Metaheuristic Solver.
    Avoids cyclic thrashing by maintaining a short-term memory Tabu List of recently inverted indices.
    """

    def __init__(self, tabu_tenure: int = 7, max_steps: int = 300):
        self.tabu_tenure = tabu_tenure
        self.max_steps = max_steps
        self.name = "Tabu Search (TS)"

    def solve_ground_state(self, J: np.ndarray, h: np.ndarray, random_seed: int = 42) -> Tuple[np.ndarray, float, List[float]]:
        n = len(h)
        if n == 0:
            return np.array([]), 0.0, [0.0]
        if n == 1:
            best_sigma = np.array([1.0])
            return best_sigma, self.compute_energy(best_sigma, J, h), [0.0]

        current_sigma = np.ones(n, dtype=np.float64)
        current_energy = self.compute_energy(current_sigma, J, h)

        best_sigma = current_sigma.copy()
        best_energy = current_energy

        # Tabu structure: step when spin i becomes non-tabu
        tabu_until = np.zeros(n, dtype=int)
        energy_history = [best_energy]

        for step in range(self.max_steps):
            best_neighbor_idx = -1
            best_neighbor_energy = float('inf')

            # Scan 1-flip neighborhood
            for i in range(n):
                cand = current_sigma.copy()
                cand[i] *= -1.0
                cand_e = self.compute_energy(cand, J, h)

                # Aspiration criterion: allow tabu if strictly better than global best
                is_tabu = step < tabu_until[i]
                if not is_tabu or cand_e < best_energy:
                    if cand_e < best_neighbor_energy:
                        best_neighbor_energy = cand_e
                        best_neighbor_idx = i

            if best_neighbor_idx == -1:
                break

            current_sigma[best_neighbor_idx] *= -1.0
            current_energy = best_neighbor_energy
            tabu_until[best_neighbor_idx] = step + self.tabu_tenure

            if current_energy < best_energy:
                best_energy = current_energy
                best_sigma = current_sigma.copy()

            energy_history.append(best_energy)

        return best_sigma, best_energy, energy_history


class GreedyHillClimbingSolver(BaseOptimizationSolver):
    """
    Greedy Steepest Descent / Hill Climbing Solver.
    Deterministic baseline: iteratively flips the variable with the steepest immediate energy reduction.
    """

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self.name = "Greedy Hill Climbing (HC)"

    def solve_ground_state(self, J: np.ndarray, h: np.ndarray, random_seed: int = 42) -> Tuple[np.ndarray, float, List[float]]:
        n = len(h)
        if n == 0:
            return np.array([]), 0.0, [0.0]
        if n == 1:
            best_sigma = np.array([1.0])
            return best_sigma, self.compute_energy(best_sigma, J, h), [0.0]

        current_sigma = np.ones(n, dtype=np.float64)
        current_energy = self.compute_energy(current_sigma, J, h)
        energy_history = [current_energy]

        for _ in range(self.max_steps):
            best_flip = -1
            best_drop = 0.0

            for i in range(n):
                cand = current_sigma.copy()
                cand[i] *= -1.0
                cand_e = self.compute_energy(cand, J, h)
                drop = current_energy - cand_e
                if drop > best_drop:
                    best_drop = drop
                    best_flip = i

            # If no neighbor strictly improves energy, reached local minimum
            if best_flip == -1 or best_drop <= 1e-7:
                break

            current_sigma[best_flip] *= -1.0
            current_energy -= best_drop
            energy_history.append(current_energy)

        return current_sigma, current_energy, energy_history


class RandomSearchSolver(BaseOptimizationSolver):
    """
    Random Uniform Search Baseline.
    Samples random binary spin vectors and keeps the lowest energy configuration found.
    """

    def __init__(self, samples: int = 500):
        self.samples = samples
        self.name = "Random Search (RS)"

    def solve_ground_state(self, J: np.ndarray, h: np.ndarray, random_seed: int = 42) -> Tuple[np.ndarray, float, List[float]]:
        n = len(h)
        if n == 0:
            return np.array([]), 0.0, [0.0]
        if n == 1:
            best_sigma = np.array([1.0])
            return best_sigma, self.compute_energy(best_sigma, J, h), [0.0]

        rng = np.random.RandomState(random_seed)
        best_sigma = np.ones(n, dtype=np.float64)
        best_energy = self.compute_energy(best_sigma, J, h)
        energy_history = [best_energy]

        for _ in range(self.samples):
            cand = rng.choice([-1.0, 1.0], size=n)
            cand_e = self.compute_energy(cand, J, h)
            if cand_e < best_energy:
                best_energy = cand_e
                best_sigma = cand.copy()
            energy_history.append(best_energy)

        return best_sigma, best_energy, energy_history


def get_optimization_solver(name: str) -> BaseOptimizationSolver:
    """Factory function to instantiate solver by name."""
    name_clean = name.lower().replace("-", "_").replace(" ", "_")
    if name_clean in ['sa', 'simulated_annealing', 'default']:
        return SimulatedAnnealingSolver()
    elif name_clean in ['ga', 'genetic_algorithm', 'genetic']:
        return GeneticAlgorithmSolver()
    elif name_clean in ['tabu', 'tabu_search', 'ts']:
        return TabuSearchSolver()
    elif name_clean in ['greedy', 'hill_climbing', 'hc']:
        return GreedyHillClimbingSolver()
    elif name_clean in ['random', 'random_search', 'rs']:
        return RandomSearchSolver()
    else:
        logger.warning(f"Unknown solver '{name}', defaulting to Simulated Annealing.")
        return SimulatedAnnealingSolver()
