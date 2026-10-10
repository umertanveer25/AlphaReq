import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .utils import setup_logger

logger = setup_logger("AlphaReq.EnergySolver")

class HamiltonianEnergySolver:
    """
    Simulated Annealing & Energy Minimization Solver for the Requirements Hamiltonian.
    Finds the ground state or minimal tension architectural state sigma* in {-1, +1}^N.
    """

    def __init__(self, initial_temp: float = 10.0, cooling_rate: float = 0.95, max_steps: int = 1000):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps

    @staticmethod
    def compute_energy(sigma: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
        """
        Computes Hamiltonian H(sigma) = - 0.5 * sigma^T * J * sigma - h^T * sigma
        """
        if len(sigma) == 0:
            return 0.0
        return -0.5 * float(np.dot(sigma, np.dot(J, sigma))) - float(np.dot(h, sigma))

    def solve_ground_state(self, J: np.ndarray, h: np.ndarray, random_seed: int = 42) -> Tuple[np.ndarray, float, List[float]]:
        """
        Simulated Annealing to find optimal spin vector sigma in {-1, +1}^N.
        """
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

    @staticmethod
    def analyze_resolution_strategy(project_df: pd.DataFrame, best_sigma: np.ndarray) -> List[Dict]:
        """
        Maps relaxed spins (sigma_i = -1) into concrete architectural patterns.
        """
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
