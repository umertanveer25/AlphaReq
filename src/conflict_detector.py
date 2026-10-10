import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from .utils import setup_logger

logger = setup_logger("AlphaReq.ConflictDetector")

class ConflictDetector:
    """
    Requirements Hamiltonian Interaction & Frustration Detector.
    Models inter-requirement trade-offs as energetic couplings in a disordered spin-glass manifold.
    Safely handles boundary conditions (N < 2, isolated single requirements, empty sets).
    """

    # Domain-specific architectural trade-off tension matrix (antiferromagnetic couplings J < 0)
    TRADE_OFF_PENALTIES = {
        ('PE', 'SE'): -0.75,  # Real-time latency vs Deep cryptographic handshakes / auth checks
        ('SE', 'US'): -0.65,  # Multi-factor authentication / strict timeouts vs frictionless UX
        ('PE', 'FT'): -0.45,  # Low latency vs Synchronous replication & consensus checkpoints
        ('SC', 'PE'): -0.40,  # Massive horizontal distribution vs Single-node raw compute speed
        ('LF', 'PE'): -0.25,  # Heavy rich multimedia UI rendering vs Strict response time envelopes
        ('MN', 'PE'): -0.30,  # Layered abstraction & indirection vs Zero-copy raw performance
        ('SE', 'O'):  -0.35,  # Zero-trust perimeter vs Legacy operational host environment
        ('FT', 'US'): -0.20,  # Crash recovery dialogues vs Seamless user experience
    }

    # Synergistic cooperative couplings (ferromagnetic couplings J > 0)
    COOPERATION_BONUSES = {
        ('LF', 'US'): 0.55,   # Intuitive look and feel directly reinforces usability
        ('F', 'O'):   0.45,   # Core features aligned with operational target
        ('A', 'FT'):  0.60,   # Fault tolerance directly guarantees high availability
        ('SC', 'A'):  0.50,   # Scalable elastic clusters improve availability
        ('SE', 'L'):  0.55,   # Security directly satisfies legal/compliance mandates
    }

    def __init__(self, critical_threshold: float = 1.80, custom_penalties: Optional[Dict] = None, custom_bonuses: Optional[Dict] = None):
        self.critical_threshold = critical_threshold
        self.penalties = dict(self.TRADE_OFF_PENALTIES)
        if custom_penalties:
            self.penalties.update(custom_penalties)
        self.bonuses = dict(self.COOPERATION_BONUSES)
        if custom_bonuses:
            self.bonuses.update(custom_bonuses)

    def compute_coupling_tensor(self, project_df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
        """
        Computes the N x N coupling matrix J for all requirements in a specific project.
        Returns (J_matrix, list_of_conflicts).
        """
        n = len(project_df)
        if n < 2:
            return np.zeros((n, n), dtype=np.float64), []

        J = np.zeros((n, n), dtype=np.float64)
        conflicts = []

        types = project_df['Type'].tolist()
        texts = project_df['Clean_Requirement'].tolist()

        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = types[i], types[j]
                pair = tuple(sorted([t1, t2]))

                coupling_val = 0.0

                if pair in self.penalties:
                    coupling_val = self.penalties[pair]
                elif (pair[1], pair[0]) in self.penalties:
                    coupling_val = self.penalties[(pair[1], pair[0])]
                elif pair in self.bonuses:
                    coupling_val = self.bonuses[pair]
                elif (pair[1], pair[0]) in self.bonuses:
                    coupling_val = self.bonuses[(pair[1], pair[0])]
                else:
                    coupling_val = 0.10 if (t1 == 'F' or t2 == 'F') else 0.05

                # Textual modulation
                intensity = self._extract_conflict_intensity(texts[i], texts[j])
                coupling_val *= intensity

                J[i, j] = coupling_val
                J[j, i] = coupling_val

                if coupling_val < -0.2:
                    conflicts.append({
                        'req_i_idx': i,
                        'req_j_idx': j,
                        'req_i_type': t1,
                        'req_j_type': t2,
                        'req_i_text': texts[i],
                        'req_j_text': texts[j],
                        'coupling': float(coupling_val),
                        'severity': 'Critical' if coupling_val < -0.5 else 'Moderate'
                    })

        return J, conflicts

    @staticmethod
    def _extract_conflict_intensity(text1: str, text2: str) -> float:
        multiplier = 1.0
        t_combined = (str(text1) + " " + str(text2)).lower()
        if re.search(r'\b(millisecond|seconds|fast|immediate|real-time)\b', t_combined):
            multiplier += 0.25
        if re.search(r'\b(encrypt|authorized|password|secure|biometric|token)\b', t_combined):
            multiplier += 0.25
        if re.search(r'\b(simultaneous|concurrent|1000|10000|multiple users)\b', t_combined):
            multiplier += 0.20
        return multiplier

    def evaluate_project_frustration(self, project_df: pd.DataFrame) -> Dict:
        """
        Computes statistical mechanical metrics for a project:
        Hamiltonian tension, Frustration Index, Constraint Density (alpha), and Phase State.
        """
        n = len(project_df)
        if n == 0:
            return {
                'n_requirements': 0,
                'num_conflicts': 0,
                'total_possible_pairs': 0,
                'frustration_index': 0.0,
                'constraint_density_alpha': 0.0,
                'energy_tension': 0.0,
                'phase_state': 'Paramagnetic (Low Risk)',
                'critical_boundary_alpha_c': self.critical_threshold,
                'critical_conflicts_sample': []
            }

        if n == 1:
            h_val = float(project_df['Normative_Weight'].iloc[0]) if 'Normative_Weight' in project_df.columns else 0.5
            return {
                'n_requirements': 1,
                'num_conflicts': 0,
                'total_possible_pairs': 0,
                'frustration_index': 0.0,
                'constraint_density_alpha': 0.0,
                'energy_tension': -h_val,
                'phase_state': 'Paramagnetic (Low Risk)',
                'critical_boundary_alpha_c': self.critical_threshold,
                'critical_conflicts_sample': []
            }

        J, conflicts = self.compute_coupling_tensor(project_df)
        h = project_df['Normative_Weight'].values if 'Normative_Weight' in project_df.columns else np.full(n, 0.5)

        sigma_all_enforced = np.ones(n)
        energy_tension = -0.5 * np.dot(sigma_all_enforced, np.dot(J, sigma_all_enforced)) - np.dot(h, sigma_all_enforced)

        total_pairs = (n * (n - 1)) / 2
        num_conflicts = len(conflicts)
        frustration_index = num_conflicts / total_pairs if total_pairs > 0 else 0.0

        # Constraint density alpha = M_frustrated / N
        alpha = num_conflicts / n if n > 0 else 0.0

        if alpha > self.critical_threshold:
            phase = "Glassy/Frustrated (High Risk)"
        elif alpha >= 0.8:
            phase = "Marginal (Moderate Risk)"
        else:
            phase = "Paramagnetic (Low Risk)"

        return {
            'n_requirements': n,
            'num_conflicts': num_conflicts,
            'total_possible_pairs': int(total_pairs),
            'frustration_index': float(frustration_index),
            'constraint_density_alpha': float(alpha),
            'energy_tension': float(energy_tension),
            'phase_state': phase,
            'critical_boundary_alpha_c': self.critical_threshold,
            'critical_conflicts_sample': conflicts[:5]
        }
