"""
AlphaReq: Automated Conflict Resolution and Autonomous Software Architecture Synthesis
from Natural Language Requirements Specifications.
"""

from .dataset_loader import PExpDataLoader
from .conflict_detector import ConflictDetector
from .energy_solver import HamiltonianEnergySolver
from .architecture_synthesizer import ArchitectureSynthesizer
from .baseline_comparisons import BasePaperModels
from .evaluation_benchmark import MasterEvaluationBenchmark

__version__ = "1.0.0"
__all__ = [
    "PExpDataLoader",
    "ConflictDetector",
    "HamiltonianEnergySolver",
    "ArchitectureSynthesizer",
    "BasePaperModels",
    "MasterEvaluationBenchmark"
]
