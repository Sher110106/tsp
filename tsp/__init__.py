"""
World-Class TSP Solver
Implements state-of-the-art algorithms with Numba acceleration and parallel processing.
"""

__version__ = "2.0.0"
__author__ = "TSP Solver Team"

from tsp.io import read_tsp_instance, write_tour, tour_cost
from tsp.timeplan import TimePlan
from tsp.seed import generate_diverse_seeds
from tsp.local import two_opt_python_wrapper, three_opt_python_wrapper
from tsp.advanced import iterated_lin_kernighan, multi_level_optimization
from tsp.distance import build_candidate_lists, get_candidate_size, test_symmetry

__all__ = [
    'read_tsp_instance',
    'write_tour',
    'tour_cost',
    'TimePlan',
    'generate_diverse_seeds',
    'two_opt_python_wrapper',
    'three_opt_python_wrapper',
    'iterated_lin_kernighan',
    'multi_level_optimization',
    'build_candidate_lists',
    'get_candidate_size',
    'test_symmetry'
]

