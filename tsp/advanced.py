"""
Advanced k-opt and Lin-Kernighan style optimizations.
Deep optimization phase for the TSP solver using pure NumPy.
"""

import numpy as np
from typing import List
import time


def compute_tour_cost_numba(tour: np.ndarray, dist: np.ndarray) -> float:
    """Fast tour cost computation."""
    n = len(tour)
    cost = 0.0
    for i in range(n):
        cost += dist[tour[i], tour[(i+1) % n]]
    return cost


def double_bridge_kick(tour: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Double-bridge perturbation for escaping local optima.
    
    Args:
        tour: Current tour
        seed: Random seed
        
    Returns:
        Perturbed tour
    """
    np.random.seed(seed)
    n = len(tour)
    
    if n < 8:
        return tour.copy()
    
    # Select 4 cut points
    positions = np.sort(np.random.choice(n, 4, replace=False))
    a, b, c, d = positions[0], positions[1], positions[2], positions[3]
    
    # Reconnect: [0:a] + [c:d] + [b:c] + [a:b] + [d:n]
    new_tour = np.concatenate((
        tour[:a],
        tour[c:d],
        tour[b:c],
        tour[a:b],
        tour[d:]
    ))
    
    return new_tour


def sequential_2opt_kicks(tour: np.ndarray, dist: np.ndarray,
                          num_kicks: int = 5) -> np.ndarray:
    """
    Apply multiple 2-opt moves to escape local optimum.
    
    Args:
        tour: Current tour
        dist: Distance matrix
        num_kicks: Number of random 2-opt moves
        
    Returns:
        Kicked tour
    """
    n = len(tour)
    current = tour.copy()
    
    for _ in range(num_kicks):
        i = np.random.randint(0, n-2)
        j = np.random.randint(i+2, n)
        
        # Apply 2-opt move
        current[i+1:j+1] = current[i+1:j+1][::-1]
    
    return current


def iterated_lin_kernighan(tour: List[int],
                           distance_matrix: np.ndarray,
                           time_limit: float = 60.0,
                           perturbation_strength: int = 4) -> List[int]:
    """
    Iterated Lin-Kernighan style optimization.
    
    This is a simplified LK approach using:
    - Intensive local search
    - Double-bridge perturbations
    - Acceptance criteria
    
    Args:
        tour: Initial tour
        distance_matrix: Distance matrix
        time_limit: Time budget for optimization
        perturbation_strength: Strength of perturbations
        
    Returns:
        Optimized tour
    """
    from tsp.local import two_opt_python_wrapper, three_opt_python_wrapper
    from tsp.io import tour_cost
    
    start_time = time.time()
    
    tour_arr = np.array(tour, dtype=np.int32)
    best_tour = tour_arr.copy()
    best_cost = tour_cost(tour, distance_matrix)
    
    current_tour = best_tour.copy()
    current_cost = best_cost
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 100  # Increased for better exploration
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        iteration += 1
        
        # Apply perturbation
        if iteration > 1:
            perturbed = double_bridge_kick(current_tour, seed=iteration)
        else:
            perturbed = current_tour.copy()
        
        # Intensive 2-opt with more iterations
        improved = two_opt_python_wrapper(perturbed.tolist(), distance_matrix,
                                         max_iters=10000)
        
        # More frequent 3-opt for better quality
        if iteration % 3 == 0:
            improved = three_opt_python_wrapper(improved, distance_matrix)
        
        improved_arr = np.array(improved, dtype=np.int32)
        improved_cost = tour_cost(improved, distance_matrix)
        
        # Update current solution (accept if better)
        if improved_cost < current_cost:
            current_tour = improved_arr
            current_cost = improved_cost
            no_improve_count = 0
            
            # Update best
            if improved_cost < best_cost:
                best_tour = improved_arr.copy()
                best_cost = improved_cost
        else:
            no_improve_count += 1
        
        # Adaptive restart if stuck
        if no_improve_count >= max_no_improve // 2:
            current_tour = best_tour.copy()
            current_cost = best_cost
            no_improve_count = 0
    
    return best_tour.tolist()


def advanced_k_opt(tour: List[int],
                   distance_matrix: np.ndarray,
                   candidates: np.ndarray,
                   time_limit: float = 30.0) -> List[int]:
    """
    Advanced k-opt with candidate lists and ejection chains.
    
    Args:
        tour: Initial tour
        distance_matrix: Distance matrix
        candidates: Candidate lists
        time_limit: Time budget
        
    Returns:
        Optimized tour
    """
    from tsp.local import two_opt_python_wrapper
    from tsp.io import tour_cost
    
    start_time = time.time()
    
    current_tour = tour.copy()
    best_tour = tour.copy()
    best_cost = tour_cost(tour, distance_matrix)
    
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Apply 2-opt with candidates
        improved = two_opt_python_wrapper(current_tour, distance_matrix,
                                         candidates, max_iters=10000)
        improved_cost = tour_cost(improved, distance_matrix)
        
        if improved_cost < best_cost:
            best_tour = improved.copy()
            best_cost = improved_cost
            current_tour = improved.copy()
        else:
            # Perturbation
            tour_arr = np.array(current_tour, dtype=np.int32)
            perturbed = double_bridge_kick(tour_arr, seed=iteration)
            current_tour = perturbed.tolist()
        
        # Check time every 10 iterations
        if iteration % 10 == 0 and time.time() - start_time >= time_limit:
            break
    
    return best_tour


def multi_level_optimization(tour: List[int],
                            distance_matrix: np.ndarray,
                            candidates: np.ndarray,
                            time_limit: float = 100.0) -> List[int]:
    """
    Multi-level optimization combining multiple strategies.
    
    Args:
        tour: Initial tour
        distance_matrix: Distance matrix
        candidates: Candidate lists
        time_limit: Total time budget
        
    Returns:
        Optimized tour
    """
    from tsp.io import tour_cost
    
    start_time = time.time()
    
    # Phase 1: Fast 2-opt (15% of time)
    phase1_time = time_limit * 0.15
    improved = advanced_k_opt(tour, distance_matrix, candidates, phase1_time)
    
    # Phase 2: Deep LK-style search (70% of time)
    phase2_time = time_limit * 0.7
    time_remaining = time_limit - (time.time() - start_time)
    phase2_budget = min(phase2_time, time_remaining)
    
    if phase2_budget > 1.0:
        improved = iterated_lin_kernighan(improved, distance_matrix, phase2_budget)
    
    # Phase 3: Final polish (remaining time)
    time_remaining = time_limit - (time.time() - start_time)
    if time_remaining > 1.0:
        # Use remaining time for more intensive search
        improved = iterated_lin_kernighan(improved, distance_matrix, time_remaining)
    
    return improved

