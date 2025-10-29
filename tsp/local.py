"""
Pure NumPy local search kernels.
Fast 2-opt and 3-opt with candidate lists and don't-look bits.
"""

import numpy as np
from typing import List, Tuple


def two_opt_delta(tour: np.ndarray, i: int, j: int, 
                  dist: np.ndarray) -> float:
    """
    Calculate delta for 2-opt move.
    
    Args:
        tour: Current tour
        i, j: Edge indices
        dist: Distance matrix
        
    Returns:
        Cost change (negative = improvement)
    """
    n = len(tour)
    a = tour[i]
    b = tour[(i + 1) % n]
    c = tour[j]
    d = tour[(j + 1) % n]
    
    old_cost = dist[a, b] + dist[c, d]
    new_cost = dist[a, c] + dist[b, d]
    
    return new_cost - old_cost


def apply_two_opt(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Apply 2-opt move: reverse segment from i+1 to j.
    
    Args:
        tour: Current tour
        i, j: Edge indices
        
    Returns:
        New tour
    """
    n = len(tour)
    new_tour = tour.copy()
    
    # Reverse segment from i+1 to j (inclusive)
    left = (i + 1) % n
    right = j
    
    if left <= right:
        new_tour[left:right+1] = new_tour[left:right+1][::-1]
    else:
        # Wrap-around case
        segment = np.concatenate((new_tour[left:], new_tour[:right+1]))
        segment = segment[::-1]
        new_tour[left:] = segment[:n-left]
        new_tour[:right+1] = segment[n-left:]
    
    return new_tour


def two_opt_first_improvement(tour: np.ndarray, 
                               dist: np.ndarray, 
                               cand: np.ndarray,
                               dont_look: np.ndarray,
                               max_iters: int = 10000) -> Tuple[np.ndarray, float, int]:
    """
    First-improvement 2-opt with candidate lists and don't-look bits.
    
    Args:
        tour: Initial tour
        dist: Distance matrix
        cand: Candidate lists (n × k)
        dont_look: Don't-look bits array
        max_iters: Maximum iterations
        
    Returns:
        Tuple of (improved_tour, improvement_count, iterations)
    """
    n = len(tour)
    k = cand.shape[1] if cand.shape[0] > 0 else 0
    current_tour = tour.copy()
    improvements = 0
    iterations = 0
    
    improved = True
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        
        for i in range(n):
            if dont_look[i]:
                continue
            
            city_i = current_tour[i]
            
            # Check candidate edges
            for c_idx in range(k):
                if c_idx >= k:
                    break
                    
                candidate_city = cand[city_i, c_idx]
                
                # Find position of candidate in tour
                j = -1
                for pos in range(n):
                    if current_tour[pos] == candidate_city:
                        j = pos
                        break
                
                if j == -1 or j == i or j == (i + 1) % n:
                    continue
                
                # Ensure proper edge ordering
                if i < j:
                    delta = two_opt_delta(current_tour, i, j, dist)
                else:
                    delta = two_opt_delta(current_tour, j, i, dist)
                
                if delta < -1e-9:  # Improvement found
                    if i < j:
                        current_tour = apply_two_opt(current_tour, i, j)
                    else:
                        current_tour = apply_two_opt(current_tour, j, i)
                    
                    improvements += 1
                    improved = True
                    
                    # Reset don't-look bits near modified edges
                    for pos in range(max(0, i - k), min(n, i + k + 1)):
                        dont_look[pos] = False
                    for pos in range(max(0, j - k), min(n, j + k + 1)):
                        dont_look[pos] = False
                    
                    break
            
            if improved:
                break
            else:
                dont_look[i] = True
    
    return current_tour, improvements, iterations


def two_opt_full_scan(tour: np.ndarray, 
                      dist: np.ndarray,
                      max_no_improve: int = 3) -> np.ndarray:
    """
    Full 2-opt scan without candidate lists (for small instances).
    
    Args:
        tour: Initial tour
        dist: Distance matrix
        max_no_improve: Maximum passes without improvement
        
    Returns:
        Improved tour
    """
    n = len(tour)
    current_tour = tour.copy()
    no_improve = 0
    
    while no_improve < max_no_improve:
        improved = False
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Skip full reversal
                
                delta = two_opt_delta(current_tour, i, j, dist)
                
                if delta < -1e-9:
                    current_tour = apply_two_opt(current_tour, i, j)
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    return current_tour


def three_opt_delta_case1(tour: np.ndarray, i: int, j: int, k: int,
                          dist: np.ndarray) -> float:
    """Calculate delta for 3-opt case 1: simple reconnection."""
    n = len(tour)
    a, b = tour[i], tour[(i+1) % n]
    c, d = tour[j], tour[(j+1) % n]
    e, f = tour[k], tour[(k+1) % n]
    
    old = dist[a,b] + dist[c,d] + dist[e,f]
    new = dist[a,c] + dist[b,d] + dist[e,f]
    return new - old


def three_opt_restricted(tour: np.ndarray,
                         dist: np.ndarray,
                         max_iters: int = 100) -> np.ndarray:
    """
    Restricted 3-opt: only check promising cases.
    
    Args:
        tour: Current tour
        dist: Distance matrix
        max_iters: Maximum iterations
        
    Returns:
        Improved tour
    """
    n = len(tour)
    current_tour = tour.copy()
    
    for _ in range(max_iters):
        improved = False
        
        for i in range(n - 4):
            for j in range(i + 2, min(i + 15, n - 2)):
                for k in range(j + 2, min(j + 15, n)):
                    # Only check simplest 3-opt case for speed
                    delta = three_opt_delta_case1(current_tour, i, j, k, dist)
                    
                    if delta < -1e-9:
                        # Apply move (simplified)
                        current_tour = apply_two_opt(current_tour, i, j)
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_tour


def two_opt_python_wrapper(tour: List[int], 
                           distance_matrix: np.ndarray,
                           candidates: np.ndarray = None,
                           max_iters: int = 10000) -> List[int]:
    """
    Python wrapper for Numba 2-opt.
    
    Args:
        tour: Tour as Python list
        distance_matrix: Distance matrix
        candidates: Candidate lists (optional)
        max_iters: Maximum iterations
        
    Returns:
        Improved tour as Python list
    """
    tour_arr = np.array(tour, dtype=np.int32)
    n = len(tour)
    
    # Initialize don't-look bits
    dont_look = np.zeros(n, dtype=np.bool_)
    
    if candidates is not None and len(candidates) > 0:
        improved_tour, _, _ = two_opt_first_improvement(
            tour_arr, distance_matrix, candidates, dont_look, max_iters
        )
    else:
        improved_tour = two_opt_full_scan(tour_arr, distance_matrix)
    
    return improved_tour.tolist()


def three_opt_python_wrapper(tour: List[int],
                             distance_matrix: np.ndarray) -> List[int]:
    """
    Python wrapper for restricted 3-opt.
    
    Args:
        tour: Tour as Python list
        distance_matrix: Distance matrix
        
    Returns:
        Improved tour as Python list
    """
    tour_arr = np.array(tour, dtype=np.int32)
    improved_tour = three_opt_restricted(tour_arr, distance_matrix)
    return improved_tour.tolist()

