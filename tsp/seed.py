"""
Construction heuristics for initial tour generation.
NN, Cheapest-Insertion, α-Random NN, Farthest-Insertion using pure NumPy.
"""

import numpy as np
from typing import List, Tuple
import random


def nearest_neighbor_numba(dist: np.ndarray, start: int) -> np.ndarray:
    """
    Numba-compiled Nearest Neighbor construction.
    
    Args:
        dist: Distance matrix
        start: Starting city
        
    Returns:
        Tour as numpy array
    """
    n = dist.shape[0]
    tour = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)
    
    tour[0] = start
    visited[start] = True
    current = start
    
    for i in range(1, n):
        best_dist = np.inf
        best_city = -1
        
        for city in range(n):
            if not visited[city]:
                d = dist[current, city]
                if d < best_dist:
                    best_dist = d
                    best_city = city
        
        tour[i] = best_city
        visited[best_city] = True
        current = best_city
    
    return tour


def alpha_random_nn_numba(dist: np.ndarray, start: int, 
                          alpha: float = 0.25, seed: int = 42) -> np.ndarray:
    """
    α-Random Nearest Neighbor with controlled randomness.
    
    Args:
        dist: Distance matrix
        start: Starting city
        alpha: Randomness parameter (0 = greedy, 1 = random)
        seed: Random seed
        
    Returns:
        Tour as numpy array
    """
    np.random.seed(seed)
    n = dist.shape[0]
    tour = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)
    
    tour[0] = start
    visited[start] = True
    current = start
    
    for i in range(1, n):
        # Get unvisited cities
        unvisited = []
        for city in range(n):
            if not visited[city]:
                unvisited.append(city)
        
        if len(unvisited) == 0:
            break
        
        # Sort by distance
        distances = np.array([dist[current, city] for city in unvisited])
        sorted_indices = np.argsort(distances)
        
        # Select from top alpha% candidates
        num_candidates = max(1, int(len(unvisited) * alpha))
        candidate_idx = np.random.randint(0, num_candidates)
        chosen_city = unvisited[sorted_indices[candidate_idx]]
        
        tour[i] = chosen_city
        visited[chosen_city] = True
        current = chosen_city
    
    return tour


def cheapest_insertion_numba(dist: np.ndarray) -> np.ndarray:
    """
    Cheapest Insertion heuristic.
    
    Args:
        dist: Distance matrix
        
    Returns:
        Tour as numpy array
    """
    n = dist.shape[0]
    
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    # Start with triangle of minimum perimeter
    best_i, best_j, best_k = 0, 1, 2
    best_perim = dist[0,1] + dist[1,2] + dist[2,0]
    
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    perim = dist[i,j] + dist[j,k] + dist[k,i]
                    if perim < best_perim:
                        best_perim = perim
                        best_i, best_j, best_k = i, j, k
    
    # Build tour incrementally
    tour_list = [best_i, best_j, best_k]
    visited = np.zeros(n, dtype=np.bool_)
    visited[best_i] = True
    visited[best_j] = True
    visited[best_k] = True
    
    # Insert remaining cities
    while len(tour_list) < n:
        best_city = -1
        best_position = -1
        best_cost_increase = np.inf
        
        for city in range(n):
            if visited[city]:
                continue
            
            # Try inserting at each position
            for pos in range(len(tour_list)):
                prev_city = tour_list[pos]
                next_city = tour_list[(pos + 1) % len(tour_list)]
                
                cost_increase = (dist[prev_city, city] + 
                               dist[city, next_city] - 
                               dist[prev_city, next_city])
                
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_city = city
                    best_position = pos + 1
        
        # Insert best city
        tour_list.insert(best_position, best_city)
        visited[best_city] = True
    
    return np.array(tour_list, dtype=np.int32)


def farthest_insertion_numba(dist: np.ndarray) -> np.ndarray:
    """
    Farthest Insertion heuristic (good for non-Euclidean).
    
    Args:
        dist: Distance matrix
        
    Returns:
        Tour as numpy array
    """
    n = dist.shape[0]
    
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    # Start with two farthest cities
    max_dist = 0.0
    start_i, start_j = 0, 1
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] > max_dist:
                max_dist = dist[i,j]
                start_i, start_j = i, j
    
    tour_list = [start_i, start_j]
    visited = np.zeros(n, dtype=np.bool_)
    visited[start_i] = True
    visited[start_j] = True
    
    # Insert remaining cities (farthest from tour first)
    while len(tour_list) < n:
        # Find farthest city from tour
        best_city = -1
        best_min_dist = -1.0
        
        for city in range(n):
            if visited[city]:
                continue
            
            # Find minimum distance to tour
            min_dist_to_tour = np.inf
            for tour_city in tour_list:
                if dist[city, tour_city] < min_dist_to_tour:
                    min_dist_to_tour = dist[city, tour_city]
            
            if min_dist_to_tour > best_min_dist:
                best_min_dist = min_dist_to_tour
                best_city = city
        
        # Find best insertion position
        best_position = 0
        best_cost_increase = np.inf
        
        for pos in range(len(tour_list)):
            prev_city = tour_list[pos]
            next_city = tour_list[(pos + 1) % len(tour_list)]
            
            cost_increase = (dist[prev_city, best_city] + 
                           dist[best_city, next_city] - 
                           dist[prev_city, next_city])
            
            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_position = pos + 1
        
        # Insert city
        tour_list.insert(best_position, best_city)
        visited[best_city] = True
    
    return np.array(tour_list, dtype=np.int32)


def generate_diverse_seeds(distance_matrix: np.ndarray,
                           problem_type: str,
                           num_seeds: int = 16) -> List[np.ndarray]:
    """
    Generate diverse initial tours using multiple heuristics.
    
    Args:
        distance_matrix: Distance matrix
        problem_type: 'EUCLIDEAN' or 'NON-EUCLIDEAN'
        num_seeds: Number of seeds to generate
        
    Returns:
        List of tours (as numpy arrays)
    """
    n = distance_matrix.shape[0]
    tours = []
    
    # 1. Nearest Neighbor from 5 different starts for better coverage
    nn_starts = [0, n//4, n//3, n//2, 2*n//3]
    for start in nn_starts:
        tour = nearest_neighbor_numba(distance_matrix, start)
        tours.append(tour)
        if len(tours) >= num_seeds:
            break
    
    # 2. Greedy Cheapest Insertion
    if len(tours) < num_seeds:
        tour = cheapest_insertion_numba(distance_matrix)
        tours.append(tour)
    
    # 3. Multiple α-Random NN with varying alpha values
    if len(tours) < num_seeds:
        # Bias to tighter alpha for large non-euclidean to improve quality
        if problem_type == 'NON-EUCLIDEAN' and n >= 200:
            alpha_values = [0.10, 0.15, 0.20, 0.25, 0.30]
            seeds_list = [42, 123, 456, 789, 2025]
        else:
            alpha_values = [0.15, 0.25, 0.35, 0.5]
            seeds_list = [42, 123, 456, 789]
        for alpha, seed_val in zip(alpha_values, seeds_list):
            if len(tours) >= num_seeds:
                break
            tour = alpha_random_nn_numba(distance_matrix, 0, alpha, seed_val)
            tours.append(tour)
    
    # 4. Farthest Insertion (good for both types)
    if len(tours) < num_seeds:
        tour = farthest_insertion_numba(distance_matrix)
        tours.append(tour)
        # Add a reversed variant to diversify edges for large non-euclidean
        if problem_type == 'NON-EUCLIDEAN' and n >= 200 and len(tours) < num_seeds:
            tours.append(tour[::-1])
    
    # 5. More diverse NN starts with different seeds
    seed_counter = 1000
    while len(tours) < num_seeds:
        start = (seed_counter * 13) % n  # Pseudo-random but deterministic
        tour = nearest_neighbor_numba(distance_matrix, start)
        tours.append(tour)
        seed_counter += 1
    
    return tours[:num_seeds]

