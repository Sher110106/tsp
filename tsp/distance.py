"""
Distance matrix utilities.
Pre-computation, symmetry detection, and memory optimization.
"""

import numpy as np
from typing import Tuple


def test_symmetry(distance_matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Test if distance matrix is symmetric.
    
    Args:
        distance_matrix: n×n distance matrix
        tolerance: Tolerance for floating-point comparison
        
    Returns:
        True if symmetric
    """
    n = distance_matrix.shape[0]
    max_diff = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(distance_matrix[i, j] - distance_matrix[j, i])
            max_diff = max(max_diff, diff)
            if diff > tolerance:
                return False
    
    return True


def compute_row_min_checksum(distance_matrix: np.ndarray) -> float:
    """
    Compute checksum (sum of row minimums) for corruption detection.
    
    Args:
        distance_matrix: n×n distance matrix
        
    Returns:
        Checksum value
    """
    n = distance_matrix.shape[0]
    checksum = 0.0
    
    for i in range(n):
        # Get minimum non-zero distance in row
        row = distance_matrix[i, :]
        non_zero = row[row > 1e-9]
        if len(non_zero) > 0:
            checksum += non_zero.min()
    
    return checksum


def optimize_matrix_storage(distance_matrix: np.ndarray, 
                           is_symmetric: bool) -> Tuple[np.ndarray, bool]:
    """
    Optimize matrix storage for memory efficiency.
    
    Args:
        distance_matrix: Original distance matrix
        is_symmetric: Whether matrix is symmetric
        
    Returns:
        Tuple of (optimized_matrix, is_upper_triangle_format)
    """
    n = distance_matrix.shape[0]
    
    # For large instances, use float32 to save memory
    if n >= 150:
        optimized = distance_matrix.astype(np.float32)
    else:
        optimized = distance_matrix.astype(np.float64)
    
    # Make sure matrix is contiguous in memory for cache efficiency
    optimized = np.ascontiguousarray(optimized)
    
    return optimized, False


def build_candidate_lists(distance_matrix: np.ndarray, 
                         k: int) -> np.ndarray:
    """
    Build k-nearest neighbor candidate lists for each city.
    
    Args:
        distance_matrix: n×n distance matrix
        k: Number of candidates per city
        
    Returns:
        n×k array of candidate city indices
    """
    n = distance_matrix.shape[0]
    k = min(k, n - 1)  # Can't have more candidates than cities
    
    candidates = np.zeros((n, k), dtype=np.int32)
    
    for i in range(n):
        # Get distances to all other cities
        distances = distance_matrix[i, :].copy()
        distances[i] = np.inf  # Exclude self
        
        # Get k nearest neighbors
        nearest_indices = np.argpartition(distances, min(k, n-1))[:k]
        # Sort the k nearest by distance
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        
        candidates[i, :] = nearest_indices
    
    return candidates


def get_candidate_size(n: int, is_euclidean: bool) -> int:
    """
    Determine optimal candidate list size.
    
    Args:
        n: Number of cities
        is_euclidean: Whether problem is Euclidean
        
    Returns:
        Candidate list size
    """
    if is_euclidean:
        return min(30, max(20, n // 8))
    else:
        return min(40, max(25, n // 6))


def precompute_distances(coords: np.ndarray) -> np.ndarray:
    """
    Precompute Euclidean distance matrix from coordinates using pure NumPy.
    
    Args:
        coords: n×2 array of coordinates
        
    Returns:
        n×n distance matrix
    """
    n = coords.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

