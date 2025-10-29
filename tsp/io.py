"""
I/O module for TSP solver.
Smart file reading with encoding detection and tour writing.
"""

import numpy as np
from typing import List, Tuple, Optional


def smart_open(path: str, mode: str = 'r'):
    """
    Smart file opening with multiple encoding fallbacks.
    
    Args:
        path: File path
        mode: File mode ('r' or 'w')
        
    Returns:
        File handle
    """
    if 'w' in mode or 'a' in mode:
        return open(path, mode, encoding='utf-8')
    
    encodings = ['utf-8', 'utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin-1']
    
    for encoding in encodings:
        try:
            f = open(path, mode, encoding=encoding)
            # Test read to verify encoding works
            pos = f.tell()
            f.read(100)
            f.seek(pos)
            return f
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise UnicodeDecodeError("Unsupported encoding", b"", 0, 1, 
                           "Could not decode file with any supported encoding")


def read_tsp_instance(filename: str) -> Tuple[str, int, np.ndarray]:
    """
    Read TSP instance from file.
    
    Args:
        filename: Path to input file
        
    Returns:
        Tuple of (problem_type, n, distance_matrix)
    """
    with smart_open(filename) as f:
        lines = f.readlines()
    
    # Parse problem type (remove BOM if present)
    problem_type = lines[0].strip()
    if problem_type.startswith('\ufeff'):
        problem_type = problem_type[1:]
    
    if problem_type not in ['EUCLIDEAN', 'NON-EUCLIDEAN']:
        raise ValueError(f"Invalid problem type: {problem_type}")
    
    # Parse number of cities
    n = int(lines[1].strip())
    if n <= 0:
        raise ValueError(f"Invalid number of cities: {n}")
    
    # Parse distance matrix
    distance_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        row_values = lines[2 + i].strip().split()
        if len(row_values) != n:
            raise ValueError(f"Row {i} has {len(row_values)} values, expected {n}")
        
        for j in range(n):
            try:
                distance_matrix[i, j] = float(row_values[j])
            except ValueError:
                raise ValueError(f"Invalid distance at ({i}, {j}): {row_values[j]}")
    
    # Validate diagonal
    for i in range(n):
        if abs(distance_matrix[i, i]) > 1e-9:
            raise ValueError(f"Diagonal ({i}, {i}) should be 0, got {distance_matrix[i, i]}")
    
    return problem_type, n, distance_matrix


def write_tour(filename: str, tour: List[int], mode: str = 'a') -> None:
    """
    Write tour to output file.
    
    Args:
        filename: Output file path
        tour: Tour as list of city indices
        mode: File mode ('w' to overwrite, 'a' to append)
    """
    with smart_open(filename, mode) as f:
        tour_str = ' '.join(map(str, tour))
        f.write(tour_str + '\n')


def tour_cost(tour: List[int], distance_matrix: np.ndarray) -> float:
    """
    Calculate tour cost efficiently.
    
    Args:
        tour: Tour as list of city indices
        distance_matrix: Distance matrix
        
    Returns:
        Total tour cost
    """
    if len(tour) <= 1:
        return 0.0
    
    # Vectorized cost calculation
    tour_arr = np.array(tour, dtype=np.int32)
    tour_next = np.roll(tour_arr, -1)
    return float(distance_matrix[tour_arr, tour_next].sum())


def validate_tour(tour: List[int], n: int) -> bool:
    """
    Validate tour is a valid permutation.
    
    Args:
        tour: Tour to validate
        n: Number of cities
        
    Returns:
        True if valid
    """
    return (len(tour) == n and 
            len(set(tour)) == n and 
            all(0 <= city < n for city in tour))

