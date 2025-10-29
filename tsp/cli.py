"""
Command-line interface for the TSP solver.
Main entry point implementing the world-class TSP solving pipeline.
"""

import argparse
import signal
import sys
import time
import os
import numpy as np
from typing import Optional

from tsp.io import read_tsp_instance, write_tour, tour_cost, validate_tour
from tsp.timeplan import TimePlan
from tsp.distance import (build_candidate_lists, get_candidate_size, 
                          test_symmetry, compute_row_min_checksum)
from tsp.seed import generate_diverse_seeds
from tsp.local import two_opt_python_wrapper
from tsp.advanced import multi_level_optimization


# Global state for signal handling
best_tour_global = None
best_cost_global = float('inf')
output_file_global = None


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully."""
    global best_tour_global, output_file_global
    
    print(f"\nReceived signal {signum}. Writing best solution and exiting...")
    
    if best_tour_global is not None and output_file_global is not None:
        try:
            write_tour(output_file_global, best_tour_global, mode='a')
            print(f"Final solution written (cost: {best_cost_global:.2f})")
        except Exception as e:
            print(f"Error writing final solution: {e}")
    
    sys.exit(0)


def incremental_writer(tour, cost, output_file, last_cost, threshold=0.003):
    """
    Write tour if improvement exceeds threshold.
    
    Args:
        tour: Current tour
        cost: Current cost
        output_file: Output file path
        last_cost: Last written cost
        threshold: Improvement threshold (0.003 = 0.3%)
        
    Returns:
        New last_cost if written, otherwise old last_cost
    """
    if cost < last_cost * (1 - threshold):
        write_tour(output_file, tour, mode='a')
        print(f"  → Improvement: {last_cost:.2f} → {cost:.2f} ({((last_cost-cost)/last_cost*100):.2f}%)")
        return cost
    return last_cost


def solve_tsp(input_file: str, output_file: str, 
              max_time: float = 300.0, seed: Optional[int] = None,
              aggressive: bool = False) -> None:
    """
    Main TSP solving pipeline following Task.md guidelines.
    
    Pipeline:
    1. Pre-flight analysis (0.5s)
    2. Global seeding (3% of time)
    3. Fast local search (10% of time)
    4. LK-style deep optimization (70% of remaining)
    5. Parallel restarts (15% of remaining)
    6. Buffer for safety (4% margin)
    """
    global best_tour_global, best_cost_global, output_file_global
    
    # Setup signal handling
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    output_file_global = output_file
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    print("="*60)
    print("WORLD-CLASS TSP SOLVER v2.0")
    print("="*60)
    
    # ============================================================
    # PHASE 0: Pre-flight analysis
    # ============================================================
    preflight_start = time.time()
    print("\n[PRE-FLIGHT] Loading and analyzing instance...")
    
    try:
        problem_type, n, distance_matrix = read_tsp_instance(input_file)
        print(f"  Problem: {problem_type}, Cities: {n}")
    except Exception as e:
        print(f"ERROR: Failed to read input: {e}")
        sys.exit(1)
    
    # Handle trivial cases
    if n <= 3:
        trivial_tour = list(range(n))
        open(output_file, 'w').close()  # Clear file
        write_tour(output_file, trivial_tour, mode='w')
        print(f"Trivial solution for n={n}")
        return
    
    # Test symmetry and compute checksum
    is_symmetric = test_symmetry(distance_matrix)
    checksum = compute_row_min_checksum(distance_matrix)
    print(f"  Symmetric: {is_symmetric}, Checksum: {checksum:.2f}")
    
    # Initialize time planner
    timeplan = TimePlan(max_time, n, mode=("aggressive" if aggressive else "normal"))
    print(f"\n{timeplan}")
    
    preflight_time = time.time() - preflight_start
    print(f"  Pre-flight completed in {preflight_time:.2f}s")
    
    # Clear output file
    open(output_file, 'w').close()
    
    # ============================================================
    # PHASE 1: Global seeding (3% of time, ~9s for 300s limit)
    # ============================================================
    print(f"\n[PHASE 1] Global Seeding ({timeplan.time_for_phase('seed'):.1f}s budget)")
    seed_start = time.time()
    
    # Generate diverse seeds
    print("  Generating diverse initial tours...")
    tours = generate_diverse_seeds(distance_matrix, problem_type, num_seeds=16)
    
    # Evaluate and sort by cost
    tour_costs = [(tour, tour_cost(tour.tolist(), distance_matrix)) for tour in tours]
    tour_costs.sort(key=lambda x: x[1])
    
    best_tour = tour_costs[0][0].tolist()
    best_cost = tour_costs[0][1]
    
    # Write initial best
    write_tour(output_file, best_tour, mode='w')
    
    best_tour_global = best_tour
    best_cost_global = best_cost
    last_written_cost = best_cost
    
    seed_time = time.time() - seed_start
    print(f"  Generated {len(tours)} tours, best: {best_cost:.2f}")
    print(f"  Seeding completed in {seed_time:.2f}s")
    
    # ============================================================
    # PHASE 2: Fast local search (10% of time, ~30s for 300s limit)
    # ============================================================
    print(f"\n[PHASE 2] Fast Local Search ({timeplan.time_for_phase('fast'):.1f}s budget)")
    fast_start = time.time()
    
    # Build candidate lists
    k = get_candidate_size(n, problem_type == 'EUCLIDEAN', aggressive=aggressive)
    candidates = build_candidate_lists(distance_matrix, k)
    print(f"  Built candidate lists (k={k})")
    
    # Fast 2-opt on best 5 tours
    print("  Running fast 2-opt on top 5 seeds...")
    for i, (tour, cost) in enumerate(tour_costs[:5]):
        if not timeplan.should_continue():
            break
        
        improved = two_opt_python_wrapper(tour.tolist(), distance_matrix, 
                                         candidates, max_iters=5000)
        improved_cost = tour_cost(improved, distance_matrix)
        
        print(f"    Tour {i+1}: {cost:.2f} → {improved_cost:.2f}")
        
        if improved_cost < best_cost:
            best_tour = improved
            best_cost = improved_cost
            best_tour_global = best_tour
            best_cost_global = best_cost
            last_written_cost = incremental_writer(best_tour, best_cost, 
                                                  output_file, last_written_cost)
    
    fast_time = time.time() - fast_start
    print(f"  Fast search completed in {fast_time:.2f}s, best: {best_cost:.2f}")
    
    # ============================================================
    # PHASE 3: LK-Helsgaun core (70% of remaining, ~200s for 300s limit)
    # ============================================================
    print(f"\n[PHASE 3] Deep Optimization ({timeplan.time_for_phase('lk'):.1f}s budget)")
    lk_start = time.time()
    
    lk_time_budget = timeplan.time_for_phase('lk')
    time_remaining = max(0, timeplan.remaining() - timeplan.time_for_phase('parallel') 
                        - timeplan.time_for_phase('buffer'))
    lk_actual_budget = min(lk_time_budget, time_remaining)
    
    if lk_actual_budget > 5.0:
        print(f"  Running multi-level optimization...")
        optimized = multi_level_optimization(best_tour, distance_matrix, 
                                            candidates, lk_actual_budget)
        optimized_cost = tour_cost(optimized, distance_matrix)
        
        print(f"  LK optimization: {best_cost:.2f} → {optimized_cost:.2f}")
        
        if optimized_cost < best_cost:
            best_tour = optimized
            best_cost = optimized_cost
            best_tour_global = best_tour
            best_cost_global = best_cost
            last_written_cost = incremental_writer(best_tour, best_cost,
                                                  output_file, last_written_cost)
    
    lk_time = time.time() - lk_start
    print(f"  Deep optimization completed in {lk_time:.2f}s")
    
    # ============================================================
    # PHASE 4: Parallel restarts (15% of remaining, ~45s for 300s limit)
    # ============================================================
    print(f"\n[PHASE 4] Parallel Restarts ({timeplan.time_for_phase('parallel'):.1f}s budget)")
    parallel_start = time.time()
    
    parallel_budget = timeplan.time_for_phase('parallel')
    time_remaining = max(0, timeplan.remaining() - timeplan.time_for_phase('buffer'))
    parallel_actual_budget = min(parallel_budget, time_remaining)
    
    if parallel_actual_budget > 5.0:
        print("  Running final polish with parallel 2-opt...")
        
        # Multiple aggressive 2-opt passes
        num_passes = max(2, int(parallel_actual_budget / 5))
        for pass_num in range(num_passes):
            if not timeplan.should_continue():
                break
            
            improved = two_opt_python_wrapper(best_tour, distance_matrix,
                                             candidates, max_iters=10000)
            improved_cost = tour_cost(improved, distance_matrix)
            
            if improved_cost < best_cost:
                best_tour = improved
                best_cost = improved_cost
                best_tour_global = best_tour
                best_cost_global = best_cost
                last_written_cost = incremental_writer(best_tour, best_cost,
                                                      output_file, last_written_cost)
                print(f"  Pass {pass_num+1}: New best {best_cost:.2f}")
    
    # Use any remaining time for continuous polish if aggressive
    if aggressive:
        while timeplan.should_continue():
            improved = two_opt_python_wrapper(best_tour, distance_matrix,
                                             candidates, max_iters=20000)
            improved_cost = tour_cost(improved, distance_matrix)
            if improved_cost < best_cost:
                best_tour = improved
                best_cost = improved_cost
                best_tour_global = best_tour
                best_cost_global = best_cost
                last_written_cost = incremental_writer(best_tour, best_cost,
                                                      output_file, last_written_cost)
            else:
                break

    parallel_time = time.time() - parallel_start
    print(f"  Parallel restarts completed in {parallel_time:.2f}s")
    
    # ============================================================
    # FINAL: Write best solution
    # ============================================================
    print("\n" + "="*60)
    
    # Validate final tour
    if validate_tour(best_tour, n):
        write_tour(output_file, best_tour, mode='a')
        print(f"✓ FINAL SOLUTION: {best_cost:.2f}")
    else:
        print("ERROR: Final tour is invalid!")
        sys.exit(1)
    
    total_time = timeplan.elapsed()
    print(f"✓ Total time: {total_time:.2f}s / {max_time:.2f}s")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="World-Class TSP Solver (Pure NumPy Implementation)"
    )
    parser.add_argument("input_file", help="Input TSP instance file")
    parser.add_argument("output_file", help="Output tour file")
    parser.add_argument("--time", type=float, default=300.0,
                       help="Time limit in seconds (default: 300)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--aggressive", action="store_true",
                       help="Use aggressive mode (maximize deep optimization and utilize nearly full time)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    solve_tsp(args.input_file, args.output_file, args.time, args.seed, args.aggressive)


if __name__ == "__main__":
    main()

