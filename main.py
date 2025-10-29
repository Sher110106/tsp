#!/usr/bin/env python3
"""
Main entry point for the world-class TSP solver.
This is the file to run for solving TSP instances.

Usage:
    python3 solve_tsp.py input.txt output.txt [--time 300] [--seed 42]
"""

import sys
import os

# Add tsp package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsp.cli import main

if __name__ == "__main__":
    main()

