"""
Adaptive time management module.
Implements dynamic budget allocation based on problem size and time limits.
"""

import time
from typing import Dict


class TimePlan:
    """Dynamic time budget manager for TSP solver."""
    
    def __init__(self, max_time: float = 300.0, n: int = 100, mode: str = "normal"):
        """
        Initialize time planner.
        
        Args:
            max_time: Maximum allowed time in seconds
            n: Problem size (number of cities)
        """
        self.max_time = max_time
        self.n = n
        self.start_time = time.time()
        
        # Calculate time allocations based on mode
        if mode == "aggressive":
            # Favor deep optimization with a smaller spare and faster early phases
            self.T_spare = 0.02 * max_time  # 2% safety margin
            self.T_seed = min(0.02 * max_time, 6)  # 2% or 6s max for seeding
            self.T_fast = min(0.08 * max_time, 20)  # 8% or 20s max for fast local search
            self.T_left = max_time - self.T_seed - self.T_fast - self.T_spare
            self.T_lk = 0.80 * self.T_left  # 80% of remaining for LK core
            self.T_parallel = 0.15 * self.T_left  # 15% for parallel restarts
            self.T_buffer = self.T_spare
        else:
            # Default allocations per Task.md
            self.T_spare = 0.04 * max_time  # 4% safety margin
            self.T_seed = min(0.03 * max_time, 10)  # 3% or 10s max for seeding
            self.T_fast = min(0.10 * max_time, 25)  # 10% or 25s max for fast local search
            self.T_left = max_time - self.T_seed - self.T_fast - self.T_spare
            self.T_lk = 0.70 * self.T_left  # 70% of remaining for LK core
            self.T_parallel = 0.15 * self.T_left  # 15% for parallel restarts
            self.T_buffer = self.T_spare  # Final buffer
        
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def remaining(self) -> float:
        """Get remaining time before hard limit."""
        return max(0, self.max_time - self.elapsed())
    
    def should_continue(self) -> bool:
        """Check if we should continue (with safety margin)."""
        return self.elapsed() < (self.max_time - self.T_spare)
    
    def time_for_phase(self, phase: str) -> float:
        """
        Get time allocation for a specific phase.
        
        Args:
            phase: One of 'seed', 'fast', 'lk', 'parallel', 'buffer'
            
        Returns:
            Time in seconds allocated for this phase
        """
        allocations = {
            'seed': self.T_seed,
            'fast': self.T_fast,
            'lk': self.T_lk,
            'parallel': self.T_parallel,
            'buffer': self.T_buffer
        }
        return allocations.get(phase, 0.0)
    
    def check_phase_time(self, phase_start: float, phase: str) -> bool:
        """
        Check if current phase should continue.
        
        Args:
            phase_start: Time when phase started
            phase: Phase name
            
        Returns:
            True if phase should continue
        """
        phase_elapsed = time.time() - phase_start
        phase_budget = self.time_for_phase(phase)
        return phase_elapsed < phase_budget and self.should_continue()
    
    def get_budget_summary(self) -> Dict[str, float]:
        """Get summary of time budget allocation."""
        return {
            'total': self.max_time,
            'seed': self.T_seed,
            'fast': self.T_fast,
            'lk': self.T_lk,
            'parallel': self.T_parallel,
            'buffer': self.T_buffer,
            'spare': self.T_spare
        }
    
    def __repr__(self) -> str:
        """String representation of time plan."""
        budget = self.get_budget_summary()
        return (f"TimePlan(total={budget['total']:.1f}s, "
                f"seed={budget['seed']:.1f}s, "
                f"fast={budget['fast']:.1f}s, "
                f"lk={budget['lk']:.1f}s, "
                f"parallel={budget['parallel']:.1f}s)")

