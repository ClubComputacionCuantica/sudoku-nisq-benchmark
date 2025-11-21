"""Lightweight memory tracking utilities for quantum circuit construction."""

import os
import psutil
from typing import Dict, Optional


class MemoryTracker:
    """Lightweight memory usage tracker.
    
    Tracks memory usage at key points during circuit construction to help
    identify memory bottlenecks when problem sizes grow large.
    
    Uses psutil for cross-platform memory monitoring with minimal overhead.
    """
    
    def __init__(self):
        """Initialize memory tracker."""
        self.process = psutil.Process(os.getpid())
        self.snapshots: Dict[str, float] = {}
        self.initial_memory: Optional[float] = None
        
    def start(self):
        """Record initial memory usage."""
        self.initial_memory = self._get_memory_mb()
        self.snapshots['initial'] = self.initial_memory
        
    def snapshot(self, label: str):
        """Record memory usage at a specific point.
        
        Args:
            label: Descriptive label for this measurement point
        """
        memory_mb = self._get_memory_mb()
        self.snapshots[label] = memory_mb
        
    def peak(self) -> float:
        """Get peak memory usage in MB.
        
        Returns:
            Peak memory usage across all snapshots in MB
        """
        if not self.snapshots:
            return 0.0
        return max(self.snapshots.values())
        
    def delta(self, from_label: str = 'initial') -> float:
        """Calculate memory increase from a specific point.
        
        Args:
            from_label: Label to measure from (default: 'initial')
            
        Returns:
            Memory increase in MB, or 0.0 if label not found
        """
        if from_label not in self.snapshots:
            return 0.0
        current = self._get_memory_mb()
        return current - self.snapshots[from_label]
        
    def report(self) -> Dict[str, float]:
        """Generate memory usage report.
        
        Returns:
            Dictionary with memory statistics:
            - 'initial_mb': Starting memory
            - 'current_mb': Current memory
            - 'peak_mb': Peak memory across all snapshots
            - 'delta_mb': Increase from initial
            - 'snapshots': All recorded snapshots
        """
        current = self._get_memory_mb()
        return {
            'initial_mb': self.initial_memory or 0.0,
            'current_mb': current,
            'peak_mb': self.peak(),
            'delta_mb': current - (self.initial_memory or 0.0),
            'snapshots': self.snapshots.copy()
        }
        
    def _get_memory_mb(self) -> float:
        """Get current process memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        # Use RSS (Resident Set Size) for actual physical memory
        mem_info = self.process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert bytes to MB
        
    def __enter__(self):
        """Context manager entry - start tracking."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record final snapshot."""
        self.snapshot('final')
        return False  # Don't suppress exceptions
