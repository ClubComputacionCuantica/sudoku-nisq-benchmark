"""Hardware metadata collectors for quantum backends.

This module provides backend-specific hardware calibration data collection
for execution metadata recording (Stage 5). Collectors extract T1/T2 times,
gate error rates, readout errors, and other calibration data at execution time.

Supported Providers:
    - IBM: BackendV2 with Target-based calibration data
    - Aer: Local simulator (returns empty metadata)
    
Provider-Agnostic Interface:
    collect_hardware_metadata(backend) -> Dict[str, Any]
    
Usage:
    from sudoku_nisq.metadata.collectors import collect_hardware_metadata
    
    # Automatically detects provider and extracts calibration data
    hardware_snapshot = collect_hardware_metadata(backend)
"""

from .dispatcher import collect_hardware_metadata
from .ibm import collect_ibm_hardware_metadata
from .aer import collect_aer_hardware_metadata

__all__ = [
    "collect_hardware_metadata",
    "collect_ibm_hardware_metadata",
    "collect_aer_hardware_metadata",
]
