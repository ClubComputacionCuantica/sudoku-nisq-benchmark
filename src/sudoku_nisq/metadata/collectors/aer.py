"""Aer simulator hardware metadata collector.

Aer simulators don't have real hardware calibration data, so this collector
returns an empty metadata structure. This maintains consistency in the
execution metadata recording pipeline.
"""

from typing import Dict, Any


def collect_aer_hardware_metadata(backend: Any) -> Dict[str, Any]:
    """Collect hardware metadata from Aer simulator backend.
    
    Aer simulators don't have real hardware calibration data (no T1/T2,
    no gate errors, no readout errors). This collector returns an empty
    metadata structure with only the backend name and provider.
    
    Args:
        backend: Aer simulator backend instance
        
    Returns:
        Dict[str, Any]: Empty hardware metadata structure:
            {
                "backend_name": str,
                "provider": "aer",
                "calibration_timestamp": None,
                "single_qubit_gate_error": {},
                "two_qubit_gate_error": {},
                "readout_error": {},
                "t1_times": {},
                "t2_times": {},
                "extra_properties": {}
            }
            
    Notes:
        - Always returns empty calibration data
        - Maintains consistent structure with IBM collector
        - Useful for distinguishing simulator vs real hardware runs
    """
    backend_name = "aer_simulator"
    backend_name_attr = getattr(backend, "name", None)
    if callable(backend_name_attr):
        try:
            backend_name = backend_name_attr()
        except Exception:
            backend_name = "aer_simulator"
    elif isinstance(backend_name_attr, str) and backend_name_attr:
        backend_name = backend_name_attr
    
    return {
        "backend_name": backend_name,
        "provider": "aer",
        "calibration_timestamp": None,
        "single_qubit_gate_error": {},
        "two_qubit_gate_error": {},
        "readout_error": {},
        "t1_times": {},
        "t2_times": {},
        "extra_properties": {},
    }
