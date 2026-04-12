"""Hardware metadata collector dispatcher.

Automatically detects backend provider and routes to appropriate collector.
Provides unified interface for collecting hardware calibration data across
IBM, Aer, and future providers.
"""

from typing import Dict, Any
import logging

from .ibm import collect_ibm_hardware_metadata
from .aer import collect_aer_hardware_metadata


logger = logging.getLogger(__name__)


def collect_hardware_metadata(backend: Any) -> Dict[str, Any]:
    """Collect hardware calibration metadata from quantum backend.
    
    Automatically detects backend provider type and dispatches to the
    appropriate collector. Supports IBM Quantum, Aer simulators, and
    gracefully handles unknown providers.
    
    Args:
        backend: Quantum backend instance (IBM BackendV2, Aer, etc.)
        
    Returns:
        Dict[str, Any]: Hardware calibration snapshot in HardwareMetadata format.
        Returns empty metadata dict if provider is unknown or collection fails.
        
    Provider Detection Logic:
        1. Check backend.name for 'aer' (case-insensitive)
        2. Check hasattr(backend, 'target') for IBM BackendV2
        3. Check backend.__class__.__module__ for 'qiskit_aer'
        4. Check backend.__class__.__module__ for 'ibm_runtime'
        5. Fall back to empty metadata with warning
        
    Examples:
        >>> from qiskit_aer import AerSimulator
        >>> backend = AerSimulator()
        >>> metadata = collect_hardware_metadata(backend)
        >>> metadata['provider']
        'aer'
        
        >>> from qiskit_ibm_runtime import QiskitRuntimeService
        >>> service = QiskitRuntimeService()
        >>> backend = service.backend('ibm_brisbane')
        >>> metadata = collect_hardware_metadata(backend)
        >>> metadata['provider']
        'ibm'
        
    Notes:
        - Collection failures are logged but don't raise exceptions
        - Returns empty structure if provider detection fails
        - Quantinuum and AWS support deferred to Phase 7
    """
    try:
        # Get backend name for provider detection
        backend_name = str(backend)
        backend_name_attr = getattr(backend, 'name', None)
        if callable(backend_name_attr):
            try:
                backend_name = backend_name_attr()
            except Exception:
                backend_name = str(backend)
        elif isinstance(backend_name_attr, str) and backend_name_attr:
            backend_name = backend_name_attr
        backend_name_lower = backend_name.lower()
        
        # Detect Aer simulators (highest priority - fastest check)
        if 'aer' in backend_name_lower:
            logger.debug(f"Detected Aer backend: {backend_name}")
            return collect_aer_hardware_metadata(backend)
        
        # Detect IBM Quantum via Target attribute (BackendV2)
        if hasattr(backend, 'target'):
            logger.debug(f"Detected IBM BackendV2 via target: {backend_name}")
            return collect_ibm_hardware_metadata(backend)
        
        # Fallback: Check module path
        module_name = backend.__class__.__module__
        
        if 'qiskit_aer' in module_name:
            logger.debug(f"Detected Aer via module: {module_name}")
            return collect_aer_hardware_metadata(backend)
        
        if 'qiskit_ibm_runtime' in module_name or 'ibm_runtime' in module_name:
            logger.debug(f"Detected IBM via module: {module_name}")
            return collect_ibm_hardware_metadata(backend)
        
        # Unknown provider - log warning and return empty metadata
        logger.warning(
            f"Unknown backend provider: {backend_name} (module: {module_name}). "
            "Returning empty hardware metadata. Supported: IBM, Aer"
        )
        return _empty_metadata(backend_name)
        
    except Exception as e:
        # Catch all errors and return empty metadata
        logger.error(f"Hardware metadata collection failed: {e}", exc_info=True)
        backend_name = getattr(backend, 'name', 'unknown')
        return _empty_metadata(backend_name)


def _empty_metadata(backend_name: str) -> Dict[str, Any]:
    """Generate empty metadata structure for unknown/failed backends.
    
    Args:
        backend_name: Name of the backend
        
    Returns:
        Dict[str, Any]: Empty metadata structure
    """
    return {
        "backend_name": backend_name,
        "provider": "unknown",
        "calibration_timestamp": None,
        "single_qubit_gate_error": {},
        "two_qubit_gate_error": {},
        "readout_error": {},
        "t1_times": {},
        "t2_times": {},
        "extra_properties": {},
    }
