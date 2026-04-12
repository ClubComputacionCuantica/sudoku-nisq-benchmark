"""IBM Quantum hardware metadata collector.

Extracts calibration data from IBM BackendV2 instances using the Target API.
Collects T1/T2 coherence times, gate error rates, readout errors, and
calibration timestamps for execution metadata recording.

TODO (optional enhancement): Consider implementing:
    - Per-qubit calibration timestamps via backend.properties().qubit_property(i)
    - Per-(gate,qargs) error/duration tracking (full InstructionProperties)
    - While maintaining current simplified aggregation for summary metrics
"""

from typing import Dict, Any


def collect_ibm_hardware_metadata(backend: Any) -> Dict[str, Any]:
    """Collect hardware calibration data from IBM Quantum backend.
    
    Extracts calibration metadata from BackendV2.target and properties() including:
    - Calibration timestamp (from backend.properties().last_update_date)
    - Single-qubit gate errors
    - Two-qubit gate errors  
    - Readout errors
    - T1/T2 coherence times
    - Additional backend properties
    
    Args:
        backend: IBM BackendV2 instance with Target
        
    Returns:
        Dict[str, Any]: Hardware calibration snapshot in HardwareMetadata format:
            {
                "backend_name": str,
                "provider": "ibm",
                "calibration_timestamp": ISO datetime string or None,
                "single_qubit_gate_error": {"qubit_idx": error_rate},  # String keys
                "two_qubit_gate_error": {"q1,q2": error_rate},  # String keys
                "readout_error": {"qubit_idx": error_rate},  # String keys
                "t1_times": {"qubit_idx": microseconds},  # String keys
                "t2_times": {"qubit_idx": microseconds},  # String keys
                "extra_properties": {...}
            }
            
    Notes:
        - Returns empty dicts for error/timing fields if target is None
        - Gracefully handles missing calibration data
        - Compatible with Qiskit 1.0+ BackendV2 + Target architecture
        - All dictionary keys are strings for JSON serialization compatibility
    """
    metadata: Dict[str, Any] = {
        "backend_name": backend.name,
        "provider": "ibm",
        "calibration_timestamp": None,
        "single_qubit_gate_error": {},
        "two_qubit_gate_error": {},
        "readout_error": {},
        "t1_times": {},
        "t2_times": {},
        "extra_properties": {},
    }
    
    # Check if backend has Target (BackendV2)
    if not hasattr(backend, 'target') or backend.target is None:
        # No target available - return empty calibration data
        return metadata
    
    target = backend.target
    
    # Extract calibration timestamp from backend properties
    try:
        props = backend.properties()
        if props and getattr(props, "last_update_date", None):
            metadata["calibration_timestamp"] = props.last_update_date.isoformat()
    except Exception:
        pass  # Properties may not be available
    
    # Store target dt (in seconds) and online_date as extra info
    try:
        if hasattr(target, 'dt') and target.dt is not None:
            metadata["extra_properties"]["dt"] = target.dt
        if hasattr(backend, 'online_date') and backend.online_date:
            metadata["extra_properties"]["online_date"] = backend.online_date.isoformat()
    except Exception:
        pass
    
    # Collect qubit properties (T1, T2, readout error)
    try:
        num_qubits = target.num_qubits if hasattr(target, 'num_qubits') else 0
        
        for qubit_idx in range(num_qubits):
            # Use string keys for JSON serialization compatibility
            qubit_str = str(qubit_idx)
            
            # T1 coherence time
            try:
                t1_property = target.qubit_properties[qubit_idx].t1
                if t1_property is not None:
                    metadata["t1_times"][qubit_str] = t1_property * 1e6  # Convert to microseconds
            except (AttributeError, IndexError, KeyError):
                pass
            
            # T2 coherence time
            try:
                t2_property = target.qubit_properties[qubit_idx].t2
                if t2_property is not None:
                    metadata["t2_times"][qubit_str] = t2_property * 1e6  # Convert to microseconds
            except (AttributeError, IndexError, KeyError):
                pass
            
            # Readout error (from measurement instruction)
            try:
                measure_props = target['measure'][(qubit_idx,)]
                if measure_props and hasattr(measure_props, 'error'):
                    metadata["readout_error"][qubit_str] = measure_props.error
            except (KeyError, AttributeError, TypeError):
                pass
    except Exception:
        pass  # Continue even if qubit properties fail
    
    # Collect gate error rates
    try:
        # Single-qubit gates (typically 'x', 'sx', 'rz')
        for gate_name in ['x', 'sx', 'rz', 'id']:
            if gate_name not in target:
                continue
            for qargs in target[gate_name]:
                if len(qargs) == 1:  # Single-qubit gate
                    qubit_str = str(qargs[0])  # String key for JSON
                    props = target[gate_name][qargs]
                    if props and hasattr(props, 'error') and props.error is not None:
                        # Store per-gate-per-qubit; take minimum error rate
                        current_error = metadata["single_qubit_gate_error"].get(qubit_str, float('inf'))
                        metadata["single_qubit_gate_error"][qubit_str] = min(current_error, props.error)
        
        # Two-qubit gates (typically 'cx', 'ecr')
        for gate_name in ['cx', 'ecr', 'cz']:
            if gate_name not in target:
                continue
            for qargs in target[gate_name]:
                if len(qargs) == 2:  # Two-qubit gate
                    # Convert tuple to string for JSON serialization
                    qubit_pair_str = f"{qargs[0]},{qargs[1]}"
                    props = target[gate_name][qargs]
                    if props and hasattr(props, 'error') and props.error is not None:
                        # Store best error rate for this qubit pair
                        current_error = metadata["two_qubit_gate_error"].get(qubit_pair_str, float('inf'))
                        metadata["two_qubit_gate_error"][qubit_pair_str] = min(current_error, props.error)
    except Exception:
        pass  # Continue even if gate error extraction fails
    
    # Add backend-specific extra properties
    try:
        if hasattr(backend, 'max_circuits'):
            metadata["extra_properties"]["max_circuits"] = backend.max_circuits
        if hasattr(backend, 'max_shots'):
            metadata["extra_properties"]["max_shots"] = backend.max_shots
        if hasattr(backend, 'backend_version'):  # Semantic version string (X.Y.Z)
            metadata["extra_properties"]["backend_version"] = backend.backend_version
    except Exception:
        pass
    
    return metadata
