"""Aer simulator metadata collector for benchmarking metrics pipeline."""

from typing import Any
import logging

from sudoku_nisq.metrics.collectors.qiskit_collector import QiskitMetadataCollector
from sudoku_nisq.metrics.data_models import HardwareMetadata

# Import Stage 5 collector at module level for mocking support
try:
    from sudoku_nisq.metadata.collectors import collect_hardware_metadata as stage5_collect_hw
except ImportError:
    stage5_collect_hw = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class AerMetadataCollector(QiskitMetadataCollector):
    """Metadata collector for Qiskit Aer simulator backends.
    
    Extends QiskitMetadataCollector to handle Aer-specific metadata collection.
    Inherits compilation, execution, and volume calculation from parent class
    since Aer uses standard Qiskit types.
    
    Delegates hardware metadata collection to the Stage 5 Aer collector for
    simulator configuration details (method, noise model, device type, etc.).
    
    Example:
        collector = AerMetadataCollector()
        
        # Collect from Aer backend
        hw_metadata = collector.collect_hardware_metadata(aer_backend)
        comp_metadata = collector.extract_compilation_metadata(
            circuit, transpiled_circuit, aer_backend
        )
        exec_result = collector.extract_execution_result(
            result, aer_backend, transpiled_circuit
        )
        volume = collector.calculate_circuit_volume(transpiled_circuit)
    """
    
    def collect_hardware_metadata(self, backend: Any) -> HardwareMetadata:
        """Collect hardware metadata from Aer simulator.
        
        Delegates to Stage 5 Aer collector which captures simulator-specific
        configuration (statevector/density_matrix/automatic method, noise model,
        device type, precision, etc.).
        
        Args:
            backend: AerSimulator instance
        
        Returns:
            HardwareMetadata with simulator configuration details
        
        Raises:
            ImportError: If Stage 5 Aer collector not available
            ValueError: If backend is not an Aer simulator
        """
        if stage5_collect_hw is None:
            logger.error("Stage 5 hardware collector not available")
            raise ImportError(
                "Cannot import collect_hardware_metadata from "
                "sudoku_nisq.metadata.collectors"
            )
        
        # Delegate to Stage 5 dispatcher (will route to Aer collector)
        hw_dict = stage5_collect_hw(backend)

        def _get_hw_value(*keys: str) -> Any:
            for key in keys:
                if key in hw_dict:
                    return hw_dict.get(key)
            return None

        # Convert dict to dataclass
        return HardwareMetadata(
            backend_name=hw_dict.get("backend_name") or "",
            provider=hw_dict.get("provider") or "",
            calibration_timestamp=hw_dict.get("calibration_timestamp"),
            single_qubit_gate_error=_get_hw_value(
                "single_qubit_gate_error", "single_qubit_gate_errors"
            ),
            two_qubit_gate_error=_get_hw_value(
                "two_qubit_gate_error", "two_qubit_gate_errors"
            ),
            readout_error=_get_hw_value("readout_error", "readout_errors"),
            t1_times=_get_hw_value("t1_times"),
            t2_times=_get_hw_value("t2_times"),
            extra_properties=hw_dict.get("extra_properties", {}),
        )
    
    # Inherit extract_compilation_metadata from QiskitMetadataCollector
    # Inherit extract_execution_result from QiskitMetadataCollector
    # Inherit calculate_circuit_volume from QiskitMetadataCollector
