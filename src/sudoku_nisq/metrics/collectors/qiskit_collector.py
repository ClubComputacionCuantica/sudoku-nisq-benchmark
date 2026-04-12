"""Qiskit/IBM metadata collector for benchmarking metrics pipeline."""

from typing import Any, Optional
from datetime import datetime
import logging

from sudoku_nisq.metrics.collectors.base_collector import MetadataCollector
from sudoku_nisq.metrics.data_models import (
    HardwareMetadata,
    CompilationMetadata,
    ExecutionResult,
)

# Import Stage 5 collector and qiskit tools at module level for mocking support
try:
    from sudoku_nisq.metadata.collectors import collect_hardware_metadata as stage5_collect_hw
except ImportError:
    stage5_collect_hw = None  # type: ignore[assignment]

try:
    from qiskit.converters import circuit_to_dag
except ImportError:
    circuit_to_dag = None

logger = logging.getLogger(__name__)


class QiskitMetadataCollector(MetadataCollector):
    """Metadata collector for Qiskit/IBM quantum backends.
    
    Collects comprehensive metadata from IBM Quantum backends and Qiskit
    execution results. Delegates hardware calibration collection to the
    existing Stage 5 function-based collector for consistency.
    
    Handles:
        - IBM backend hardware metadata (via Stage 5 collector)
        - Circuit compilation metadata (layouts, gate counts, optimization)
        - Execution result extraction (legacy Result and SamplerV2 formats)
        - Circuit volume calculation (DAG-based layer analysis)
    
    Example:
        collector = QiskitMetadataCollector()
        
        # Collect from IBM backend
        hw_metadata = collector.collect_hardware_metadata(backend)
        comp_metadata = collector.extract_compilation_metadata(
            circuit, transpiled_circuit, backend
        )
        exec_result = collector.extract_execution_result(
            result, backend, transpiled_circuit
        )
        volume = collector.calculate_circuit_volume(transpiled_circuit)
    """
    
    def collect_hardware_metadata(self, backend: Any) -> HardwareMetadata:
        """Collect hardware metadata from IBM Quantum backend.
        
        Delegates to the Stage 5 IBM hardware collector for consistency.
        Converts the dict format to HardwareMetadata dataclass.
        
        Args:
            backend: Qiskit BackendV2 or IBMBackend instance
        
        Returns:
            HardwareMetadata with calibration data, error rates, coherence times
        
        Raises:
            ImportError: If Stage 5 collector not available
            ValueError: If backend format is unsupported
        """
        if stage5_collect_hw is None:
            logger.error("Stage 5 hardware collector not available")
            raise ImportError(
                "Cannot import collect_hardware_metadata from "
                "sudoku_nisq.metadata.collectors"
            )
        
        # Delegate to Stage 5 function-based collector
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
    
    def extract_compilation_metadata(
        self,
        circuit: Any,
        compiled_circuit: Any,
        backend: Optional[Any] = None,
    ) -> CompilationMetadata:
        """Extract metadata from Qiskit circuit transpilation.
        
        Extracts:
        - Initial/final qubit layouts from compiled_circuit._layout
        - Gate counts via compiled_circuit.count_ops()
        - Circuit depth via compiled_circuit.depth()
        - Optimization level from circuit metadata
        - SWAP counts and routing metrics
        
        Args:
            circuit: Original QuantumCircuit
            compiled_circuit: Transpiled QuantumCircuit
            backend: Optional backend for additional context
        
        Returns:
            CompilationMetadata with layouts, gate counts, depth, opt level
        """
        # Extract layouts from compiled circuit
        # Prefer public .layout over private ._layout for Qiskit 1.x/2.x compatibility
        layout_obj = getattr(compiled_circuit, "layout", None) or getattr(
            compiled_circuit, "_layout", None
        )
        
        initial_layout = None
        final_layout = None
        
        if layout_obj is not None:
            # Use TranspileLayout helper methods (more stable than raw attributes)
            if hasattr(layout_obj, "initial_virtual_layout") and callable(
                layout_obj.initial_virtual_layout
            ):
                try:
                    virt_layout = layout_obj.initial_virtual_layout()
                    if hasattr(virt_layout, "get_virtual_bits"):
                        initial_layout = {
                            str(k): v
                            for k, v in virt_layout.get_virtual_bits().items()
                        }
                except Exception as e:
                    logger.debug(f"Failed to extract initial layout: {e}")
            
            # final_index_layout() gives input-qubit -> final position mapping
            if hasattr(layout_obj, "final_index_layout") and callable(
                layout_obj.final_index_layout
            ):
                try:
                    final_layout = layout_obj.final_index_layout()
                except Exception as e:
                    logger.debug(f"Failed to extract final layout: {e}")
            
            # Fallback to legacy private attributes if helpers unavailable
            if initial_layout is None and hasattr(layout_obj, "initial_layout"):
                try:
                    initial_layout = {
                        str(vqubit): layout_obj.initial_layout[vqubit]
                        for vqubit in layout_obj.initial_layout
                    }
                except Exception:
                    pass
            
            if final_layout is None and hasattr(layout_obj, "final_layout"):
                try:
                    final_layout = layout_obj.final_layout
                except Exception:
                    pass
        
        # Get gate counts
        pre_gate_counts = circuit.count_ops() if hasattr(circuit, "count_ops") else {}
        post_gate_counts = (
            compiled_circuit.count_ops()
            if hasattr(compiled_circuit, "count_ops")
            else {}
        )
        
        # Calculate depths
        pre_depth = circuit.depth() if hasattr(circuit, "depth") else None
        post_depth = (
            compiled_circuit.depth() if hasattr(compiled_circuit, "depth") else None
        )
        
        # Extract optimization level (may be in metadata)
        optimization_level = None
        if hasattr(compiled_circuit, "metadata") and compiled_circuit.metadata:
            optimization_level = compiled_circuit.metadata.get("optimization_level")
        
        # initial_layout and final_layout already extracted above
        from typing import List
        initial_layout_list: Optional[List[int]] = None
        if isinstance(initial_layout, dict):
            # Convert dict {virtual: physical} to list format
            # Produces list where index i = physical qubit for virtual qubit i
            try:
                max_idx = max(int(k) for k in initial_layout.keys())
                initial_layout_list = []
                for i in range(max_idx + 1):
                    val = initial_layout.get(str(i))
                    if val is None:
                        # Missing virtual qubit mapping - invalid layout
                        initial_layout_list = None
                        break
                    # Convert to int (handles both int and string values)
                    initial_layout_list.append(int(val) if isinstance(val, (int, str)) and str(val).isdigit() else val)
            except (ValueError, TypeError):
                initial_layout_list = None
        elif isinstance(initial_layout, list):
            try:
                initial_layout_list = [int(x) for x in initial_layout if isinstance(x, (int, str))]
            except (ValueError, TypeError):
                initial_layout_list = None
        
        return CompilationMetadata(
            initial_layout=initial_layout_list,
            final_layout=final_layout,
            pre_transpile_gates=pre_gate_counts,
            post_transpile_gates=post_gate_counts,
            pre_transpile_depth=pre_depth,
            post_transpile_depth=post_depth,
            optimization_level=optimization_level if isinstance(optimization_level, int) else 0,
        )
    
    def extract_execution_result(
        self,
        result: Any,
        backend: Any,
        compiled_circuit: Any,
        job: Optional[Any] = None,
    ) -> ExecutionResult:
        """Extract execution metadata from Qiskit result object.
        
        Handles multiple Qiskit result formats:
        - Legacy Result.get_counts()
        - SamplerV2 PrimitiveResult with join_data() for multi-register support
        
        Args:
            result: Qiskit Result or PrimitiveResult
            backend: Backend used for execution
            compiled_circuit: Compiled QuantumCircuit
            job: Optional job object for extracting job_id (V2 primitives)
        
        Returns:
            ExecutionResult with counts, shots, timing, circuit metrics
        
        Raises:
            ValueError: If result format is unrecognized
        """
        # Extract counts (handle multiple formats)
        counts = None
        shots = None
        joined_data = None  # Store for shots extraction

        if hasattr(result, "get_counts") and callable(result.get_counts):
            # Legacy Result format or convenience method
            try:
                # Try with index (some builds require it)
                counts = result.get_counts(0)
            except (TypeError, IndexError):
                # Fallback to no-arg version
                counts = result.get_counts()
            
            # Extract shots from legacy Result
            if hasattr(result, "results") and result.results:
                shots = getattr(result.results[0], "shots", None)
        
        elif hasattr(result, "__getitem__"):
            # SamplerV2 PrimitiveResult format - use join_data() for robustness
            try:
                pub_result = result[0]
                
                # join_data() combines all classical registers deterministically
                if hasattr(pub_result, "join_data") and callable(pub_result.join_data):
                    joined_data = pub_result.join_data()  # Returns BitArray
                    counts = joined_data.get_counts()
                    
                    # Extract shots from BitArray (more reliable than metadata)
                    if hasattr(joined_data, "num_shots"):
                        shots = joined_data.num_shots
                else:
                    # Fallback: try direct register access (assumes "meas" name)
                    if hasattr(pub_result, "data") and hasattr(pub_result.data, "meas"):
                        logger.debug(
                            "Using legacy .data.meas access - consider upgrading to join_data()"
                        )
                        counts = pub_result.data.meas.get_counts()
                        
                        # Try metadata as last resort
                        if hasattr(pub_result, "metadata"):
                            shots = pub_result.metadata.get("shots")
            
            except (IndexError, AttributeError) as e:
                logger.warning(f"Failed to extract SamplerV2 counts: {e}")
        
        if counts is None:
            raise ValueError(
                f"Unrecognized result format: {type(result)}. "
                "Expected legacy Result or SamplerV2 PrimitiveResult."
            )
        
        # If shots not found, infer from counts
        if shots is None:
            shots = sum(counts.values())
        
        # Extract timing information
        execution_time = None
        timestamp = datetime.utcnow()
        
        # Extract job ID from job object (not result) for V2 compatibility
        job_id = None
        if job is not None:
            # Prefer method over property (avoids deprecation warnings)
            if hasattr(job, "job_id") and callable(job.job_id):
                try:
                    job_id = job.job_id()
                except Exception:
                    pass
            # Fallback to property if method unavailable
            if job_id is None and hasattr(job, "job_id"):
                try:
                    job_id = job.job_id
                except Exception:
                    pass
        
        # Get backend name
        backend_name_attr = getattr(backend, "name", None)
        if callable(backend_name_attr):
            try:
                backend_name = backend_name_attr()
            except Exception:
                backend_name = str(backend)
        else:
            backend_name = backend_name_attr if backend_name_attr is not None else str(backend)
        
        # Extract circuit metrics
        num_qubits = (
            compiled_circuit.num_qubits
            if hasattr(compiled_circuit, "num_qubits")
            else 0
        )
        circuit_depth = (
            compiled_circuit.depth() if hasattr(compiled_circuit, "depth") else 0
        )
        
        # Get gate counts
        gate_counts = (
            compiled_circuit.count_ops()
            if hasattr(compiled_circuit, "count_ops")
            else {}
        )
        
        # Calculate two-qubit gate count using arity (future-proof for custom gates)
        two_qubit_gates = 0
        if hasattr(compiled_circuit, "data"):
            # Count instructions by qubit arity
            try:
                two_qubit_gates = sum(
                    1 for inst, qargs, _ in compiled_circuit.data if len(qargs) == 2
                )
            except (TypeError, AttributeError):
                # Fallback to name-based counting if .data iteration fails
                two_qubit_gate_names = [
                    "cx", "cz", "cy", "ch", "swap", "iswap", "ecr",
                    "dcx", "rzz", "rxx", "ryy",
                ]
                two_qubit_gates = sum(
                    gate_counts.get(gate, 0) for gate in two_qubit_gate_names
                )
        else:
            # Fallback to name-based counting if .data unavailable
            two_qubit_gate_names = [
                "cx", "cz", "cy", "ch", "swap", "iswap", "ecr",
                "dcx", "rzz", "rxx", "ryy",
            ]
            two_qubit_gates = sum(
                gate_counts.get(gate, 0) for gate in two_qubit_gate_names
            )
        
        return ExecutionResult(
            counts=counts,
            shots=shots,
            execution_time=execution_time if execution_time is not None else 0.0,
            timestamp=timestamp,
            backend_name=backend_name,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            gate_counts=gate_counts,
            two_qubit_gates=two_qubit_gates,
            circuit_volume=None,  # Computed separately via calculate_circuit_volume
            raw_result=result,
            job_id=job_id,
            run_id=None,  # Populated by Stage 5 if available
            metadata=None,  # Can be populated with hardware snapshot link
        )
    
    def calculate_circuit_volume(self, circuit: Any) -> Optional[int]:
        """Calculate circuit volume using Qiskit DAG layer analysis.
        
        Circuit volume = sum of active gates in each layer (depth-1 slice).
        Excludes barriers, measurements, and directives.
        
        For circuits with unknown control flow (e.g., while loops with runtime
        conditions), returns None per Qiskit convention.
        
        Args:
            circuit: QuantumCircuit (preferably compiled/transpiled)
        
        Returns:
            Circuit volume (int) or None if control flow prevents calculation
        """
        if circuit_to_dag is None:
            logger.warning("Cannot import circuit_to_dag from qiskit.converters")
            return None
        
        try:
            # Convert to DAG
            dag = circuit_to_dag(circuit)
            
            # Check for control flow
            if hasattr(dag, "control_flow_op_nodes"):
                cf_nodes = list(dag.control_flow_op_nodes())
                if cf_nodes:
                    # Has control flow - cannot compute exact volume
                    logger.debug(
                        f"Circuit has {len(cf_nodes)} control flow nodes, "
                        "returning None for volume"
                    )
                    return None
            
            # Count active gates per layer
            volume = 0
            for layer in dag.layers():
                layer_dag = layer["graph"]
                # Count op nodes (excludes input/output nodes)
                layer_ops = list(layer_dag.op_nodes())
                
                # Filter out barriers and measurements
                active_ops = [
                    node
                    for node in layer_ops
                    if node.op.name not in ["barrier", "measure"]
                ]
                
                volume += len(active_ops)
            
            return volume
            
        except Exception as e:
            logger.warning(f"Failed to calculate circuit volume: {e}")
            return None
