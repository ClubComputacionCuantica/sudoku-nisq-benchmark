"""Stage 2a: Logical IR construction metadata manager.

Tracks circuit construction resources before transpilation.
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import hashlib
import json
from datetime import datetime
from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig


class LogicalIRMetadataManager(StageMetadataManager):
    """Manages Stage 2a: Logical IR Construction metadata.
    
    Tracks logical circuit resources:
    - Gate counts (n_gates, depth, n_qubits)
    - SDK type (pytket/qiskit/braket)
    - Circuit hash (deterministic identifier)
    - Decomposition options (decompose_cnz, etc.)
    
    Auto-extracts resources from circuit objects (PyTKET Circuit, Qiskit QuantumCircuit, Braket Circuit).
    
    Storage: {puzzle_hash}/stage_2a_logical_ir.json
    Structure: {solver_name: {encoding: {resources, circuit_hash, ...}}}
    """
    
    @property
    def stage_number(self) -> int:
        return 2  # Stage 2a
    
    def __init__(self, cache_base: Path, puzzle_hash: str):
        """Initialize logical IR metadata manager.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        :param puzzle_hash: Puzzle identifier (determines storage subdirectory)
        """
        self.cache_base = Path(cache_base)
        self.puzzle_hash = puzzle_hash
        self._storage_path = self.cache_base / puzzle_hash / MetadataConfig.STAGE_2A_LOGICAL_IR
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def _detect_sdk(self, circuit: Any) -> str:
        """Detect SDK type from circuit object.
        
        :param circuit: Circuit object (PyTKET Circuit, Qiskit QuantumCircuit, Braket Circuit)
        :returns: SDK name ("pytket", "qiskit", or "braket")
        :raises ValueError: If circuit type is unknown
        """
        circuit_type = type(circuit).__name__
        module_name = type(circuit).__module__
        
        if "pytket" in module_name:
            return "pytket"
        elif "qiskit" in module_name:
            return "qiskit"
        elif "braket" in module_name:
            return "braket"
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type} from module {module_name}")
    
    def _extract_resources_pytket(self, circuit: Any) -> Dict[str, int]:
        """Extract resources from PyTKET circuit.
        
        :param circuit: PyTKET Circuit instance
        :returns: Dict with n_qubits, n_gates, depth
        """
        from pytket.utils import gate_counts
        
        gate_count_dict = gate_counts(circuit)
        n_gates = sum(gate_count_dict.values())
        
        return {
            "n_qubits": circuit.n_qubits,
            "n_gates": n_gates,
            "depth": circuit.depth(),
            "gate_breakdown": dict(gate_count_dict),  # Store detailed gate counts
        }
    
    def _extract_resources_qiskit(self, circuit: Any) -> Dict[str, int]:
        """Extract resources from Qiskit circuit.
        
        :param circuit: Qiskit QuantumCircuit instance
        :returns: Dict with n_qubits, n_gates, depth
        """
        # Count gates (excluding barriers and measurements)
        ops_count = circuit.count_ops()
        n_gates = sum(v for k, v in ops_count.items() if k not in ['barrier', 'measure'])
        
        return {
            "n_qubits": circuit.num_qubits,
            "n_gates": n_gates,
            "depth": circuit.depth(),
            "gate_breakdown": dict(ops_count),
        }
    
    def _extract_resources_braket(self, circuit: Any) -> Dict[str, int]:
        """Extract resources from Braket circuit.
        
        :param circuit: Braket Circuit instance
        :returns: Dict with n_qubits, n_gates, depth
        """
        # Braket circuit resource extraction
        n_gates = sum(1 for _ in circuit.instructions)
        
        return {
            "n_qubits": circuit.qubit_count,
            "n_gates": n_gates,
            "depth": circuit.depth,
        }
    
    def _extract_resources(self, circuit: Any, sdk_type: str) -> Dict[str, Any]:
        """Extract resources from circuit based on SDK type.
        
        :param circuit: Circuit object
        :param sdk_type: SDK name ("pytket", "qiskit", "braket")
        :returns: Dict with resource metrics
        :raises ValueError: If SDK type is unknown
        """
        if sdk_type == "pytket":
            return self._extract_resources_pytket(circuit)
        elif sdk_type == "qiskit":
            return self._extract_resources_qiskit(circuit)
        elif sdk_type == "braket":
            return self._extract_resources_braket(circuit)
        else:
            raise ValueError(f"Unknown SDK type: {sdk_type}")
    
    def _compute_circuit_hash(self, circuit: Any, sdk_type: str) -> str:
        """Compute deterministic hash for circuit.
        
        Uses circuit dict/JSON representation for deterministic hashing.
        
        :param circuit: Circuit object
        :param sdk_type: SDK name ("pytket", "qiskit", "braket")
        :returns: Hex digest of circuit hash
        """
        try:
            if sdk_type == "pytket":
                # PyTKET: Use circuit dict representation
                circuit_dict = circuit.to_dict()
                circuit_str = json.dumps(circuit_dict, sort_keys=True)
                return hashlib.sha256(circuit_str.encode()).hexdigest()
            
            elif sdk_type == "qiskit":
                # Qiskit: Use JSON representation
                from qiskit.qpy import dump
                import io
                buffer = io.BytesIO()
                dump(circuit, buffer)
                return hashlib.sha256(buffer.getvalue()).hexdigest()
            
            elif sdk_type == "braket":
                # Braket: Use circuit string representation
                circuit_str = str(circuit)
                return hashlib.sha256(circuit_str.encode()).hexdigest()
            
            else:
                raise ValueError(f"Unknown SDK type: {sdk_type}")
                
        except Exception as e:
            # Fallback: Hash based on resource metrics
            import warnings
            warnings.warn(
                f"Failed to compute circuit-based hash ({e}). Using resource-based hash instead.",
                RuntimeWarning
            )
            resources = self._extract_resources(circuit, sdk_type)
            resource_str = json.dumps(resources, sort_keys=True)
            return hashlib.sha256(resource_str.encode()).hexdigest()
    
    def record(self, **kwargs) -> str:
        """Record logical circuit metadata with auto-extracted resources.
        
        :param kwargs: Circuit-specific parameters:
            - solver_name (str): Name of solver (e.g., "ExactCoverQuantumSolver")
            - encoding (str): Encoding type (e.g., "simple", "pattern")
            - circuit (Any): Circuit object (auto-detects SDK)
            - sdk_type (str, optional): Explicit SDK override
            - decompose_cnz (bool, optional): Decomposition setting
            - additional solver options
        :returns: circuit_hash (for linking to Stage 3)
        """
        solver_name = kwargs.get("solver_name")
        encoding = kwargs.get("encoding")
        circuit = kwargs.get("circuit")
        
        if not all([solver_name, encoding, circuit]):
            raise ValueError("record() requires solver_name, encoding, and circuit")
        
        # Auto-detect or use explicit SDK
        sdk_type = kwargs.get("sdk_type") or self._detect_sdk(circuit)
        
        # Extract resources
        resources = self._extract_resources(circuit, sdk_type)
        
        # Compute circuit hash
        circuit_hash = self._compute_circuit_hash(circuit, sdk_type)
        
        # Build metadata record
        record = {
            "circuit_hash": circuit_hash,
            "sdk_type": sdk_type,
            "resources": resources,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        # Add optional solver configuration
        for key in ["decompose_cnz", "track_memory", "solver_options"]:
            if key in kwargs:
                record[key] = kwargs[key]
        
        # Load existing data
        data = self._load_json()
        
        # Update nested structure: {solver_name: {encoding: {...}}}
        solver_key = solver_name if solver_name is not None else "unknown"
        encoding_key = encoding if encoding is not None else "unknown"
        if solver_key not in data:
            data[solver_key] = {}
        data[solver_key][encoding_key] = record
        
        # Save atomically
        self._save_json(data)
        
        return circuit_hash
    
    def query(self, **filters) -> Union[List[Dict], Dict, None]:
        """Query logical IR records by solver/encoding.
        
        :param filters: Filter criteria:
            - solver_name (str, optional): Filter by solver
            - encoding (str, optional): Filter by encoding
            - circuit_hash (str, optional): Filter by circuit hash
        :returns: Matching metadata dict(s) or None
        """
        data = self._load_json()
        
        solver_name = filters.get("solver_name")
        encoding = filters.get("encoding")
        circuit_hash = filters.get("circuit_hash")
        
        # No filters: return all
        if not any([solver_name, encoding, circuit_hash]):
            return data
        
        # Filter by solver_name
        if solver_name and solver_name not in data:
            return None
        
        if solver_name and encoding:
            # Specific solver + encoding
            if solver_name in data and encoding in data[solver_name]:
                return data[solver_name][encoding]
            return None
        
        if solver_name:
            # All encodings for solver
            return data.get(solver_name)
        
        # Filter by circuit_hash (scan all records)
        if circuit_hash:
            for solver, encodings in data.items():
                for enc, record in encodings.items():
                    if record.get("circuit_hash") == circuit_hash:
                        return record
            return None
        
        return data
