# mypy: ignore-errors
import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any
from pytket import Circuit, OpType
from pytket.utils import gate_counts
from pytket.passes import FlattenRegisters
from qiskit import QuantumCircuit

from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
from sudoku_nisq.metadata_manager import MetadataManager

class QuantumSolver(ABC):
    """Abstract base class that provides infrastructure for quantum Sudoku solvers.
    
    This class does not implement any specific solving algorithm. Instead, it provides
    a common framework and shared functionality that concrete solver implementations
    can inherit and build upon. The actual quantum algorithms for solving Sudoku
    puzzles are implemented in subclasses such as ExactCoverQuantumSolver and 
    GraphColoringQuantumSolver.
    
    Key Infrastructure Services:
    - Circuit caching and persistence to avoid rebuilding expensive circuits
    - Metadata management for tracking solver performance and resource usage  
    - Backend integration for transpilation and execution across different quantum platforms
    - Resource estimation and analysis capabilities
    - Common visualization and plotting utilities
    
    Subclasses must implement the abstract methods _build_circuit() and resource_estimation()
    to define their specific quantum algorithms and resource requirements.
    """

    def __init__(
        self, 
        puzzle: SudokuPuzzle | None = None,
        metadata_manager: MetadataManager | None = None,
        encoding: str | None = None, 
        store_transpiled: bool = True,
    ):
        """Initialize the QuantumSolver base class infrastructure.
        
        Sets up the common infrastructure that concrete solver subclasses will use,
        including puzzle context, metadata management, and caching configuration.
        This base class does not implement any solving algorithm itself.

        Args:
            puzzle (SudokuPuzzle, optional): The Sudoku puzzle instance to solve. 
                Can be None for generic exact cover problems.
            metadata_manager (MetadataManager, optional): Instance for managing circuit metadata,
                caching, and performance tracking. Can be None for minimal usage.
            encoding (str, optional): Encoding strategy name for the quantum algorithm.
                If None, defaults to "default".
            store_transpiled (bool, optional): Whether to save transpiled circuits to
                disk for caching. Defaults to True for better performance on repeated runs.
        """
        
        # Sudoku integration for puzzle-specific caching (optional for generic problems)
        self.puzzle = puzzle
        self._metadata = metadata_manager
        self.encoding = encoding or "default"  # Default encoding if not specified
        self.store_transpiled = store_transpiled
        
        # Circuit management - now SDK-agnostic
        self.main_circuit: Any | None = None
        
        # Cache base derived from metadata manager (if provided)
        self.cache_base = self._metadata.cache_base if self._metadata else Path(".quantum_solver_cache")
        
    @abstractmethod
    def _build_sdk_circuit(self, sdk_type: str) -> Any:
        """Build circuit using specific SDK.
        
        This method allows subclasses to separate their circuit construction
        logic by SDK while maintaining a clean interface. The base class
        handles SDK detection and delegates to this method.
        
        Args:
            sdk_type (str): The target SDK ("pytket", "qiskit", "braket")
            
        Returns:
            Any: Circuit object in the requested SDK format
        """
        pass
    
    def _build_circuit(self, backend: Any = None, sdk: str | None = None) -> Any:
        """Construct quantum circuit with automatic or explicit SDK selection.
        
        This method handles SDK detection and delegates to the abstract
        _build_sdk_circuit method that subclasses must implement.
        
        Args:
            backend (Any, optional): Backend instance for automatic SDK detection.
            sdk (str | None, optional): Explicit SDK selection ('pytket', 'qiskit', 'braket').
                If provided, overrides automatic backend-based detection.
                
        Returns:
            Any: Quantum circuit in the format of the selected SDK.
            
        Raises:
            ValueError: If sdk parameter specifies an invalid SDK name.
        """
        if sdk is not None:
            # Explicit SDK selection - validate the value
            valid_sdks = ('pytket', 'qiskit', 'braket')
            if sdk not in valid_sdks:
                raise ValueError(
                    f"Invalid SDK '{sdk}'. Must be one of {valid_sdks}"
                )
            sdk_type = sdk
        else:
            # Automatic detection based on backend
            sdk_type = self._detect_backend_sdk(backend)
        
        return self._build_sdk_circuit(sdk_type)
    
    @abstractmethod
    def resource_estimation(self):
        """Estimate and return resource requirements for executing the main circuit.

        Calculates quantum resource requirements including qubit count, gate count,
        and potentially circuit depth and other metrics for the main circuit. This
        method is algorithm-specific and implemented by subclasses.

        Returns:
            dict: Dictionary of resource metrics. The required keys are:
                - n_qubits (int): Number of qubits required
                - n_gates (int): Total number of quantum gates
            Optional keys may include (when available):
                - depth (int | None): Circuit depth. May be omitted or None if not estimated
                - n_mcx_gates (int): Multi-controlled-X gate count
                - Any solver-specific metrics
        """
        pass
    
    @abstractmethod
    def _transpile_pytket(self, backend: Any, opt_level: int) -> Circuit:
        """Transpile circuit using PyTKET backend.
        
        Args:
            backend: PyTKET backend instance with get_compiled_circuit method
            opt_level: Optimization level for transpilation (0-2 typically)
            
        Returns:
            Circuit: Transpiled circuit in PyTKET format
        """
        pass
    
    @abstractmethod
    def _transpile_qiskit(self, backend: Any, opt_level: int) -> Any:
        """Transpile circuit using Qiskit native transpiler.
        
        Args:
            backend: Qiskit backend instance
            opt_level: Optimization level for transpilation (0-3 typically)
            
        Returns:
            QuantumCircuit: Transpiled circuit in Qiskit format
        """
        pass
    
    @abstractmethod
    def _transpile_braket(self, backend: Any, opt_level: int) -> None:
        """Braket doesn't support client-side transpilation.
        
        AWS Braket performs transpilation server-side. Access the transpiled
        circuit after execution via task.result().
        
        Args:
            backend: Braket backend instance (unused)
            opt_level: Optimization level (unused)
            
        Raises:
            NotImplementedError: Always, as Braket doesn't support pre-transpilation
        """
        pass
    
    @property
    def puzzle_hash(self) -> str:
        """str: Unique hash identifier for the current problem (Sudoku puzzle or generic exact cover)."""
        if self.puzzle is not None:
            return self.puzzle.get_hash()
        elif hasattr(self, '_problem_hash'):
            # For generic exact cover problems, use stored hash
            return self._problem_hash
        else:
            # Fallback: generate hash from solver parameters
            import hashlib
            content = f"{self.solver_name}_{self.encoding}"
            if hasattr(self, 'universe') and hasattr(self, 'subsets'):
                # Hash the universe and subsets for generic problems
                from sudoku_nisq.exact_cover_problem import ExactCoverProblem
                temp_problem = ExactCoverProblem(
                    universe=self.universe,
                    subsets=self.subsets
                )
                return temp_problem.get_hash()
            return hashlib.sha256(content.encode()).hexdigest()

    @property
    def solver_name(self) -> str:
        """str: Class name of the quantum solver implementation."""
        return type(self).__name__
    
    @property
    def board(self) -> list[list[int]]:
        """list[list[int]]: The Sudoku puzzle board as a 2D list of integers."""
        return self.puzzle.board

    @property
    def size(self) -> int:
        """int: Size of the Sudoku board (board_size x board_size)."""
        return self.puzzle.board_size

    @property
    def num_missing_cells(self) -> int:
        """int: Number of empty cells in the Sudoku puzzle."""
        return self.puzzle.num_missing_cells

    def get_gate_counts(self) -> dict | None:
        """Get dictionary of gate counts from circuit construction.
        
        Returns gate counts generated during circuit building, or None if the
        circuit hasn't been built yet or the solver doesn't support gate counting.
        The dictionary uses clear naming: 'H', 'X', 'CX', 'CCX', 'C3X', etc.
        
        Returns:
            dict | None: Gate counts dictionary or None if unavailable.
        """
        return getattr(self, 'gate_counts', None)
    
    def get_memory_usage(self) -> dict | None:
        """Get memory usage statistics from circuit construction (advanced/dev feature).
        
        Returns memory tracking data collected during circuit building if memory
        tracking was enabled. Returns None if the circuit hasn't been built yet,
        memory tracking was disabled, or the feature is unavailable.
        
        Note: Memory tracking is disabled by default. Enable with track_memory=True
        when setting the solver for development and profiling purposes.
        
        Returns:
            dict | None: Memory statistics dictionary with keys:
                - 'initial_mb': Starting memory usage
                - 'current_mb': Current memory usage
                - 'peak_mb': Peak memory across all snapshots
                - 'delta_mb': Memory increase from initial
                - 'snapshots': All recorded memory snapshots

            Returns None if tracking was disabled or unavailable.
        """
        return getattr(self, 'memory_usage', None)

    @property
    def metadata_path(self) -> Path:
        """Path: File path to the puzzle's metadata JSON file.
        
        Format: .quantum_solver_cache/{puzzle_hash}/metadata.json
        """
        return self.cache_base / self.puzzle_hash / "metadata.json"

    @property
    def cache_root(self) -> Path:
        """Path: Root directory for this solver's cached files.
        
        Format: .quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}
        """
        return self.cache_base / self.puzzle_hash / self.solver_name / self.encoding

    @property
    def main_circuit_path(self) -> Path:
        """Path: File path for the main quantum circuit cache.
        
        Format: .quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}/main_circuit.json
        """
        return self.cache_root / "main_circuit.json"
    
    def transpiled_circuit_path(self, backend_alias: str, opt_level: int, sdk_type: str | None = None) -> Path:
        """Get the file path for a transpiled circuit cache.
        
        Constructs the cache path for a circuit transpiled for a specific backend
        and optimization level. Optionally includes SDK type to prevent cache conflicts.
        
        Args:
            backend_alias (str): Alias of the target backend.
            opt_level (int): Optimization level used for transpilation.
            sdk_type (str | None): SDK type for cache separation ("pytket", "qiskit", "braket").
                If None, uses SDK-agnostic path for backward compatibility.
            
        Returns:
            Path: File path for the transpiled circuit cache.
                Format (with SDK): .quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}/
                        {backend_alias}/opt{opt_level}_{sdk_type}_circuit.json
                Format (without SDK): .quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}/
                        {backend_alias}/opt{opt_level}_circuit.json
        """
        if sdk_type:
            return self.cache_root / backend_alias / f"opt{opt_level}_{sdk_type}_circuit.json"
        else:
            return self.cache_root / backend_alias / f"opt{opt_level}_circuit.json"

    def build_main_circuit(self, backend: Any = None, sdk: str | None = None, force_overwrite: bool = False, flatten: bool = True) -> Any:
        """Load or build the main quantum circuit for the solving algorithm.
        
        Manages circuit caching by loading from disk if available, or building a new
        circuit using _build_circuit() and saving it for future use. Automatically
        handles register flattening for backend compatibility and updates metadata
        with circuit resource information.
        
        Args:
            backend (Any, optional): Backend instance that determines which SDK to use.
                If None (default), uses PyTKET for general-purpose circuit building.
                If provided, uses the backend's provider-specific SDK (IBM→Qiskit, etc.).
            sdk (str | None, optional): Explicitly select which SDK to use ('pytket', 'qiskit', 'braket').
                If provided, overrides automatic backend-based SDK detection.
                Enables SDK comparison and testing without backend initialization.
            force_overwrite (bool, optional): If True, rebuilds the circuit even if
                a cached version exists. Defaults to False.
            flatten (bool, optional): If True, applies register flattening for
                compatibility with generic backends. Defaults to True.
                
        Returns:
            Any: The main quantum circuit ready for transpilation and execution,
                in the format appropriate for the selected SDK (pytket.Circuit, qiskit.QuantumCircuit, etc.).
            
        Raises:
            ValueError: If sdk parameter specifies an invalid SDK name.
            
        Side Effects:
            - Sets self.main_circuit to the built/loaded circuit
            - Saves circuit to disk cache if newly built (pytket format for compatibility)
            - Updates metadata with circuit resource metrics (including SDK type)
            - Persists metadata to disk
        """
        path = self.main_circuit_path
        
        # Determine target SDK (explicit or auto-detected)
        target_sdk = sdk if sdk is not None else self._detect_backend_sdk(backend)
        
        # Only use cache if: (1) cache exists, (2) not forcing rebuild, and (3) no explicit SDK override
        # When explicit SDK is provided, always rebuild to ensure correct SDK format
        if path.exists() and not force_overwrite and sdk is None:
            circ = self.load_circuit(path)
        else:
            # Delegate to the subclass with backend context and explicit SDK selection
            circ = self._build_circuit(backend, sdk=sdk)
            
            # For caching and metadata, convert to pytket format if needed
            cache_circuit = self._ensure_pytket_format(circ, sdk_type=target_sdk)
            
            # Flatten registers for compatibility with generic backends (pytket specific)
            if flatten and hasattr(cache_circuit, 'n_qubits'):  # pytket circuit
                FlattenRegisters().apply(cache_circuit)
            
            # Persist cache only if in pytket format (has to_dict)
            if hasattr(cache_circuit, "to_dict"):
                self.save_circuit(cache_circuit, path)
            
            # Record main circuit resources using available format
            main_res = self._get_circuit_resources(cache_circuit)
            
            # Track SDK type in metadata (explicit selection or auto-detected)
            self._metadata.set_main_circuit_resources(self.solver_name, self.encoding, main_res, sdk_type=target_sdk)
            self._metadata.save()

        self.main_circuit = circ
        return circ

    def _detect_backend_sdk(self, backend: Any) -> str:
        """Detect which SDK the backend uses based on its interface.
        
        When no backend is provided (backend=None), defaults to PyTKET as the
        most versatile SDK with broad compatibility.
        
        Now properly detects native Qiskit runtime backends from qiskit_ibm_runtime
        and distinguishes them from PyTKET's IBMQBackend wrapper.
        
        Args:
            backend: The backend instance to analyze, or None for default
            
        Returns:
            str: SDK name ("pytket", "qiskit", "braket")
        """
        if backend is None:
            return "pytket"  # Default: PyTKET for general-purpose usage
        
        # Check backend module to identify native qiskit_ibm_runtime backends
        backend_module = getattr(backend, '__module__', '')
        
        # Native Qiskit runtime backend (highest priority check)
        if 'qiskit_ibm_runtime' in backend_module:
            return "qiskit"
        
        # Check for braket backend characteristics BEFORE Qiskit checks
        # (because Mock objects return True for hasattr on any attribute)
        # Check both module and type string for braket
        if 'braket' in backend_module.lower() or 'braket' in str(type(backend)).lower():
            return "braket"
        
        # Check for pytket backend characteristics (includes IBMQBackend wrapper)
        if hasattr(backend, 'get_compiled_circuit') and hasattr(backend, 'process_circuit'):
            return "pytket"
        
        # Check for Qiskit backend characteristics (AerSimulator, legacy IBMQ, etc.)
        # Use hasattr for 'target' which is present in modern Qiskit backends
        if hasattr(backend, 'target') or hasattr(backend, 'configuration'):
            return "qiskit"
        elif 'qiskit' in str(type(backend)).lower():
            return "qiskit"
            
        else:
            # Default to pytket for unknown backends
            return "pytket"

    def _ensure_pytket_format(self, circuit: Any, sdk_type: str | None = None, backend: Any = None) -> Circuit:
        """Convert circuit to pytket format for caching and metadata consistency.
        
        Args:
            circuit: Circuit in any SDK format
            sdk_type: Explicit SDK type ('pytket', 'qiskit', 'braket')
            backend: Backend context (optional, used if sdk_type not provided)
            
        Returns:
            Circuit: Circuit converted to pytket format
        """
        # If already pytket, return as-is
        if hasattr(circuit, 'n_qubits') and hasattr(circuit, 'n_gates'):
            return circuit
        
        # Determine SDK type
        if sdk_type is None:
            sdk_type = self._detect_backend_sdk(backend)
        
        if sdk_type == "qiskit":
            # Convert qiskit to pytket
            try:
                from pytket.extensions.qiskit import qiskit_to_tk
                return qiskit_to_tk(circuit)
            except Exception:
                # If conversion tools are unavailable, skip conversion and cache as-is
                return circuit
            
        elif sdk_type == "braket":
            # Convert braket to pytket (would need appropriate converter)
            # For now, assume direct conversion or raise error
            raise NotImplementedError("Braket to pytket conversion not implemented yet")
            
        else:
            # Assume it's already compatible or return as-is
            return circuit

    def _get_circuit_resources(self, circuit: Any) -> dict:
        """Get circuit resource metrics in a uniform format.
        
        Args:
            circuit: Circuit in any SDK format
            
        Returns:
            dict: Resource metrics (n_qubits, n_gates, gate_counts, etc.)
        """
        if hasattr(circuit, 'n_qubits'):  # pytket format
            resources = {
                "n_qubits": circuit.n_qubits,
                "n_gates": circuit.n_gates,
                "n_mcx_gates": self.count_mcx_gates(circuit),
                "depth": circuit.depth(),
            }
        elif hasattr(circuit, 'num_qubits'):  # qiskit format
            gate_count = sum(circuit.count_ops().values()) if hasattr(circuit, 'count_ops') else 0
            mcx_count = circuit.count_ops().get('mcx', 0) if hasattr(circuit, 'count_ops') else 0
            resources = {
                "n_qubits": circuit.num_qubits,
                "n_gates": gate_count,
                "n_mcx_gates": mcx_count,
                "depth": circuit.depth() if hasattr(circuit, 'depth') else 0,
            }
        else:
            # Default/unknown format
            resources = {
                "n_qubits": 0,
                "n_gates": 0,
                "n_mcx_gates": 0,
                "depth": 0,
            }
        
        # Add gate_counts if available from the solver
        gate_counts = getattr(self, 'gate_counts', None)
        if gate_counts is not None:
            resources["gate_counts"] = gate_counts
        
        # Add memory_usage if available from the solver
        memory_usage = getattr(self, 'memory_usage', None)
        if memory_usage is not None:
            resources["memory_usage"] = memory_usage
        
        return resources

    def draw_circuit(self, circuit: Any | None = None, **kwargs) -> None:
        """Draw a visual representation of the quantum circuit.
        
        Renders the quantum circuit using pytket's Jupyter display functionality.
        If no circuit is provided, uses the main circuit that must be built first.

        Args:
            circuit (Circuit, optional): The pytket Circuit object to visualize.
                If None, uses self.main_circuit. Defaults to None.
            **kwargs: Additional keyword arguments passed to the drawing function.

        Returns:
            The rendered circuit visualization for Jupyter notebook display.
            
        Raises:
            ValueError: If no circuit is provided and self.main_circuit is None.
        """
        from pytket.circuit.display import render_circuit_jupyter as draw
        if circuit is None:
            if self.main_circuit is None:
                raise ValueError("No main circuit available. Please build it first.")
            circuit = self.main_circuit
        
        return draw(circuit)

    def save_circuit(self, circuit: Circuit, path: Path) -> None:
        """Serialize a pytket Circuit to JSON format on disk.
        
        Converts the quantum circuit to a JSON-serializable dictionary format
        and writes it to the specified file path. Automatically creates parent
        directories if they don't exist.

        Args:
            circuit (Circuit): The pytket Circuit object to serialize and save.
            path (Path): Full file path where the JSON should be written.
                Parent directories will be created automatically.
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize to JSON-compatible dict
        circ_dict = circuit.to_dict()
        # Write out
        with path.open("w") as f:
            json.dump(circ_dict, f)

    def load_circuit(self, path: Path) -> Circuit:
        """Load a pytket Circuit from JSON format on disk.
        
        Deserializes a quantum circuit from a JSON file that was previously
        saved using save_circuit(). The loaded circuit will be functionally
        identical to the original.

        Args:
            path (Path): Full file path to a JSON file created by save_circuit().

        Returns:
            Circuit: A pytket Circuit object semantically identical to the one
                that was originally saved.
                
        Raises:
            ValueError: If the circuit fails to load from the JSON file.
        """
        with path.open("r") as f:
            circ_dict = json.load(f)
        circ = Circuit.from_dict(circ_dict)
        if circ is None:
            raise ValueError(f"Failed to load circuit from {path}")
        return circ
    
    def transpile_and_analyze(
        self,
        backend: Any,
        backend_alias: str,
        opt_level: int = 0,
        force_overwrite: bool = False,
        force_rebuild_main: bool = False
    ) -> dict[str, Any]:
        """Transpile the main circuit for a specific backend and analyze resources.
        
        Compiles the main quantum circuit for the specified backend at the given
        optimization level. Handles caching of transpiled circuits and collects
        resource metrics. Updates metadata with compilation results. Automatically
        routes to SDK-specific transpilation methods based on backend type.

        Args:
            backend (Any): The backend instance to compile for (PyTKET, Qiskit, or Braket).
            backend_alias (str): Human-readable alias for the backend used in
                metadata and error reporting.
            opt_level (int, optional): Optimization level for transpilation.
                Higher values may reduce circuit size. Defaults to 0.
            force_overwrite (bool, optional): If True, recompiles even if a
                cached transpiled circuit exists. Defaults to False.
            force_rebuild_main (bool, optional): If True, rebuilds the main
                circuit before transpilation. Defaults to False.

        Returns:
            dict[str, Any]: Dictionary containing either:
                - Success: {"n_qubits": int, "n_gates": int, "depth": int, "sdk_type": str, ...}
                - Failure: {"error": str} with error description
                
        Side Effects:
            - Caches transpiled circuit to disk (if store_transpiled is True)
            - Updates metadata with backend resource metrics and SDK type
            - Persists metadata to disk
        """
        # ensure main circuit
        if self.main_circuit is None:
            self.build_main_circuit(force_overwrite=force_rebuild_main)

        # Detect SDK type from backend
        sdk_type = self._detect_backend_sdk(backend)
        
        # Get SDK-aware cache path
        path = self.transpiled_circuit_path(backend_alias, opt_level, sdk_type=sdk_type)

        try:
            # Check if Braket (which doesn't support client-side transpilation)
            if sdk_type == "braket":
                raise NotImplementedError(
                    f"AWS Braket performs transpilation server-side. "
                    f"Pre-transpilation is not supported for backend '{backend_alias}'. "
                    f"Access transpiled circuit information after execution via task.result()."
                )
            
            # load or compile
            if self.store_transpiled and path.exists() and not force_overwrite:
                # Load from cache - need to handle SDK-specific formats
                if sdk_type == "pytket":
                    tcirc = self.load_circuit(path)
                elif sdk_type == "qiskit":
                    tcirc = self._load_qiskit_circuit(path)
                else:
                    tcirc = self.load_circuit(path)  # Fallback
            else:
                # Route to SDK-specific transpilation
                if sdk_type == "pytket":
                    tcirc = self._transpile_pytket(backend, opt_level)
                elif sdk_type == "qiskit":
                    tcirc = self._transpile_qiskit(backend, opt_level)
                elif sdk_type == "braket":
                    tcirc = self._transpile_braket(backend, opt_level)
                else:
                    raise ValueError(f"Unsupported SDK type: {sdk_type}")
                
                # Cache if enabled
                if self.store_transpiled:
                    if sdk_type == "pytket":
                        self.save_circuit(tcirc, path)
                    elif sdk_type == "qiskit":
                        self._save_qiskit_circuit(tcirc, path)

            # Extract metrics using SDK-aware method
            res = self._extract_transpiled_metrics(tcirc, sdk_type)
            res["sdk_type"] = sdk_type

            # persist metadata
            self._metadata.set_backend_resources(
                self.solver_name, self.encoding,
                backend_alias, opt_level, res
            )
            self._metadata.save()
            
            return res

        except Exception as e:
            err = {"error": str(e), "sdk_type": sdk_type}
            self._metadata.set_backend_resources(
                self.solver_name, self.encoding,
                backend_alias, opt_level, err
            )
            self._metadata.save()
            
            return err

    def _run_qiskit_native(
        self,
        backend: Any,
        circuit: QuantumCircuit,
        shots: int = 1024,
    ) -> Any:
        """Execute a Qiskit circuit on a native Qiskit backend.
        
        This method handles execution for native qiskit_ibm_runtime backends
        that use backend.run() instead of pytket's process_circuit().
        
        Args:
            backend: Native Qiskit runtime backend instance
            circuit: Transpiled Qiskit QuantumCircuit
            shots: Number of measurement shots
            
        Returns:
            Job result object with counts and metadata
        """
        # Native Qiskit execution uses backend.run()
        job = backend.run(circuit, shots=shots)
        result = job.result()
        # Attach compiled circuit to result for downstream metrics
        try:
            setattr(result, 'compiled_circuit', circuit)
        except Exception:
            pass
        return result
    
    def _run_pytket(
        self,
        backend: Any,
        circuit: Circuit,
        shots: int = 1024,
    ) -> Any:
        """Execute a pytket circuit on a pytket backend.
        
        Args:
            backend: PyTKET backend instance
            circuit: Transpiled pytket Circuit
            shots: Number of measurement shots
            
        Returns:
            Result object from backend execution
        """
        handle = backend.process_circuit(circuit, n_shots=shots)
        result = backend.get_result(handle)
        return result

    def run(
        self,
        backend: Any,
        backend_alias: str,
        shots: int = 1024,
        force_run: bool = False,
        optimisation_level: int = 1,
        use_zne: bool = False,
        use_pec: bool = False,
        zne_scale_noise: Any = None,
        zne_factory: Any = None,
        pec_representations: Any = None,
    ):
        """Run the transpiled circuit on the specified quantum backend.
        
        Executes the quantum circuit on the provided backend after transpilation.
        Handles circuit caching and ensures type safety throughout execution.
        The circuit is automatically built and transpiled if not already available.
        Optionally applies error mitigation via ZNE or PEC.
        
        Now supports both PyTKET backends (using process_circuit) and native
        Qiskit runtime backends (using backend.run()).

        Args:
            backend (Any): The backend instance to execute on (PyTKET or native Qiskit).
            backend_alias (str): Human-readable alias of the backend used for
                metadata tracking and logging purposes.
            shots (int, optional): Number of measurement shots for execution.
                Higher values provide better statistics. Defaults to 1024.
            force_run (bool, optional): If True, re-transpiles and re-executes
                even if cached transpiled circuit exists. Defaults to False.
            optimisation_level (int, optional): Optimization level for circuit
                transpilation. Higher levels may reduce gate count. Defaults to 1.
            use_zne (bool, optional): Enable Zero Noise Extrapolation error mitigation.
                Defaults to False.
            use_pec (bool, optional): Enable Probabilistic Error Cancellation mitigation.
                Defaults to False.
            zne_scale_noise (Callable, optional): Noise scaling function for ZNE.
                If None, uses Mitiq's default folding strategy.
            zne_factory (Any, optional): Extrapolation factory for ZNE.
                If None, uses Richardson extrapolation with default polynomial degree.
            pec_representations (Any, optional): OperationRepresentation list for PEC.
                Required if use_pec=True. Maps ideal gates to noisy implementations.
        
        TODO:
            Align external naming with QSudoku.run which uses ``opt_level``.
            Consider accepting both and mapping consistently across layers.

        Returns:
            Any: Result object from backend execution containing measurement
                outcomes and job metadata. If mitigation is enabled, result
                contains additional `mitigated_success_prob` attribute.
                
        Raises:
            TypeError: If the compiled circuit is not in the expected format.
            ImportError: If use_zne or use_pec is True but Mitiq is not installed.
            ValueError: If use_pec is True but pec_representations is None,
                or if both use_zne and use_pec are True.
        """
        
        # Ensure main circuit is built with backend context
        if self.main_circuit is None:
            self.build_main_circuit(backend)
        
        # Detect backend SDK type
        sdk_type = self._detect_backend_sdk(backend)
        
        # Get SDK-aware transpiled circuit path
        path = self.transpiled_circuit_path(backend_alias, optimisation_level, sdk_type=sdk_type)
        
        # Load or transpile circuit
        if self.store_transpiled and path.exists() and not force_run:
            # Load from cache
            if sdk_type == "qiskit":
                compiled_circuit = self._load_qiskit_circuit(path)
            else:  # pytket
                compiled_circuit = self.load_circuit(path)
        else:
            # Transpile using SDK-specific method
            if sdk_type == "pytket":
                compiled_circuit = self._transpile_pytket(backend, optimisation_level)
            elif sdk_type == "qiskit":
                compiled_circuit = self._transpile_qiskit(backend, optimisation_level)
            else:
                raise ValueError(f"Unsupported SDK type for execution: {sdk_type}")
            
            # Cache if enabled
            if self.store_transpiled:
                if sdk_type == "qiskit":
                    self._save_qiskit_circuit(compiled_circuit, path)
                else:  # pytket
                    self.save_circuit(compiled_circuit, path)

        # Type validation based on SDK
        if sdk_type == "pytket":
            if not isinstance(compiled_circuit, Circuit):
                raise TypeError(f"Expected pytket Circuit, got {type(compiled_circuit)}")
        elif sdk_type == "qiskit":
            if not isinstance(compiled_circuit, QuantumCircuit):
                raise TypeError(f"Expected Qiskit QuantumCircuit, got {type(compiled_circuit)}")

        # Record last run context for metadata and metrics
        try:
            setattr(self, 'last_backend_alias', backend_alias)
            setattr(self, 'last_opt_level', optimisation_level)
            setattr(self, 'transpiled_circuit', compiled_circuit)
            if sdk_type == "qiskit":
                try:
                    self.gate_counts = dict(compiled_circuit.count_ops())
                except Exception:
                    pass
        except Exception:
            pass
        
        # Apply error mitigation if requested
        if use_zne or use_pec:
            from sudoku_nisq.mitigation.executors import apply_zne, apply_pec
            
            if use_zne and use_pec:
                raise ValueError("Cannot use both ZNE and PEC simultaneously. Choose one.")
            
            if use_zne:
                mitigated_expectation = apply_zne(
                    compiled_circuit,
                    backend,
                    self,
                    shots=shots,
                    scale_noise=zne_scale_noise,
                    factory=zne_factory,
                )
                # Standard execution for full result
                if sdk_type == "qiskit":
                    result = self._run_qiskit_native(backend, compiled_circuit, shots)
                else:  # pytket
                    result = self._run_pytket(backend, compiled_circuit, shots)
                
                # Attach mitigated value as metadata
                if hasattr(result, '__dict__'):
                    result.mitigated_success_prob = mitigated_expectation
                return result
            
            if use_pec:
                if pec_representations is None:
                    raise ValueError(
                        "PEC requires 'pec_representations' parameter. "
                        "See Mitiq documentation for OperationRepresentation generation."
                    )
                mitigated_expectation = apply_pec(
                    compiled_circuit,
                    backend,
                    self,
                    pec_representations,
                    shots=shots,
                )
                # Standard execution
                if sdk_type == "qiskit":
                    result = self._run_qiskit_native(backend, compiled_circuit, shots)
                else:  # pytket
                    result = self._run_pytket(backend, compiled_circuit, shots)
                
                # Attach mitigated value
                if hasattr(result, '__dict__'):
                    result.mitigated_success_prob = mitigated_expectation
                return result
        
        # Standard execution (no mitigation)
        if sdk_type == "qiskit":
            return self._run_qiskit_native(backend, compiled_circuit, shots)
        else:  # pytket
            return self._run_pytket(backend, compiled_circuit, shots)
    
    def run_aer(
        self, 
        shots: int = 1024,
        method: str = "automatic",
        noise_model: Any = None,
        coupling_map: Any = None,
        basis_gates: list[str] | None = None,
        device: str = "CPU",
        precision: str = "double",
        optimization_level: int = 1,
        seed_simulator: int | None = None,
        max_parallel_threads: int | None = None,
        max_parallel_experiments: int | None = None,
        blocking_enable: bool = True,
        blocking_qubits: int = 5,
        **backend_options
    ) -> Any:
        """Run the main circuit on Qiskit Aer simulator with full configuration support.
        
        Executes the quantum circuit using native Qiskit Aer with comprehensive
        control over simulation method, noise models, device selection, and performance
        options. Supports ideal and noisy simulation, multiple simulation methods
        (statevector, density_matrix, MPS, etc.), and GPU acceleration when available.
        
        Args:
            shots (int, optional): Number of measurement shots for simulation.
                Higher values provide better statistical accuracy. Defaults to 1024.
            method (str, optional): Simulation method to use. Options include:
                - "automatic": Auto-select based on circuit (default)
                - "statevector": Dense statevector simulation (ideal for small circuits)
                - "density_matrix": Density matrix simulation (supports noise)
                - "stabilizer": Clifford simulator (fast for Clifford circuits)
                - "extended_stabilizer": Approximate Clifford+T simulator
                - "matrix_product_state": MPS/tensor network simulator
                - "unitary": Compute circuit unitary (no measurement)
                - "superop": Compute superoperator representation
                Defaults to "automatic".
            noise_model (NoiseModel, optional): Qiskit Aer noise model for noisy
                simulation. Can be created from real devices or custom error channels.
            coupling_map (list or CouplingMap, optional): Device coupling map for
                layout constraints and hardware emulation.
            basis_gates (list, optional): Basis gates for device emulation. Circuit
                will be decomposed to these gates during transpilation.
            device (str, optional): Compute device selection: "CPU" or "GPU".
                GPU requires qiskit-aer-gpu package. Defaults to "CPU".
            precision (str, optional): Floating point precision: "single" or "double".
                Single precision uses less memory and may be faster. Defaults to "double".
            optimization_level (int, optional): Qiskit transpiler optimization level
                (0-3). Higher levels apply more optimizations but take longer.
                Defaults to 1.
            seed_simulator (int, optional): Random seed for reproducible simulation.
                If None, uses random seed.
            max_parallel_threads (int, optional): Maximum threads for OpenMP
                parallelization. If None, uses system default.
            max_parallel_experiments (int, optional): Maximum parallel circuit
                executions for batched jobs. If None, uses Aer default.
            blocking_enable (bool, optional): Enable automatic qubit blocking for
                large circuits to reduce memory usage. Defaults to True.
            blocking_qubits (int, optional): Qubits per block when blocking is enabled.
                Defaults to 5.
            **backend_options: Additional AerSimulator backend options. See Qiskit
                Aer documentation for complete list of supported options.
            
        Returns:
            Any: Qiskit Result object containing measurement counts, execution
                metadata, and optional saved state data.
                
        Raises:
            ImportError: If qiskit-aer is not installed.
            
        Examples:
            Ideal statevector simulation:
            
            >>> result = solver.run_aer(shots=1024, method="statevector")
            
            Noisy simulation with custom noise model:
            
            >>> from qiskit_aer.noise import NoiseModel, depolarizing_error
            >>> noise = NoiseModel()
            >>> noise.add_all_qubit_quantum_error(
            ...     depolarizing_error(0.01, 2), ['cx']
            ... )
            >>> result = solver.run_aer(
            ...     shots=4096,
            ...     method="density_matrix",
            ...     noise_model=noise
            ... )
            
            GPU-accelerated MPS simulation:
            
            >>> result = solver.run_aer(
            ...     shots=2048,
            ...     method="matrix_product_state",
            ...     device="GPU",
            ...     precision="single"
            ... )
            
            Device emulation from real backend:
            
            >>> from qiskit_ibm_runtime import QiskitRuntimeService
            >>> from qiskit_aer.noise import NoiseModel
            >>> service = QiskitRuntimeService()
            >>> real_backend = service.backend("ibm_brisbane")
            >>> noise = NoiseModel.from_backend(real_backend)
            >>> result = solver.run_aer(
            ...     shots=8192,
            ...     method="density_matrix",
            ...     noise_model=noise,
            ...     coupling_map=real_backend.coupling_map,
            ...     basis_gates=real_backend.configuration().basis_gates,
            ...     optimization_level=2
            ... )
        """
        try:
            from qiskit_aer import AerSimulator
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        except ImportError as e:
            raise ImportError(
                "qiskit-aer is required for Aer simulation. "
                "Install with: pip install qiskit-aer"
            ) from e
        
        # Build circuit in Qiskit format (SDK detection will handle this)
        if self.main_circuit is None or not hasattr(self.main_circuit, 'qubits'):
            # Build or convert to Qiskit format
            self.main_circuit = self.build_main_circuit(sdk="qiskit")
        
        # Ensure we have a Qiskit circuit
        if not isinstance(self.main_circuit, QuantumCircuit):
            # Try to convert from pytket if needed
            if hasattr(self.main_circuit, 'to_qiskit'):
                from pytket.extensions.qiskit import tk_to_qiskit
                qc = tk_to_qiskit(self.main_circuit)
            else:
                # Rebuild as Qiskit circuit
                qc = self.build_main_circuit(sdk="qiskit")
        else:
            qc = self.main_circuit
        
        # Configure AerSimulator with all options
        aer_options = {
            "method": method,
            "device": device,
            "precision": precision,
            "blocking_enable": blocking_enable,
            "blocking_qubits": blocking_qubits,
        }
        
        # Add optional parameters
        if noise_model is not None:
            aer_options["noise_model"] = noise_model
        if coupling_map is not None:
            aer_options["coupling_map"] = coupling_map
        if basis_gates is not None:
            aer_options["basis_gates"] = basis_gates
        if seed_simulator is not None:
            aer_options["seed_simulator"] = seed_simulator
        if max_parallel_threads is not None:
            aer_options["max_parallel_threads"] = max_parallel_threads
        if max_parallel_experiments is not None:
            aer_options["max_parallel_experiments"] = max_parallel_experiments
        
        # Merge additional backend options
        aer_options.update(backend_options)
        
        # Create AerSimulator
        backend = AerSimulator(**aer_options)
        
        # Transpile circuit for Aer
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend
        )
        transpiled_qc = pm.run(qc)
        
        # Run simulation
        job = backend.run(transpiled_qc, shots=shots)
        result = job.result()
        # Attach transpiled circuit to result for eta/eta2 computation
        try:
            setattr(result, 'compiled_circuit', transpiled_qc)
        except Exception:
            pass
        
        return result
    
    def counts_plot(self, counts=None, backend_alias=None, shots=None, top_n=20, 
                    show_valid_only=False, figsize=(12, 6), show_summary=True):
        """Create a bar plot of measurement counts with Sudoku-specific enhancements.
        
        Provides rich visualization for analyzing quantum execution results with
        automatic validation highlighting and comprehensive summary statistics.
        Supports filtering for valid solutions and customizable display options.
        
        Args:
            counts (dict or Result, optional): Dictionary of measurement outcomes
                or pytket Result object. If None, raises ValueError.
            backend_alias (str, optional): Name of the backend used for title
                display. If None, the title defaults to "Aer Simulator" at this
                level. The QSudoku wrapper supplies "Unknown Backend" when a value
                isn't provided.
            shots (int, optional): Total number of shots for percentage calculation.
                If None, calculated from counts data.
            top_n (int, optional): Maximum number of most frequent outcomes to display.
                Remaining outcomes are grouped as "Others". Defaults to 20.
            show_valid_only (bool, optional): If True, only displays outcomes that
                represent valid Sudoku solutions. Defaults to False.
            figsize (tuple, optional): Figure size as (width, height) in inches.
                Defaults to (12, 6).
            show_summary (bool, optional): If True, prints summary statistics
                including valid solution percentages. Defaults to True.
            
        Raises:
            ValueError: If counts is None, empty, or not a valid format.
            
        Examples:
            Plot results from Aer simulation:
            
            >>> result = solver.run_aer(shots=1024)
            >>> solver.counts_plot(result, backend_alias="Aer", shots=1024)
            
            Plot only valid solutions from hardware run:
            
            >>> result = solver.run("ibm_brisbane", opt_level=1, shots=100)
            >>> solver.counts_plot(result, backend_alias="IBM Brisbane", 
            ...                    show_valid_only=True)
            
            Plot without summary statistics:
            
            >>> solver.counts_plot(result, backend_alias="Aer", show_summary=False)
        """
        # TODO: Align default backend alias title across QuantumSolver and QSudoku
        # to avoid confusion ("Aer Simulator" vs "Unknown Backend").
        import matplotlib.pyplot as plt

        # Handle different input types
        if counts is None:
            raise ValueError("No counts provided. Pass counts dictionary or run a quantum execution first.")

        if hasattr(counts, 'get_counts'):
            counts_dict = counts.get_counts()
        elif isinstance(counts, dict):
            counts_dict = counts
        else:
            raise ValueError("counts must be a dictionary or pytket Result object with get_counts() method")

        if not counts_dict:
            raise ValueError("Empty counts dictionary")

        # Filter valid solutions if requested
        if show_valid_only:
            filtered_counts = {}
            for bitstring, count in counts_dict.items():
                bitstring_str = ''.join(str(bit) for bit in bitstring) if isinstance(bitstring, tuple) else str(bitstring)
                if self._is_valid_solution(bitstring_str):
                    filtered_counts[bitstring] = count
            counts_dict = filtered_counts
            if not counts_dict:
                print("No valid solutions found in measurement outcomes")
                return

        # Sort by frequency and take top_n
        sorted_counts = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_counts) > top_n:
            top_counts = dict(sorted_counts[:top_n])
            other_count = sum(count for _, count in sorted_counts[top_n:])
            if other_count > 0:
                top_counts[f"Others ({len(sorted_counts) - top_n})"] = other_count
        else:
            top_counts = dict(sorted_counts)

        total_shots = shots or sum(counts_dict.values())
        labels = list(top_counts.keys())
        values = list(top_counts.values())
        percentages = [100 * v / total_shots for v in values]

        # Convert labels to strings for display
        label_strings = []
        for label in labels:
            if isinstance(label, tuple):
                label_strings.append(''.join(str(bit) for bit in label))
            else:
                label_strings.append(str(label))

        fig, ax = plt.subplots(figsize=figsize)

        # Determine coloring strategy
        colors = []
        has_validation = hasattr(self, '_is_valid_solution') and callable(getattr(self, '_is_valid_solution', None))
        use_validation_colors = not show_valid_only and has_validation
        for label_str in label_strings:
            if label_str.startswith("Others"):
                colors.append('lightgray')
            elif show_valid_only or not use_validation_colors:
                colors.append('steelblue')
            else:
                try:
                    colors.append('steelblue' if self._is_valid_solution(label_str) else 'lightcoral')
                except Exception:
                    colors.append('steelblue')

        bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7)
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Measurement Outcomes (Bitstrings)')
        ax.set_ylabel('Counts')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(label_strings, rotation=45, ha='right')

        title_parts = [f"{self.solver_name} Results"]
        title_parts.append(f"Backend: {backend_alias}" if backend_alias else "Backend: Aer Simulator")
        title_parts.append(f"Shots: {total_shots}")
        if show_valid_only:
            title_parts.append("(Valid Solutions Only)")
        ax.set_title(" | ".join(title_parts))

        ax.grid(axis='y', alpha=0.3)
        if use_validation_colors:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='steelblue', alpha=0.7, label='Valid Solutions'),
                Patch(facecolor='lightcoral', alpha=0.7, label='Invalid Solutions'),
                Patch(facecolor='lightgray', alpha=0.7, label='Others')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

        if show_summary:
            valid_count = sum(count for bitstring, count in counts_dict.items()
                              if self._is_valid_solution(''.join(str(bit) for bit in bitstring) if isinstance(bitstring, tuple) else str(bitstring)))
            print("\nSummary:")
            print(f"Total unique outcomes: {len(counts_dict)}")
            print(f"Total shots: {total_shots}")
            print(f"Valid solutions found: {valid_count} ({100*valid_count/total_shots:.2f}%)")
            most_frequent_key = sorted_counts[0][0]
            most_frequent_str = ''.join(str(bit) for bit in most_frequent_key) if isinstance(most_frequent_key, tuple) else str(most_frequent_key)
            print(f"Most frequent outcome: {most_frequent_str} ({100*sorted_counts[0][1]/total_shots:.2f}%)")

    def _is_valid_solution(self, bitstring):
        """Check if a measurement outcome represents a valid Sudoku solution.
        
        This default implementation returns True for all inputs, meaning all
        measurement outcomes will be colored as valid in visualization plots.
        Subclasses should override this method to implement proper validation
        logic based on their specific encoding schemes and constraint checking.
        
        Args:
            bitstring (str): String representation of the measurement outcome
                from quantum circuit execution.
            
        Returns:
            bool: True if the bitstring represents a valid Sudoku solution,
                False otherwise. Default implementation always returns True.
                
        Note:
            This is a placeholder implementation. Subclasses must override this
            method with proper validation logic that:
            1. Decodes the bitstring according to their encoding scheme
            2. Converts the decoded result to a Sudoku solution format
            3. Validates the solution against Sudoku constraints
            
        Example implementation for subclasses:
        
        >>> def _is_valid_solution(self, bitstring):
        ...     # Convert bitstring to Sudoku solution
        ...     solution = self.decode_bitstring(bitstring)
        ...     # Check if solution satisfies Sudoku constraints
        ...     return self.validate_sudoku_solution(solution)
        """
        # Default implementation - shows that validation is "implemented" but always returns True
        # Override with real validation
        return True
    
    def _extract_transpiled_metrics(self, circuit: Any, sdk_type: str) -> dict[str, Any]:
        """Extract resource metrics from a transpiled circuit in SDK-specific format.
        
        Args:
            circuit: Transpiled circuit in any SDK format
            sdk_type: SDK type of the circuit ("pytket", "qiskit", "braket")
            
        Returns:
            dict: Resource metrics including n_qubits, n_gates, depth, and SDK-specific data
        """
        if sdk_type == "pytket":
            # PyTKET format
            return {
                "n_qubits": circuit.n_qubits,
                "n_gates": circuit.n_gates,
                "depth": circuit.depth(),
            }
        elif sdk_type == "qiskit":
            # Qiskit format - preserve both total depth and depth_by_qubit
            gate_count = sum(circuit.count_ops().values()) if hasattr(circuit, 'count_ops') else 0
            metrics = {
                "n_qubits": circuit.num_qubits,
                "n_gates": gate_count,
                "depth": circuit.depth() if hasattr(circuit, 'depth') else 0,
            }
            # Add Qiskit-specific gate counts
            if hasattr(circuit, 'count_ops'):
                metrics["gate_counts"] = dict(circuit.count_ops())
            # Add depth_by_qubit if available (Qiskit-specific)
            if hasattr(circuit, 'depth') and callable(circuit.depth):
                try:
                    # Try to get per-qubit depth (available in some Qiskit versions)
                    from qiskit.converters import circuit_to_dag
                    dag = circuit_to_dag(circuit)
                    if hasattr(dag, 'depth'):
                        metrics["depth_by_qubit"] = {
                            f"q{i}": dag.depth(filter_function=lambda node: i in [q.index for q in node.qargs])
                            for i in range(circuit.num_qubits)
                        }
                except Exception:
                    pass  # Skip if not available
            return metrics
        elif sdk_type == "braket":
            # Braket format (if we ever support it)
            raise NotImplementedError("Braket transpilation metrics not implemented")
        else:
            raise ValueError(f"Unknown SDK type: {sdk_type}")
    
    def _save_qiskit_circuit(self, circuit: Any, path: Path) -> None:
        """Save a Qiskit QuantumCircuit to JSON format.
        
        Args:
            circuit: Qiskit QuantumCircuit to save
            path: File path for saving
        """
        from qiskit import qpy
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use QPY format for Qiskit circuits (more robust than QASM)
        try:
            with path.open("wb") as f:
                qpy.dump(circuit, f)
        except Exception as e:
            # QPY serialization can fail for certain circuits (e.g., with custom gates)
            # Fall back to QASM if available, or skip caching
            import logging
            logging.warning(f"Failed to save Qiskit circuit using QPY: {e}. Circuit caching skipped.")
            # Don't raise - just skip caching for this circuit
    
    def _load_qiskit_circuit(self, path: Path) -> Any:
        """Load a Qiskit QuantumCircuit from JSON format.
        
        Args:
            path: File path to load from
            
        Returns:
            Qiskit QuantumCircuit
        """
        from qiskit import qpy
        
        with path.open("rb") as f:
            circuits = qpy.load(f)
            return circuits[0] if isinstance(circuits, list) else circuits

    @staticmethod
    def count_mcx_gates(circuit: Circuit) -> int:
        """Count the total number of multi-controlled X (CnX) gates in a circuit.
        
        Analyzes the quantum circuit to count all multi-controlled X gates, which
        are typically expensive operations on quantum hardware and important for
        resource estimation and algorithm analysis.

        Args:
            circuit (Circuit): A pytket Circuit object to analyze for MCX gates.

        Returns:
            int: The total count of OpType.CnX gates found in the circuit.
                Returns 0 if no MCX gates are present.
        """
        counts = gate_counts(circuit)
        return counts.get(OpType.CnX, 0)