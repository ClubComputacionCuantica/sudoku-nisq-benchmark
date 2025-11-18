import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any
from pytket import Circuit, OpType
from pytket.utils import gate_counts
from pytket.passes import FlattenRegisters

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
        puzzle: SudokuPuzzle,
        metadata_manager: MetadataManager,
        encoding: str | None = None, 
        store_transpiled: bool = True,
    ):
        """Initialize the QuantumSolver base class infrastructure.
        
        Sets up the common infrastructure that concrete solver subclasses will use,
        including puzzle context, metadata management, and caching configuration.
        This base class does not implement any solving algorithm itself.

        Args:
            puzzle (SudokuPuzzle): The Sudoku puzzle instance to solve. Must not be None.
            metadata_manager (MetadataManager): Instance for managing circuit metadata,
                caching, and performance tracking. Must not be None.
            encoding (str, optional): Encoding strategy name for the quantum algorithm.
                If None, defaults to "default".
            store_transpiled (bool, optional): Whether to save transpiled circuits to
                disk for caching. Defaults to True for better performance on repeated runs.
                
        Raises:
            ValueError: If puzzle or metadata_manager is None.
        """
        
        if puzzle is None:
            raise ValueError("A SudokuPuzzle instance is required.")
        if metadata_manager is None:
            raise ValueError("A MetadataManager instance is required.")
        
        # Sudoku integration for puzzle-specific caching
        self.puzzle = puzzle
        self._metadata = metadata_manager
        self.encoding = encoding or "default"  # Default encoding if not specified
        self.store_transpiled = store_transpiled
        
        # Circuit management - now SDK-agnostic
        self.main_circuit: Any | None = None
        
        # Cache base derived from metadata manager
        self.cache_base = self._metadata.cache_base
        
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
    
    def _build_circuit(self, backend: Any = None) -> Any:
        """Construct quantum circuit with automatic SDK detection.
        
        This method handles SDK detection and delegates to the abstract
        _build_sdk_circuit method that subclasses must implement.
        """
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
    
    @property
    def puzzle_hash(self) -> str:
        """str: Unique hash identifier for the current Sudoku puzzle."""
        return self.puzzle.get_hash()

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
    
    def transpiled_circuit_path(self, backend_alias: str, opt_level: int) -> Path:
        """Get the file path for a transpiled circuit cache.
        
        Constructs the cache path for a circuit transpiled for a specific backend
        and optimization level.
        
        Args:
            backend_alias (str): Alias of the target backend.
            opt_level (int): Optimization level used for transpilation.
            
        Returns:
            Path: File path for the transpiled circuit cache.
                Format: .quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}/
                        {backend_alias}/opt{opt_level}_circuit.json
        """
        return self.cache_root / backend_alias / f"opt{opt_level}_circuit.json"

    def build_main_circuit(self, backend: Any = None, force_overwrite: bool = False, flatten: bool = True) -> Any:
        """Load or build the main quantum circuit for the solving algorithm.
        
        Manages circuit caching by loading from disk if available, or building a new
        circuit using _build_circuit() and saving it for future use. Automatically
        handles register flattening for backend compatibility and updates metadata
        with circuit resource information.
        
        Args:
            backend (Any, optional): Backend instance that determines which SDK to use.
                If provided, the circuit will be built using the backend's native SDK.
            force_overwrite (bool, optional): If True, rebuilds the circuit even if
                a cached version exists. Defaults to False.
            flatten (bool, optional): If True, applies register flattening for
                compatibility with generic backends. Defaults to True.
                
        Returns:
            Any: The main quantum circuit ready for transpilation and execution,
                in the format appropriate for the backend (pytket.Circuit, qiskit.QuantumCircuit, etc.).
            
        Side Effects:
            - Sets self.main_circuit to the built/loaded circuit
            - Saves circuit to disk cache if newly built (pytket format for compatibility)
            - Updates metadata with circuit resource metrics
            - Persists metadata to disk
        """
        path = self.main_circuit_path
        if path.exists() and not force_overwrite:
            circ = self.load_circuit(path)
        else:
            # Delegate to the subclass with backend context
            circ = self._build_circuit(backend)
            
            # For caching and metadata, convert to pytket format if needed
            cache_circuit = self._ensure_pytket_format(circ, backend)
            
            # Flatten registers for compatibility with generic backends (pytket specific)
            if flatten and hasattr(cache_circuit, 'n_qubits'):  # pytket circuit
                FlattenRegisters().apply(cache_circuit)
            
            # Persist (always save main circuit regardless of store_transpiled flag)
            self.save_circuit(cache_circuit, path)
            
            # Record main circuit resources using pytket format for consistency
            main_res = self._get_circuit_resources(cache_circuit)
            self._metadata.set_main_circuit_resources(self.solver_name, self.encoding, main_res)
            self._metadata.save()

        self.main_circuit = circ
        return circ

    def _detect_backend_sdk(self, backend: Any) -> str:
        """Detect which SDK the backend uses based on its interface.
        
        Args:
            backend: The backend instance to analyze
            
        Returns:
            str: SDK name ("pytket", "qiskit", "braket")
        """
        if backend is None:
            return "pytket"  # Default fallback
            
        # Check for pytket backend characteristics
        if hasattr(backend, 'get_compiled_circuit') and hasattr(backend, 'process_circuit'):
            return "pytket"
        
        # Check for qiskit backend characteristics  
        elif hasattr(backend, 'transpile') or 'qiskit' in str(type(backend)).lower():
            return "qiskit"
            
        # Check for braket backend characteristics
        elif hasattr(backend, 'run') and 'braket' in str(type(backend)).lower():
            return "braket"
            
        else:
            # Default to pytket for unknown backends
            return "pytket"

    def _ensure_pytket_format(self, circuit: Any, backend: Any = None) -> Circuit:
        """Convert circuit to pytket format for caching and metadata consistency.
        
        Args:
            circuit: Circuit in any SDK format
            backend: Backend context (optional)
            
        Returns:
            Circuit: Circuit converted to pytket format
        """
        # If already pytket, return as-is
        if hasattr(circuit, 'n_qubits') and hasattr(circuit, 'n_gates'):
            return circuit
            
        sdk_type = self._detect_backend_sdk(backend)
        
        if sdk_type == "qiskit":
            # Convert qiskit to pytket
            from pytket.extensions.qiskit import qiskit_to_tk
            return qiskit_to_tk(circuit)
            
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
            dict: Resource metrics (n_qubits, n_gates, etc.)
        """
        if hasattr(circuit, 'n_qubits'):  # pytket format
            return {
                "n_qubits": circuit.n_qubits,
                "n_gates": circuit.n_gates,
                "n_mcx_gates": self.count_mcx_gates(circuit),
                "depth": circuit.depth(),
            }
        elif hasattr(circuit, 'num_qubits'):  # qiskit format
            gate_count = sum(circuit.count_ops().values()) if hasattr(circuit, 'count_ops') else 0
            mcx_count = circuit.count_ops().get('mcx', 0) if hasattr(circuit, 'count_ops') else 0
            return {
                "n_qubits": circuit.num_qubits,
                "n_gates": gate_count,
                "n_mcx_gates": mcx_count,
                "depth": circuit.depth() if hasattr(circuit, 'depth') else 0,
            }
        else:
            # Default/unknown format
            return {
                "n_qubits": 0,
                "n_gates": 0,
                "n_mcx_gates": 0,
                "depth": 0,
            }

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
        resource metrics. Updates metadata with compilation results.

        Args:
            backend (Any): The pytket backend instance to compile for.
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
                - Success: {"n_qubits": int, "n_gates": int, "depth": int}
                - Failure: {"error": str} with error description
                
        Side Effects:
            - Caches transpiled circuit to disk (if store_transpiled is True)
            - Updates metadata with backend resource metrics
            - Persists metadata to disk
        """
        # ensure main circuit
        if self.main_circuit is None:
            self.build_main_circuit(force_overwrite=force_rebuild_main)

        path = self.transpiled_circuit_path(backend_alias, opt_level)

        try:
            # load or compile
            if self.store_transpiled and path.exists() and not force_overwrite:
                tcirc = self.load_circuit(path)
            else:
                try:
                    tcirc = backend.get_compiled_circuit(
                        self.main_circuit,
                        optimisation_level=opt_level
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to compile circuit for {backend_alias} "
                        f"at opt_level {opt_level}: {e}"
                    )
                if self.store_transpiled:
                    self.save_circuit(tcirc, path)

            # extract metrics
            res = {
                "n_qubits": tcirc.n_qubits,
                "n_gates":  tcirc.n_gates,
                "depth":    tcirc.depth(),
            }

            # persist metadata
            self._metadata.set_backend_resources(
                self.solver_name, self.encoding,
                backend_alias, opt_level, res
            )
            self._metadata.save()
            
            return res

        except Exception as e:
            err = {"error": str(e)}
            self._metadata.set_backend_resources(
                self.solver_name, self.encoding,
                backend_alias, opt_level, err
            )
            self._metadata.save()
            
            return err

    def run(
        self,
        backend: Any,
        backend_alias: str,
        shots: int = 1024,
        force_run: bool = False,
        optimisation_level: int = 1
    ):
        """Run the transpiled circuit on the specified quantum backend.
        
        Executes the quantum circuit on the provided backend after transpilation.
        Handles circuit caching and ensures type safety throughout execution.
        The circuit is automatically built and transpiled if not already available.

        Args:
            backend (Any): The pytket backend instance to execute on.
            backend_alias (str): Human-readable alias of the backend used for
                metadata tracking and logging purposes.
            shots (int, optional): Number of measurement shots for execution.
                Higher values provide better statistics. Defaults to 1024.
            force_run (bool, optional): If True, re-transpiles and re-executes
                even if cached transpiled circuit exists. Defaults to False.
            optimisation_level (int, optional): Optimization level for circuit
                transpilation. Higher levels may reduce gate count. Defaults to 1.
        
        TODO:
            Align external naming with QSudoku.run which uses ``opt_level``.
            Consider accepting both and mapping consistently across layers.

        Returns:
            Any: Result object from backend execution containing measurement
                outcomes and job metadata.
                
        Raises:
            TypeError: If the compiled circuit is not a valid Circuit instance.
        """
        
        # Ensure main circuit is built with backend context
        if self.main_circuit is None:
            self.build_main_circuit(backend)
            
        path = self.transpiled_circuit_path(backend_alias, optimisation_level)
        
        if self.store_transpiled and path.exists() and not force_run:
            compiled_circuit = self.load_circuit(path)
        else:
            # Transpile
            compiled_circuit = backend.get_compiled_circuit(self.main_circuit, optimisation_level=optimisation_level)
            # Cache only if store_transpiled is True
            if self.store_transpiled:
                self.save_circuit(compiled_circuit, path)

        # Guarantee type safety
        if not isinstance(compiled_circuit, Circuit):
            raise TypeError(f"Expected Circuit, got {type(compiled_circuit)}")
        handle = backend.process_circuit(compiled_circuit, n_shots=shots)  # type: ignore[arg-type]
        result = backend.get_result(handle)
        return result
    
    def run_aer(self, shots: int = 1024, **kwargs) -> Any:
        """Run the main circuit on the local Aer simulator.
        
        Executes the logical quantum circuit on the Qiskit Aer simulator via
        pytket's AerBackend without hardware-specific transpilation. By default
        this runs an ideal (noise-free) simulation; however, Aer supports noise
        models and additional configuration which can be enabled separately.
        
        Args:
            shots (int, optional): Number of measurement shots for simulation.
                Higher values provide better statistical accuracy. Defaults to 1024.
            **kwargs: Additional keyword arguments for compatibility with other
                run methods. Currently unused.
            
        Returns:
            Any: Result object from Aer simulation containing measurement counts
                and execution metadata.
        """
        # Ensure main circuit is built
        if self.main_circuit is None:
            self.build_main_circuit()
            
        # Import Aer backend locally
        from pytket.extensions.qiskit import AerBackend
        
        # Create Aer backend and run simulation
        aer = AerBackend()
        # At this point main_circuit is guaranteed to be non-None
        assert self.main_circuit is not None
        handle = aer.process_circuit(self.main_circuit, n_shots=shots)
        result = aer.get_result(handle)
        
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