import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Sequence, TYPE_CHECKING
from pytket import Circuit, OpType
from pytket.utils import gate_counts
from pytket.passes import FlattenRegisters
from sudoku_nisq.backends import BackendManager

if TYPE_CHECKING:
    from .q_sudoku import QSudoku

class QuantumSolver(ABC):
    """Abstract base class for quantum solvers.
    Provides a framework for building, compiling, and executing quantum circuits
    for solving Sudoku puzzles.
    """

    def __init__(
        self, 
        sudoku: "QSudoku", 
        encoding: str | None = None, 
        store_transpiled: bool = True,
    ):
        """
        Initialize the QuantumSolver with QSudoku instance.

        Args:
            sudoku: QSudoku instance (required)
            encoding: Encoding strategy name
            store_transpiled: Whether to save transpiled circuits to disk (default True)
        """
        
        # Input validation
        if sudoku is None:
            raise ValueError("sudoku instance is required - cannot be None")
        
        # Sudoku integration for puzzle-specific caching
        self.sudoku = sudoku
        self.encoding = encoding or "default"  # Default encoding if not specified
        self.store_transpiled = store_transpiled
        
        # Circuit management
        self.main_circuit: Circuit | None = None

        # Use QSudoku's metadata manager (single source of truth)
        self._metadata = sudoku._metadata
        
        # Cache base derived from metadata manager
        self.cache_base = self._metadata.cache_base
        
    @abstractmethod
    def _build_circuit(self) -> Circuit:
        """
        Construct and return the quantum circuit implementing the solving algorithm.
        """
        pass
    
    @abstractmethod
    def resource_estimation(self):
        """
        Estimate and return resource requirements (num qubits, num gates) for executing main circuit.
        """
        pass
    
    @property
    def puzzle_hash(self) -> str:
        return self.sudoku.get_hash()

    @property
    def solver_name(self) -> str:
        return type(self).__name__
    
    @property
    def board(self) -> list[list[int]]:
        return self.sudoku.board
    
    @property
    def size(self) -> int:
        return self.sudoku.board_size
    
    @property
    def num_missing_cells(self) -> int:
        return self.sudoku.num_missing_cells
    
    @property
    def metadata_path(self) -> Path:
        # .quantum_solver_cache/{puzzle_hash}/metadata.json
        return self.cache_base / self.puzzle_hash / "metadata.json"

    @property
    def cache_root(self) -> Path:
        # .quantum_solver_cache/{puzzle_hash}/{solver_name}
        return self.cache_base / self.puzzle_hash / self.solver_name / self.encoding

    @property
    def main_circuit_path(self) -> Path:
        # .quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}/main_circuit.json
        return self.cache_root / "main_circuit.json"
    
    def transpiled_circuit_path(self, backend_alias: str, opt_level: int) -> Path:
        # .quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}/{backend_alias}/opt{opt_level}_circuit.json
        return self.cache_root / backend_alias / f"opt{opt_level}_circuit.json"

    def build_main_circuit(self, force_overwrite: bool = False, flatten: bool = True) -> Circuit:
        """
        Load the main circuit from disk if cached (and not force_overwrite),
        otherwise call self._build_circuit(), flatten registers once (default for compatibility),
        save to disk, update metadata, and return.
        """
        path = self.main_circuit_path
        if path.exists() and not force_overwrite:
            circ = self.load_circuit(path)
        else:
            # Delegate to the subclass
            circ = self._build_circuit()
            # Flatten registers for compatibility with generic backends
            if flatten:
                FlattenRegisters().apply(circ)
            # Persist (always save main circuit regardless of store_transpiled flag)
            self.save_circuit(circ, path)
            # Record main circuit resources
            main_res = {
                "n_qubits": circ.n_qubits,
                "n_gates":  circ.n_gates,
                "n_mcx_gates": self.count_mcx_gates(circ),
                "depth":     circ.depth(),
            }
            self._metadata.set_main_circuit_resources(self.solver_name, self.encoding, main_res)
            self._metadata.save()

        self.main_circuit = circ
        return circ

    def draw_circuit(self, circuit: Circuit | None = None, **kwargs) -> Circuit:
        """
        Draw the main circuit to a file or return it as a Circuit object.

        Args:
            circuit: The Circuit object to draw (default is self.main_circuit).
            **kwargs: Additional arguments passed to the drawing function.

        Returns:
            The Circuit object representing the drawn circuit.
        """
        from pytket.circuit.display import render_circuit_jupyter as draw
        if circuit is None:
            if self.main_circuit is None:
                raise ValueError("No main circuit available. Please build it first.")
            circuit = self.main_circuit
        return draw(circuit)

    def save_circuit(self, circuit: Circuit, path: Path) -> None:
        """
        Serialize a pytket Circuit to JSON on disk using circuit.to_dict().

        Args:
            circuit:   The pytket Circuit to save.
            path:      Full file path where JSON should be written.
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize to JSON-compatible dict
        circ_dict = circuit.to_dict()
        # Write out
        with path.open("w") as f:
            json.dump(circ_dict, f)

    def load_circuit(self, path: Path) -> Circuit:
        """
        Load a pytket Circuit from JSON on disk using Circuit.from_dict().

        Args:
            path:  Full file path to a JSON file created by save_circuit.
        Returns:
            A Circuit object semantically identical to the one that was saved.
        """
        with path.open("r") as f:
            circ_dict = json.load(f)
        circ = Circuit.from_dict(circ_dict)
        if circ is None:
            raise ValueError(f"Failed to load circuit from {path}")
        return circ
    
    def transpile_and_analyze(
        self,
        backend_alias: str,
        opt_levels: int | Sequence[int] = (0, 1, 2, 3),
        force_overwrite: bool = False,
        force_rebuild_main: bool = False
    ) -> dict[int, dict[str, Any]]:
        """
        Transpile self.main_circuit for each opt_level independently.
        You may pass a single int or a sequence of ints.

        Returns a mapping:
            {
              opt_level: { "n_qubits":…, "n_gates":…, "depth":… }
              OR
              opt_level: { "error": <string> }
            }
        """
        # Ensure main circuit is in memory
        if self.main_circuit is None:
            self.build_main_circuit(force_overwrite=force_rebuild_main)
        
        # Phase 2: Use backend from puzzle's attached backends
        # Backend validation already done in QSudoku.transpile()
        if hasattr(self.sudoku, '_attached_backends') and backend_alias in self.sudoku._attached_backends:
            backend = self.sudoku._attached_backends[backend_alias]
        else:
            # Fallback to global registry for backward compatibility
            backend = BackendManager.get(backend_alias)

        # 1) Normalize to list of ints
        if isinstance(opt_levels, int):
            levels = [opt_levels]
        else:
            levels = list(opt_levels)

        results: dict[int, dict[str, Any]] = {}

        for lvl in levels:
            try:
                # 2) Check if we should try to load from cache
                path = self.transpiled_circuit_path(backend_alias, lvl)
                tcirc = None
                
                if self.store_transpiled and path.exists() and not force_overwrite:
                    # Load from cached file
                    tcirc = self.load_circuit(path)
                else:
                    # 3) Transpile/compile for this backend+level
                    try:
                        tcirc = backend.get_compiled_circuit(self.main_circuit, optimisation_level=lvl)
                    except Exception as e:
                        raise RuntimeError(f"Failed to compile circuit for {backend_alias} at opt_level {lvl}: {e}")
                    
                    # 4) Cache it only if store_transpiled is True
                    if self.store_transpiled:
                        self.save_circuit(tcirc, path)

                # 5) Extract resources from transpiled circuit
                res = {
                    "n_qubits": tcirc.n_qubits,
                    "n_gates":  tcirc.n_gates,
                    "depth":    tcirc.depth(),
                }
                results[lvl] = res

                # 6) Persist to metadata immediately (always done regardless of store_transpiled)
                self._metadata.set_backend_resources(
                    self.solver_name, self.encoding, backend_alias, lvl, res
                )
                self._metadata.save()

            except Exception as e:
                # Record the error and continue onward
                error_res = {"error": str(e)}
                results[lvl] = error_res
                
                # Log error to metadata (triggers CSV)
                self._metadata.set_backend_resources(
                    self.solver_name, self.encoding, backend_alias, lvl, error_res
                )
                self._metadata.save()

        return results

    def run(
        self,
        backend_alias: str,
        shots: int = 1024,
        force_run: bool = False,
        optimisation_level: int = 1
    ):
        """
        Run the transpiled circuit on the specified backend.

        Args:
            backend_alias: Backend alias to execute the circuit.
            shots: Number of shots for execution.
            force_run: If True, re-transpile and re-run even if cached result exists.
            optimisation_level: Optimisation level for transpilation.

        Returns:
            Result object from backend execution.
        """
        
        # Ensure main circuit is built
        if self.main_circuit is None:
            self.build_main_circuit()
        
        # Phase 2: Use backend from puzzle's attached backends  
        # Backend validation already done in QSudoku.run()
        if hasattr(self.sudoku, '_attached_backends') and backend_alias in self.sudoku._attached_backends:
            backend = self.sudoku._attached_backends[backend_alias]
        else:
            # Fallback to global registry for backward compatibility
            backend = BackendManager.get(backend_alias)
            
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
        """
        Run the main circuit on Aer simulator (no transpilation needed).
        
        Args:
            shots: Number of shots for simulation
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Result from Aer simulation
        """
        # Ensure main circuit is built
        if self.main_circuit is None:
            self.build_main_circuit()
            
        # Import Aer backend locally (not part of global registry)
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
        """
        Create a bar plot of measurement counts with Sudoku-specific enhancements.
        
        This method provides rich visualization for analyzing quantum execution results
        with automatic validation highlighting and summary statistics.
        
        Args:
            counts: Dictionary of measurement outcomes, pytket Result object, or None
            backend_alias: Name of the backend used (for title display)
            shots: Total number of shots (for title and percentage calculation)
            top_n: Show only the top N most frequent outcomes (default: 20)
            show_valid_only: If True, only show outcomes that represent valid Sudoku solutions
            figsize: Figure size tuple (width, height)
            show_summary: If True, print summary statistics (default: True)
            
        Examples:
            # Plot results from Aer simulation
            result = solver.run_aer(shots=1024)
            solver.counts_plot(result, backend_alias="Aer", shots=1024)
            
            # Plot only valid solutions from hardware run
            result = solver.run("ibm_brisbane", opt_level=1, shots=100)
            solver.counts_plot(result, backend_alias="IBM Brisbane", show_valid_only=True)
            
            # Plot without summary statistics
            solver.counts_plot(result, backend_alias="Aer", show_summary=False)
        """
        import matplotlib.pyplot as plt
        
        # Handle different input types
        if counts is None:
            raise ValueError("No counts provided. Pass counts dictionary or run a quantum execution first.")
        
        # Extract counts from pytket Result object if needed
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
                # Convert to string format for validation
                if isinstance(bitstring, tuple):
                    bitstring_str = ''.join(str(bit) for bit in bitstring)
                else:
                    bitstring_str = str(bitstring)
                
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
        
        # Calculate total shots and percentages
        total_shots = shots or sum(counts_dict.values())
        labels = list(top_counts.keys())
        values = list(top_counts.values())
        percentages = [100 * v / total_shots for v in values]
        
        # Convert labels to strings for display and validation
        label_strings = []
        for label in labels:
            if isinstance(label, tuple):
                # Convert tuple to bitstring format
                label_str = ''.join(str(bit) for bit in label)
            elif isinstance(label, str):
                label_str = label
            else:
                label_str = str(label)
            label_strings.append(label_str)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color bars - hybrid approach: blue default, validation colors when analyzing validity
        colors = []
        # Check if we should use validation colors (method exists and we're not filtering to valid only)
        has_validation = hasattr(self, '_is_valid_solution') and callable(getattr(self, '_is_valid_solution', None))
        use_validation_colors = not show_valid_only and has_validation
        
        for i, label_str in enumerate(label_strings):
            if label_str.startswith("Others"):
                colors.append('lightgray')
            elif show_valid_only or not use_validation_colors:
                # Always blue when filtering for valid only, or when not doing validation analysis
                colors.append('steelblue')
            else:
                # Use validation colors only when explicitly analyzing validity
                try:
                    if self._is_valid_solution(label_str):
                        colors.append('steelblue')
                    else:
                        colors.append('lightcoral')
                except Exception:
                    # Fallback to blue if validation fails
                    colors.append('steelblue')
        
        bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7)
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + total_shots*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        ax.set_xlabel('Measurement Outcomes (Bitstrings)')
        ax.set_ylabel('Counts')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(label_strings, rotation=45, ha='right')
        
        # Enhanced title with key information
        title_parts = [f"{self.solver_name} Results"]
        if backend_alias:
            title_parts.append(f"Backend: {backend_alias}")
        else:
            title_parts.append("Backend: Aer Simulator")  # Default for run_aer
        if total_shots:
            title_parts.append(f"Shots: {total_shots}")
        if show_valid_only:
            title_parts.append("(Valid Solutions Only)")
        
        ax.set_title(" | ".join(title_parts))
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend only when using validation colors
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
        
        # Print summary statistics only if requested
        if show_summary:
            valid_count = sum(count for bitstring, count in counts_dict.items() 
                             if self._is_valid_solution(
                                 ''.join(str(bit) for bit in bitstring) if isinstance(bitstring, tuple) else str(bitstring)
                             ))
            print("\nSummary:")
            print(f"Total unique outcomes: {len(counts_dict)}")
            print(f"Total shots: {total_shots}")
            print(f"Valid solutions found: {valid_count} ({100*valid_count/total_shots:.2f}%)")
            most_frequent_key = sorted_counts[0][0]
            most_frequent_str = ''.join(str(bit) for bit in most_frequent_key) if isinstance(most_frequent_key, tuple) else str(most_frequent_key)
            print(f"Most frequent outcome: {most_frequent_str} ({100*sorted_counts[0][1]/total_shots:.2f}%)")

    def _is_valid_solution(self, bitstring):
        """
        Check if a measurement outcome represents a valid Sudoku solution.
        
        This default implementation returns True for all inputs, which means all bars
        will be blue. Subclasses should override this method to implement proper
        validation based on their specific encoding schemes.
        
        Args:
            bitstring: String representation of measurement outcome
            
        Returns:
            bool: True if the bitstring represents a valid solution, False otherwise
            
        Example implementation for subclasses:
            def _is_valid_solution(self, bitstring):
                # Convert bitstring to Sudoku solution
                solution = self.decode_bitstring(bitstring)
                # Check if solution satisfies Sudoku constraints
                return self.validate_sudoku_solution(solution)
        """
        # Default implementation - shows that validation is "implemented" but always returns True
        # This encourages subclasses to override with real validation
        return True

    @staticmethod
    def count_mcx_gates(circuit: Circuit) -> int:
        """
        Count the total number of multi-controlled X (CnX) gates in `circuit`.

        Args:
            circuit: A pytket Circuit in which to count MCX gates.
        Returns:
            The integer count of OpType.CnX gates.
        """
        counts = gate_counts(circuit)
        return counts.get(OpType.CnX, 0)
    
    def export_metadata_csv(self, path: Path) -> None:
        """
        Flatten this solver's JSON metadata and write to the specified CSV path.
        Creates multiple rows (one per solver+encoding+backend+opt_level combination).
        """
        self._metadata.export_full_metadata_csv(path)
    
    def ensure_puzzle_metadata(self):
        """
        Record puzzle-level info to metadata after sudoku instance is fully initialized.
        This should be called after the sudoku board is properly set up.
        """
        self._metadata.ensure_puzzle_fields(
            size=self.size,
            num_missing_cells=self.num_missing_cells,
            board=self.board,
        )
        self._metadata.save()