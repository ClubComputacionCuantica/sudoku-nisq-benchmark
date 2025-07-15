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