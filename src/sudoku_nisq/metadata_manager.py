import os
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping
import warnings

class MetadataManager:
    """Manages JSON persistence and caching for quantum Sudoku experiment metadata.

    This class provides atomic operations for storing, retrieving, and managing
    puzzle metadata including circuit resources, solver configurations, and
    execution results. It implements lazy loading and atomic writes to ensure
    data consistency across multiple experiments.

    The metadata is organized hierarchically:

    * Puzzle-level: hash, size, missing cells, board configuration
    * Solver-level: encoding types, circuit resources
    * Backend-level: optimization levels, transpiled resources

    :ivar Path cache_base: Base directory for cache storage.
    :ivar str puzzle_hash: Unique identifier for the puzzle configuration.
    :ivar bool sort_keys: Whether to sort JSON keys for consistent formatting on save.
    :ivar Path metadata_path: Full path to the ``metadata.json`` file.

    Example:
        .. code-block:: python

            from pathlib import Path

            # Initialize metadata manager for a specific puzzle
            cache_dir = Path(".quantum_cache")
            puzzle_hash = "abc123def456"
            metadata = MetadataManager(cache_dir, puzzle_hash)

            # Set puzzle configuration
            metadata.ensure_puzzle_fields(
                size=9,
                num_missing_cells=45,
                board=[[0, 1, 2], [3, 0, 4], [5, 6, 0]]
            )

            # Store circuit resources
            metadata.set_main_circuit_resources(
                solver_name="ExactCoverQuantumSolver",
                encoding="pattern",
                resources={"n_qubits": 81, "n_gates": 1024, "depth": 256}
            )

            # Persist to disk
            metadata.save()
    """
    __slots__ = (
        "cache_base",
        "puzzle_hash",
        "sort_keys",
        "metadata_path",
        "_data",
        "_dirty",
    )
    
    def __init__(
        self,
        cache_base: Path,
        puzzle_hash: str,
        *,
        sort_keys: bool = True,
    ):
        """Initialize the metadata manager for a specific puzzle.
        
        Sets up the file paths and initializes internal state for managing
        metadata persistence. The actual metadata loading is deferred until
        the first access (lazy loading).
        
        Args:
            cache_base (Path): Base directory where cache files are stored.
                Will be created if it doesn't exist.
            puzzle_hash (str): Unique hexadecimal hash identifying the puzzle
                configuration.
            sort_keys (bool, optional): Whether to sort JSON keys when writing
                to disk for consistent formatting. Defaults to True.
                
        Example:
            .. code-block:: python

                from pathlib import Path

                cache_dir = Path(".quantum_cache")
                puzzle_hash = "d6d5713893ca50df092c48b099d9eeb4e1a22fbd"

                manager = MetadataManager(cache_dir, puzzle_hash, sort_keys=True)
                print(f"Metadata will be stored at: {manager.metadata_path}")
        """
        # Base folder (e.g. ".quantum_solver_cache")
        self.cache_base    = cache_base
        self.puzzle_hash   = puzzle_hash
        self.sort_keys     = sort_keys

        # Full path to metadata.json for this puzzle
        self.metadata_path = self.cache_base / self.puzzle_hash / "metadata.json"

        # In-memory store & dirty flag
        self._data: dict[str, Any] | None = None
        self._dirty = False

    def load(self) -> dict[str, Any]:
        """Lazily load metadata from disk into memory.
        
        Reads the metadata.json file for this puzzle if it exists, or initializes
        an empty dictionary if the file doesn't exist. Implements error recovery
        by resetting corrupted files to empty state with a warning.
        
        Returns:
            Dict[str, Any]: The complete metadata dictionary for this puzzle,
                containing puzzle configuration, solver data, and resource metrics.
                
        Warns:
            UserWarning: If the metadata file exists but contains invalid JSON,
                the file is considered corrupted and reset to empty state.
                
        Example:
            .. code-block:: python

                metadata = MetadataManager(cache_dir, puzzle_hash)

                # First call loads from disk (or creates empty dict)
                data = metadata.load()

                # Subsequent calls return cached in-memory data
                same_data = metadata.load()
                assert data is same_data  # Same object reference

                print(f"Puzzle size: {data.get('size', 'unknown')}")
                print(f"Solvers: {list(data.get('solvers', {}).keys())}")
        """
        if self._data is None:
            if self.metadata_path.exists():
                try:
                    self._data = json.loads(self.metadata_path.read_text())
                except json.JSONDecodeError:
                    warnings.warn(
                        f"{self.metadata_path!s} is corrupt; resetting metadata to empty",
                        UserWarning,
                        stacklevel=2
                    )
                    self._data = {}
            else:
                self._data = {}
        return self._data

    def save(self) -> None:
        """Atomically persist metadata to disk if changes have been made.
        
        Uses a temporary file with atomic rename to ensure data consistency
        even if the process is interrupted during writing. Only performs I/O
        if the metadata has been modified since the last save.
        
        The atomic write process:
        1. Create temporary file in same directory
        2. Write JSON data and flush to disk
        3. Atomically rename temp file to final location
        
        Raises:
            OSError: If unable to create directories or write to disk.
            PermissionError: If lacking write permissions to cache directory.
            
        Example:
            .. code-block:: python

                metadata = MetadataManager(cache_dir, puzzle_hash)

                # Make some changes
                metadata.ensure_puzzle_fields(size=9, num_missing_cells=45, board=board)

                # Persist changes (only writes if dirty flag is set)
                metadata.save()

                # Subsequent saves do nothing if no changes made
                metadata.save()  # No-op
        """
        if not self._dirty:
            return

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        dir_ = self.metadata_path.parent

        # Create a .tmp file, write+fsync, then rename it in place
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp",
            dir=dir_, delete=False, encoding="utf-8"
        ) as tf:
            json.dump(self._data, tf, indent=2, sort_keys=self.sort_keys)
            tf.flush()
            os.fsync(tf.fileno())
            tmp_name = tf.name

        os.replace(tmp_name, self.metadata_path)
        self._dirty = False
        
    def unload(self) -> None:
        """Release in-memory metadata to free RAM.
        
        Drops the cached metadata dictionary from memory without saving.
        The next call to load() will re-read from disk. This is useful for
        long-running processes that need to manage memory usage.
        
        Note:
            Any unsaved changes will be lost. Call save() before unload()
            if you want to persist modifications.
            
        Example:
            .. code-block:: python

                metadata = MetadataManager(cache_dir, puzzle_hash)

                # Load and modify data
                data = metadata.load()
                data["new_field"] = "some_value"

                # Save changes before unloading
                metadata.save()

                # Free memory
                metadata.unload()

                # Next load() will read from disk again
                reloaded_data = metadata.load()
                assert "new_field" in reloaded_data
        """
        self._data = None
        self._dirty = False

    def ensure_puzzle_fields(
        self,
        *,
        size: int,
        num_missing_cells: int,
        board: list[list[int]],
    ) -> None:
        """Set or update puzzle-level metadata fields.
        
        Updates the core puzzle configuration in metadata, marking the data
        as dirty if any values have changed. The board matrix is serialized
        to JSON string format for storage.
        
        Args:
            size (int): Sudoku grid size (typically 9 for standard Sudoku).
            num_missing_cells (int): Number of empty cells in the puzzle.
            board (List[List[int]]): 2D matrix representing the puzzle state,
                where 0 indicates empty cells.
                
        Example:
            .. code-block:: python

                # Standard 9x9 Sudoku with some filled cells
                board = [
                    [1, 0, 3, 0, 0, 5, 0, 8, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    # ... more rows
                ]

                metadata.ensure_puzzle_fields(
                    size=9,
                    num_missing_cells=45,
                    board=board
                )

                # Changes are marked as dirty but not yet saved
                metadata.save()  # Persist to disk
        """
        md = self.load()
        # Convert the board (matrix) to a string representation
        board_str = json.dumps(board)
        fields = {
            "puzzle_hash":       self.puzzle_hash,
            "size":              size,
            "num_missing_cells": num_missing_cells,
            "board":             board_str,  # Save as string
        }
        for k, v in fields.items():
            if md.get(k) != v:
                md[k] = v
                self._dirty = True

    def set_main_circuit_resources(
        self,
        solver_name: str,
        encoding: str,
        resources: Mapping[str, int],
    ) -> None:
        """Store quantum circuit resource metrics for a solver's main circuit.
        
        Records the resource requirements of the logical quantum circuit before
        hardware-specific compilation. This data is used for algorithm analysis
        and comparison across different solver approaches.
        
        Args:
            solver_name (str): Name of the quantum solver class (e.g.,
                "ExactCoverQuantumSolver", "BacktrackingQuantumSolver").
            encoding (str): Encoding strategy used (e.g., "pattern", "simple").
            resources (Mapping[str, int]): Dictionary of resource metrics including:
                - n_qubits: Number of qubits required
                - n_gates: Total gate count in the circuit
                - n_mcx_gates: Number of multi-controlled X gates
                - depth: Circuit depth (critical path length)
                
        Example:
            .. code-block:: python

                # Record resources for exact cover solver with pattern encoding
                resources = {
                    "n_qubits": 81,
                    "n_gates": 2048,
                    "n_mcx_gates": 45,
                    "depth": 512
                }

                metadata.set_main_circuit_resources(
                    solver_name="ExactCoverQuantumSolver",
                    encoding="pattern",
                    resources=resources
                )

                metadata.save()
        """
        md = self.load()
        # Get (or create) the solver section
        solver_section = md.setdefault("solvers", {}) \
                        .setdefault(solver_name, {})
        # Get (or create) the encoding sub-section
        encoding_section = solver_section.setdefault("encodings", {}) \
                                        .setdefault(encoding, {})
        # Now write the main_circuit_resources if they’ve changed
        if encoding_section.get("main_circuit_resources") != resources:
            encoding_section["main_circuit_resources"] = dict(resources)
            self._dirty = True

    def set_backend_resources(
        self,
        solver_name: str,
        encoding: str,
        backend_alias: str,
        opt_level: int,
        resources: Mapping[str, int | str],
    ) -> None:
        """Store hardware-specific circuit resource metrics after transpilation.
        
        Records the resource requirements after the circuit has been compiled
        for a specific quantum hardware backend. This includes the effects of
        gate decomposition, routing, and optimization.
        
        Args:
            solver_name (str): Name of the quantum solver class.
            encoding (str): Encoding strategy used (e.g., "pattern", "simple").
            backend_alias (str): Alias of the quantum backend (e.g., "ibm_brisbane").
            opt_level (int): Compiler optimization level (typically 0-3).
            resources (Mapping[str, Union[int, str]]): Transpiled resource metrics:
                - n_qubits: Physical qubits required after routing
                - n_gates: Gate count after decomposition and optimization  
                - depth: Circuit depth after compilation
                - Additional backend-specific metrics
                
        Example:
            .. code-block:: python

                # Record resources after transpiling for IBM hardware
                transpiled_resources = {
                    "n_qubits": 127,      # Physical qubits on hardware
                    "n_gates": 3156,      # After gate decomposition
                    "depth": 892,         # After routing and optimization
                    "optimization_level": 2
                }

                metadata.set_backend_resources(
                    solver_name="ExactCoverQuantumSolver",
                    encoding="pattern",
                    backend_alias="ibm_brisbane",
                    opt_level=2,
                    resources=transpiled_resources
                )

                metadata.save()
        """
        md = self.load()
        solvers = md.setdefault("solvers", {})
        sol_md  = solvers.setdefault(solver_name, {})
        encs    = sol_md.setdefault("encodings", {})
        enc_md  = encs.setdefault(encoding, {})
        backends = enc_md.setdefault("backends", {})
        be = backends.setdefault(backend_alias, {})

        lvl = str(opt_level)
        if be.get(lvl) != resources:
            be[lvl] = dict(resources)
            self._dirty = True

    def remove_solver(self, solver_name: str) -> None:
        """Remove all metadata entries for a specific solver.
        
        Deletes all stored data for the specified solver including circuit
        resources, backend configurations, and execution results. This is
        useful for cleaning up obsolete solver data or when algorithms change.
        
        Args:
            solver_name (str): Name of the quantum solver to remove completely.
                
        Example:
            .. code-block:: python

                # Remove outdated solver data
                metadata.remove_solver("OldBacktrackingQuantumSolver")

                # Clean up multiple solvers
                obsolete_solvers = ["SolverV1", "ExperimentalSolver"]
                for solver in obsolete_solvers:
                    metadata.remove_solver(solver)

                metadata.save()
        """
        md = self.load()
        solvers = md.get("solvers", {})
        if solver_name in solvers:
            solvers.pop(solver_name)
            self._dirty = True

    def get_solver_data(self, solver_name: str) -> dict[str, Any] | None:
        """Retrieve the complete metadata for a specific solver.
        
        Returns the raw metadata dictionary containing all encodings, backends,
        and resource data for the specified solver. Useful for detailed
        analysis and debugging.
        
        Args:
            solver_name (str): Name of the quantum solver to retrieve data for.
            
        Returns:
            Optional[Dict[str, Any]]: Complete solver metadata dictionary, or
                None if the solver has no stored data. The structure includes:
                - encodings: Dict of encoding strategies and their data
                - Each encoding contains: main_circuit_resources, backends
                - Backend data includes optimization levels and metrics
                
        Example:
            .. code-block:: python

                # Get complete data for analysis
                solver_data = metadata.get_solver_data("ExactCoverQuantumSolver")

                if solver_data:
                    encodings = solver_data.get("encodings", {})
                    for enc_name, enc_data in encodings.items():
                        circuit_res = enc_data.get("main_circuit_resources", {})
                        print(f"{enc_name}: {circuit_res.get('n_qubits')} qubits")

                        backends = enc_data.get("backends", {})
                        for backend_name in backends:
                            print(f"  - Available on {backend_name}")
                else:
                    print(f"No data found for {solver_name}")
        """
        md = self.load()
        return md.get("solvers", {}).get(solver_name)

    def get_resource_summary(self) -> dict[str, Any]:
        """Generate a user-friendly summary of all stored resource data.
        
        Creates a structured overview of puzzle configuration and quantum
        resource requirements across all solvers, encodings, and backends.
        This method provides an easy way to analyze and compare different
        quantum algorithm approaches.
        
        Returns:
            Dict[str, Any]: Comprehensive resource summary with structure:
                - puzzle_info: Basic puzzle metadata (hash, size, missing cells)
                - solvers: Dict mapping solver names to their resource data
                  - Each solver contains encoding strategies
                  - Each encoding has main_circuit resources and backend data
                  - Backend data includes transpiled resources by optimization level
                
        Example:
            .. code-block:: python

                summary = metadata.get_resource_summary()
                puzzle_info = summary['puzzle_info']
                print(puzzle_info)

                # Iterate solvers and encodings succinctly
                for solver_name, solver_data in summary['solvers'].items():
                    for encoding, enc_data in solver_data.items():
                        print(solver_name, encoding, enc_data['main_circuit'].get('n_qubits'))
        """
        md = self.load()
        
        summary = {
            "puzzle_info": {
                "hash": md.get("puzzle_hash"),
                "size": md.get("size"),
                "num_missing_cells": md.get("num_missing_cells")
            },
            "solvers": {}
        }
        
        solvers = md.get("solvers", {})
        for solver_name, solver_data in solvers.items():
            summary["solvers"][solver_name] = {}
            # Type assertion to help mypy understand this is a dict
            solver_summary = summary["solvers"][solver_name]
            assert isinstance(solver_summary, dict)
            encodings = solver_data.get("encodings", {})
            
            for encoding_name, encoding_data in encodings.items():
                encoding_summary = {
                    "main_circuit": encoding_data.get("main_circuit_resources", {}),
                    "backends": {}
                }
                
                backends = encoding_data.get("backends", {})
                for backend_alias, backend_data in backends.items():
                    encoding_summary["backends"][backend_alias] = backend_data
                
                solver_summary[encoding_name] = encoding_summary
        
        return summary
