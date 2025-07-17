"""
ExperimentRunner: Bulk experiment framework for quantum Sudoku solving algorithms.

Provides robust bulk experimentation with automatic CSV logging, bounded memory usage,
comprehensive error handling, and real-time progress tracking.
"""
import csv
import gc
from pathlib import Path
from typing import List, Dict, Type, Union
from .q_sudoku import QSudoku
from .quantum_solver import QuantumSolver
from .backends import BackendManager


class ExperimentRunner:
    """Bulk experiment runner for quantum Sudoku solving algorithms.
    
    Features:
    - Automatic CSV logging via MetadataManager
    - Bounded memory usage with explicit cleanup
    - Comprehensive error handling (including main circuit build failures)
    - Real-time progress tracking by CSV rows written
    - Atomic JSON and CSV persistence
    """
    
    def __init__(
        self,
        solvers: List[Type[QuantumSolver]],
        encodings_map: Dict[Type[QuantumSolver], List[str]],
        backends: List[str],
        opt_levels: List[int],
        subgrid_sizes: List[int],
        num_missing_vals: List[int],
        samples_per_combo: int,
        csv_path: Union[str, Path],
        cache_base: Union[str, Path],
        canonicalize: bool = True,
        cache_main: bool = True,
        cache_transpiled: bool = False,
        show_progress: bool = True,
        progress_interval: int = 10
    ):
        """Initialize experiment runner.
        
        Args:
            solvers: List of solver classes to test
            encodings_map: Map from solver class to list of encodings
            backends: List of backend aliases to test on
            opt_levels: Optimization levels to test
            subgrid_sizes: List of subgrid sizes (2 for 4x4, 3 for 9x9, etc.)
            num_missing_vals: List of missing cell counts to test
            samples_per_combo: Number of puzzle samples per size/missing combination
            csv_path: Path to CSV output file
            cache_base: Base directory for caching circuits
            canonicalize: Whether to canonicalize puzzles
            cache_main: Whether to cache main circuits
            cache_transpiled: Whether to cache transpiled circuits
            show_progress: Whether to show progress updates
            progress_interval: Progress update interval (number of experiments)
        """
        self.solvers = solvers
        self.encodings_map = encodings_map
        self.backends = backends
        self.opt_levels = opt_levels
        self.subgrid_sizes = subgrid_sizes
        self.num_missing_vals = num_missing_vals
        self.samples_per_combo = samples_per_combo
        self.csv_path = Path(csv_path)
        self.cache_base = Path(cache_base)
        self.canonicalize = canonicalize
        self.cache_main = cache_main
        self.cache_transpiled = cache_transpiled
        self.show_progress = show_progress
        self.progress_interval = progress_interval
        
        # Track processed puzzle hashes
        self._seen_hashes = set()
        self._load_seen_hashes_from_csv()
        
        # Progress tracking (counts CSV rows written)
        self._experiments_completed = 0
        self._total_experiments = self._calculate_total()
        
        # Validate backends upfront (fail-fast)
        self._validate_backends()

    def _load_seen_hashes_from_csv(self):
        """Load seen hashes from the existing CSV file."""
        if self.csv_path.exists():
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if "puzzle_hash" in reader.fieldnames:
                    for row in reader:
                        self._seen_hashes.add(row["puzzle_hash"])
        if self.show_progress:
            print(f"Loaded {len(self._seen_hashes)} seen hashes from {self.csv_path}")
    
    def _validate_backends(self):
        """Validate all backend aliases exist in BackendManager."""
        for alias in self.backends:
            try:
                BackendManager.get(alias)
            except ValueError as e:
                available = BackendManager.aliases()
                raise ValueError(
                    f"Backend '{alias}' not found in BackendManager. "
                    f"Available backends: {available}"
                ) from e
    
    def _calculate_total(self) -> int:
        """Calculate total CSV rows for accurate progress tracking.
        
        Each solver/encoding combination generates exactly:
        - 1 main circuit row (success or error)
        - (backends × opt_levels) backend rows (success or error)
        """
        puzzle_count = (len(self.subgrid_sizes) * 
                       len(self.num_missing_vals) * 
                       self.samples_per_combo)
        
        solver_encoding_count = sum(len(encodings) for encodings in self.encodings_map.values())
        rows_per_solver = 1 + (len(self.backends) * len(self.opt_levels))
        
        return puzzle_count * solver_encoding_count * rows_per_solver
    
    def run_batch(self):
        """Run the complete batch of experiments."""
        if self.show_progress:
            print(f"Starting batch: {self._total_experiments} total CSV rows")
            print(f"CSV output: {self.csv_path}")
            print(f"Cache base: {self.cache_base}")
        
        try:
            for size in self.subgrid_sizes:
                for missing in self.num_missing_vals:
                    for sample_idx in range(self.samples_per_combo):
                        self._run_single_puzzle(size, missing, sample_idx)
                        
        except KeyboardInterrupt:
            print(f"\nInterrupted after {self._experiments_completed} CSV rows")
            raise
        except Exception as e:
            print(f"Failed after {self._experiments_completed} CSV rows: {e}")
            raise
        
        if self.show_progress:
            print(f"Completed: {self._experiments_completed} CSV rows")
    
    def _run_single_puzzle(self, size: int, missing: int, sample_idx: int):
        """Process one puzzle through all solver/backend combinations."""
        qs = QSudoku.generate(
            subgrid_size=size,
            num_missing_cells=missing,
            canonicalize=self.canonicalize,
            cache_base=self.cache_base,
            csv_path=self.csv_path
        )
        
        puzzle_hash = qs.get_hash()
        if puzzle_hash in self._seen_hashes:
            if self.show_progress:
                print(f"Skipping duplicate puzzle with hash: {puzzle_hash}")
            return  # Skip this puzzle
        
        try:
            # Attach all backends once per puzzle
            for alias in self.backends:
                qs.attach_backend(alias)
            
            # Run all solver/encoding combinations
            for SolverCls in self.solvers:
                for encoding in self.encodings_map[SolverCls]:
                    self._run_single_solver(qs, SolverCls, encoding)
                    print(f"Completed {SolverCls.__name__} with encoding '{encoding}' for puzzle {puzzle_hash}")
                    
        finally:
            self._seen_hashes.add(puzzle_hash)  # Mark this hash as seen
            # Cleanup: free puzzle memory only
            del qs
            gc.collect()
    
    def _run_single_solver(self, qs: QSudoku, SolverCls: Type[QuantumSolver], encoding: str):
        """Run one solver/encoding through all backends."""
        try:
            # Set solver (CSV logging already enabled via puzzle csv_path)
            qs.set_solver(
                SolverCls,
                encoding=encoding,
                store_transpiled=self.cache_transpiled
            )
                        
            # Build main circuit with error logging
            try:
                qs.build_circuit()  # Auto-logs main circuit CSV row on success
                self._experiments_completed += 1
                self._maybe_show_progress()
            except Exception as e:
                # Log main circuit build error to CSV
                self._log_main_circuit_error(qs, SolverCls.__name__, encoding, str(e))
                self._experiments_completed += 1
                self._maybe_show_progress()
                # Continue to transpilation (may still work with dummy circuit)
            
            # Transpile for each backend (auto-logs backend CSV rows)
            for alias in self.backends:
                qs.transpile(alias, self.opt_levels)
                
                # Each opt_level generates one CSV row
                self._experiments_completed += len(self.opt_levels)
                self._maybe_show_progress()
                
        except Exception as e:
            if self.show_progress:
                print(f"Solver error {SolverCls.__name__}/{encoding}: {e}")
            # Count remaining experiments as completed (failed)
            batch_size = 1 + len(self.backends) * len(self.opt_levels)
            done_mod = self._experiments_completed % batch_size
            self._experiments_completed += (batch_size - done_mod)
        finally:
            # Single cleanup point
            qs.drop_solver()
    
    def _log_main_circuit_error(self, qs: QSudoku, solver_name: str, encoding: str, error: str):
        """Log main circuit build error directly to CSV."""
        row_data = {
            "puzzle_hash": qs.get_hash(),
            "size": qs.board_size,
            "num_missing_cells": qs.num_missing_cells,
            "solver_name": solver_name,
            "encoding": encoding,
            "backend_alias": None,  # Main circuit has no backend
            "opt_level": None,
            "main_n_qubits": None,
            "main_n_gates": None,
            "main_n_mcx_gates": None,
            "main_depth": None,
            "backend_n_qubits": None,
            "backend_n_gates": None,
            "backend_depth": None,
            "error": error
        }
        qs._metadata._append_csv_row(row_data)
    
    def _maybe_show_progress(self):
        """Show progress update if interval reached."""
        if (self.show_progress and 
            self._experiments_completed % self.progress_interval == 0):
            progress_pct = (self._experiments_completed / self._total_experiments) * 100
            print(f"Progress: {self._experiments_completed}/{self._total_experiments} CSV rows ({progress_pct:.1f}%)")


def run_experiment_batch(
    solvers_config: Dict[Type[QuantumSolver], List[str]],
    backends: List[str],
    csv_path: str = "sudoku_experiments.csv",
    cache_base: str = "./experiment_cache",
    puzzle_sizes: List[int] = [2],  # 4x4 puzzles by default
    missing_cells: List[int] = [3, 5, 7],
    samples_per_combo: int = 5,
    opt_levels: List[int] = [0, 1, 2],
    **kwargs
) -> Path:
    """Convenience function to run a batch of experiments.
    
    Args:
        solvers_config: Dict mapping solver classes to their encoding lists
        backends: List of backend aliases
        csv_path: Output CSV file path
        cache_base: Cache directory for circuits
        puzzle_sizes: List of subgrid sizes to test
        missing_cells: List of missing cell counts
        samples_per_combo: Puzzle samples per size/missing combination
        opt_levels: Optimization levels to test
        **kwargs: Additional arguments for ExperimentRunner
    
    Returns:
        Path to the generated CSV file
    """
    runner = ExperimentRunner(
        solvers=list(solvers_config.keys()),
        encodings_map=solvers_config,
        backends=backends,
        opt_levels=opt_levels,
        subgrid_sizes=puzzle_sizes,
        num_missing_vals=missing_cells,
        samples_per_combo=samples_per_combo,
        csv_path=csv_path,
        cache_base=cache_base,
        **kwargs
    )
    
    runner.run_batch()
    return Path(csv_path)
