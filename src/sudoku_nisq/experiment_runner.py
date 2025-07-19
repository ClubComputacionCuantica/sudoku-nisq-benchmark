"""
ExperimentRunner: Comprehensive quantum solver benchmarking automation.

Provides crash-safe experiment execution with atomic progress tracking,
memory hygiene, automated backend management, and detailed performance timing.

Key design principles:
- Crash-safe: persistent "seen" index with atomic writes
- Memory hygiene: automatic cleanup after each puzzle/solver  
- Backend efficiency: direct access from global registry, no unnecessary attachments
- Performance tracking: detailed timing for circuit building and transpilation
- Configurable: supports multiple solvers, encodings, backends, optimization levels

Usage:
    # Define solver configuration
    from sudoku_nisq.exact_cover_solver import ExactCoverQuantumSolver
    
    solvers_config = {
        ExactCoverQuantumSolver: ["simple", "pattern"]
    }
    
    # Authenticate and register backends
    from sudoku_nisq.backends import BackendManager
    
    # IBM authentication and device registration
    BackendManager.authenticate_ibm(
        api_token=api_token,
        instance=instance
    )
    BackendManager.add_ibm_device(device="ibm_torino", alias="ibm_torino")
    BackendManager.add_ibm_device(device="ibm_brisbane", alias="ibm_brisbane")
    
    # Quantinuum authentication and device registration
    BackendManager.authenticate_quantinuum()
    BackendManager.add_quantinuum_device(device="H1-1", alias="quantinuum_h1")
    
    # Check registered backends
    available_backends = BackendManager.aliases()
    print(f"Registered backends: {available_backends}")
    
    # Create and run experiment
    runner = ExperimentRunner(
        solvers=list(solvers_config.keys()),
        encodings_map=solvers_config,
        backends=["ibm_brisbane", "quantinuum_h1"],
        opt_levels=[0, 1],
        subgrid_sizes=[2, 3],
        num_missing_vals=[2, 4, 6],
        samples_per_combo=5,
        cache_base="./experiment_cache",
        canonicalize=True,
        cache_transpiled=False
    )
    
    runner.run_batch()
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Type, Any

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.backends import BackendManager


class ExperimentRunner:
    """
    Comprehensive quantum solver benchmarking with crash-safe progress tracking.
    
    Features:
    - Crash-safe execution with atomic "seen" index persistence
    - Memory hygiene with automatic cleanup after each puzzle/solver
    - Direct backend access from global registry (no per-puzzle attachments)
    - Configurable solver/encoding combinations and backend parameters
    - Detailed timing measurement for circuit building and transpilation operations
    - Progress and timing summary reporting
    
    Attributes:
        solvers: List of solver classes to benchmark
        encodings_map: Map from solver classes to supported encodings
        backends: List of backend aliases to test
        opt_levels: List of optimization levels to test
        subgrid_sizes: List of puzzle subgrid sizes to test
        num_missing_vals: List of missing cell counts to test
        samples_per_combo: Number of puzzle samples per size/difficulty combination
        cache_base: Base directory for caching and metadata
        canonicalize: Whether to canonicalize generated puzzles
        cache_transpiled: Whether to cache transpiled circuits
        seen: In-memory "seen" index tracking completed work
    """
    
    def __init__(
        self,
        solvers: List[Type],
        encodings_map: Dict[Type, List[str]],
        backends: List[str],
        opt_levels: List[int],
        subgrid_sizes: List[int],
        num_missing_vals: List[int],
        samples_per_combo: int,
        cache_base: str = "./experiment_cache",
        canonicalize: bool = True,
        cache_transpiled: bool = False,
    ):
        """
        Initialize ExperimentRunner with configuration.
        
        Note: Backend authentication must be done separately using BackendManager
        before creating the ExperimentRunner instance.
        
        Args:
            solvers: List of quantum solver classes to benchmark
            encodings_map: Dictionary mapping solver classes to their supported encodings
            backends: List of backend aliases to test (must be registered in BackendManager)
            opt_levels: List of optimization levels to test
            subgrid_sizes: List of subgrid sizes (e.g., [2, 3] for 4x4 and 9x9 puzzles)
            num_missing_vals: List of missing cell counts to test per puzzle size
            samples_per_combo: Number of puzzle samples to generate per size/difficulty combo
            cache_base: Base directory for caching circuits and metadata
            canonicalize: Whether to canonicalize puzzle boards after generation
            cache_transpiled: Whether to cache transpiled circuits to disk
            
        Raises:
            ValueError: If configuration is invalid or backends not registered
        """
        # Store configuration
        self.solvers = solvers
        self.encodings_map = encodings_map
        self.backends = backends
        self.opt_levels = opt_levels
        self.subgrid_sizes = subgrid_sizes
        self.num_missing_vals = num_missing_vals
        self.samples_per_combo = samples_per_combo
        self.cache_base = Path(cache_base)
        self.canonicalize = canonicalize
        self.cache_transpiled = cache_transpiled
        
        # Validate configuration
        self._validate_config()
        
        # Validate that all requested backends are properly registered
        self._validate_backends()
        
        # Load or initialize crash-safe "seen" index
        self.index_path = self.cache_base / "seen_index.json"
        self.seen = self._load_seen_index()
        
        # Ensure cache directory exists
        self.cache_base.mkdir(parents=True, exist_ok=True)
        
    def _validate_config(self) -> None:
        """Validate experiment configuration parameters."""
        # Validate solvers and encodings
        if not self.solvers:
            raise ValueError("At least one solver must be specified")
            
        for solver_cls in self.solvers:
            if solver_cls not in self.encodings_map:
                raise ValueError(f"Solver {solver_cls.__name__} not found in encodings_map")
            if not self.encodings_map[solver_cls]:
                raise ValueError(f"No encodings specified for solver {solver_cls.__name__}")
                
        # Validate backends
        if not self.backends:
            raise ValueError("At least one backend must be specified")
                
        # Validate other parameters
        if not self.opt_levels:
            raise ValueError("At least one optimization level must be specified")
        if not self.subgrid_sizes:
            raise ValueError("At least one subgrid size must be specified")
        if not self.num_missing_vals:
            raise ValueError("At least one num_missing_cells value must be specified")
        if self.samples_per_combo < 1:
            raise ValueError("samples_per_combo must be at least 1")
            
    def _validate_backends(self) -> None:
        """Validate that all requested backends are properly registered."""
        for alias in self.backends:
            try:
                BackendManager.validate_alias(alias)
            except ValueError as e:
                raise ValueError(f"Backend validation failed for '{alias}': {e}") from e
                
    def _load_seen_index(self) -> Dict[str, Any]:
        """Load the crash-safe "seen" index from disk or return empty dict."""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    seen = json.load(f)
                print(f"Loaded seen index with {len(seen)} puzzle hashes")
                return seen
            else:
                print("• No existing seen index found, starting fresh")
                return {}
        except Exception as e:
            print(f"Failed to load seen index, starting fresh: {e}")
            return {}
            
    def _save_seen_index(self) -> None:
        """Atomically save the "seen" index to disk."""
        try:
            # Write to temporary file first
            tmp_path = self.cache_base / ".seen.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(self.seen, f, indent=2, sort_keys=True)
                
            # Atomic rename
            tmp_path.rename(self.index_path)
            
        except Exception as e:
            print(f"Failed to save seen index: {e}")
            
    def _timed_transpile(self, qs: QSudoku, backend_alias: str, opt_level: int) -> tuple[Any, float]:
        """
        Transpile circuit with timing measurement.
        
        Args:
            qs: QSudoku instance with active solver
            backend_alias: Backend alias for transpilation
            opt_level: Optimization level
            
        Returns:
            Tuple of (transpilation_result, elapsed_seconds)
        """
        # Get backend directly from global registry (no need to attach to QSudoku)
        backend = BackendManager.get(backend_alias)
        
        # Ensure we don't accidentally trigger any circuit building during transpilation
        start_time = time.time()
        if qs._solver is None:
            raise ValueError("No solver configured on QSudoku instance")
        result = qs._solver.transpile_and_analyze(backend, backend_alias, opt_level)
        elapsed = time.time() - start_time
        return result, elapsed
        
    def _timed_build_circuit(self, qs: QSudoku) -> float:
        """
        Build quantum circuit with timing measurement.
        
        Args:
            qs: QSudoku instance with active solver
            
        Returns:
            Elapsed seconds for circuit building
        """
        start_time = time.time()
        qs.build_circuit()
        elapsed = time.time() - start_time
        return elapsed
        
    def run_batch(self) -> None:
        """
        Execute the complete experiment batch with crash-safe progress tracking.
        
        Iterates through all combinations of puzzle sizes, difficulties, and samples,
        generating puzzles and benchmarking all configured solver/encoding/backend
        combinations. Progress is saved after each puzzle to enable crash recovery.
        """
        total_puzzles = len(self.subgrid_sizes) * len(self.num_missing_vals) * self.samples_per_combo
        puzzle_count = 0
        
        print("Starting experiment batch:")
        print(f"   • {len(self.solvers)} solvers × {sum(len(encodings) for encodings in self.encodings_map.values())} encodings")
        print(f"   • {len(self.backends)} backends × {len(self.opt_levels)} optimization levels")  
        print(f"   • {total_puzzles} total puzzles")
        print(f"   • Cache: {self.cache_base}")
        print()
        
        for size in self.subgrid_sizes:
            for missing in self.num_missing_vals:
                for sample_idx in range(self.samples_per_combo):
                    puzzle_count += 1
                    print(f"[{puzzle_count}/{total_puzzles}] Processing puzzle: {size*size}×{size*size}, {missing} missing, sample {sample_idx + 1}")
                    
                    try:
                        self._run_puzzle(size, missing, sample_idx)
                    except Exception as e:
                        print(f"Puzzle failed: {e}")
                        continue
                        
                    # Save progress after each puzzle
                    self._save_seen_index()
                    
        # Final save
        self._save_seen_index()
        print("Experiment batch completed successfully!")
        
    def _run_puzzle(self, size: int, missing: int, sample_idx: int) -> None:
        """
        Process a single puzzle through all solver/encoding/backend combinations.
        
        Args:
            size: Subgrid size (e.g., 2 for 4x4 puzzle, 3 for 9x9 puzzle)
            missing: Number of missing cells in the puzzle
            sample_idx: Sample index for this size/missing combination
        """
        # Generate puzzle
        qs = QSudoku.generate(
            subgrid_size=size,
            num_missing_cells=missing,
            canonicalize=self.canonicalize,
            cache_base=str(self.cache_base)
        )
        
        puzzle_hash = qs.get_hash()
        solver_map = self.seen.setdefault(puzzle_hash, {})
        
        print(f"Puzzle hash: {puzzle_hash[:8]}...")
        
        try:
            # Process each solver and encoding combination
            for solver_cls in self.solvers:
                solver_name = solver_cls.__name__
                encoding_map = solver_map.setdefault(solver_name, {})
                
                for encoding in self.encodings_map[solver_cls]:
                    backend_map = encoding_map.setdefault(encoding, {})
                    
                    print(f"{solver_name} ({encoding})")
                    
                    try:
                        # Set solver with configuration flags
                        qs.set_solver(
                            solver_cls,
                            encoding=encoding,
                            store_transpiled=self.cache_transpiled
                        )
                        
                        # Build circuit once per solver/encoding - BEFORE any backend processing
                        # Use a more explicit check to prevent any possibility of duplicate builds
                        circuit_already_built = backend_map.get("__built__", False)
                        
                        if not circuit_already_built:
                            print("Building circuit...")
                            build_duration = self._timed_build_circuit(qs)
                            backend_map["__built__"] = True
                            backend_map["build_time"] = build_duration
                            print(f"Circuit built in {build_duration:.2f}s")
                        else:
                            print("Skipping circuit build - already done")
                            
                        # NOW transpile for each backend and optimization level
                        for backend_alias in self.backends:
                            opt_map = backend_map.setdefault(backend_alias, {})
                            
                            for opt_level in self.opt_levels:
                                if opt_level not in opt_map:
                                    print(f"Transpiling {backend_alias} (opt={opt_level})...")
                                    
                                    try:
                                        _, duration = self._timed_transpile(qs, backend_alias, opt_level)
                                        opt_map[opt_level] = {"duration": duration}
                                        print(f"Completed in {duration:.2f}s")
                                    except Exception as e:
                                        opt_map[opt_level] = {"error": str(e)}
                                        print(f"Failed: {e}")
                                else:
                                    print(f"Skipping {backend_alias} (opt={opt_level}) - already done")
                                    
                        # Clean up solver after all transpilations
                        qs.drop_solver()
                        
                    except Exception as e:
                        print(f"Solver {solver_name} ({encoding}) failed: {e}")
                        # Still try to clean up
                        try:
                            qs.drop_solver()
                        except Exception:
                            pass
                        continue
                        
        finally:
            # Puzzle-level cleanup
            try:
                del qs
                gc.collect()
            except Exception as e:
                print(f"Cleanup warning: {e}")
                
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get a summary of experiment progress.
        
        Returns:
            Dictionary containing progress statistics
        """
        total_puzzles = 0
        completed_puzzles = 0
        total_configs = 0
        completed_configs = 0
        
        # Calculate expected configs per puzzle: sum of (encodings per solver) * backends * opt_levels
        expected_configs_per_puzzle = (
            sum(len(encodings) for encodings in self.encodings_map.values()) *
            len(self.backends) * 
            len(self.opt_levels)
        )
        
        for size in self.subgrid_sizes:
            for missing in self.num_missing_vals:
                for _ in range(self.samples_per_combo):
                    total_puzzles += 1
                    
        for puzzle_hash, solver_map in self.seen.items():
            puzzle_complete = True
            puzzle_configs = 0
            
            for solver_name, encoding_map in solver_map.items():
                for encoding, backend_map in encoding_map.items():
                    if "__built__" in backend_map:
                        for backend_alias in self.backends:
                            if backend_alias in backend_map:
                                for opt_level in self.opt_levels:
                                    total_configs += 1
                                    puzzle_configs += 1
                                    if opt_level in backend_map[backend_alias]:
                                        completed_configs += 1
                                    else:
                                        puzzle_complete = False
                            else:
                                puzzle_complete = False
                                
            if puzzle_complete and puzzle_configs == expected_configs_per_puzzle:
                completed_puzzles += 1
                
        return {
            "total_puzzles": total_puzzles,
            "completed_puzzles": completed_puzzles,
            "puzzle_progress": f"{completed_puzzles}/{total_puzzles}",
            "total_configs": total_configs,
            "completed_configs": completed_configs,
            "config_progress": f"{completed_configs}/{total_configs}",
            "puzzle_completion_rate": completed_puzzles / total_puzzles if total_puzzles > 0 else 0,
            "config_completion_rate": completed_configs / total_configs if total_configs > 0 else 0,
        }
        
    def get_timing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of timing statistics from completed work.
        
        Returns:
            Dictionary containing timing statistics for build and transpilation
        """
        build_times = []
        transpile_times = []
        timing_by_solver: dict[str, dict[str, list[float]]] = {}
        timing_by_backend: dict[str, list[float]] = {}
        
        for puzzle_hash, solver_map in self.seen.items():
            for solver_name, encoding_map in solver_map.items():
                solver_times = timing_by_solver.setdefault(solver_name, {"build": [], "transpile": []})
                
                for encoding, backend_map in encoding_map.items():
                    # Circuit build timing
                    if "build_time" in backend_map:
                        build_time = backend_map["build_time"]
                        build_times.append(build_time)
                        solver_times["build"].append(build_time)
                    
                    # Transpilation timing
                    for backend_alias in self.backends:
                        if backend_alias in backend_map:
                            backend_times = timing_by_backend.setdefault(backend_alias, [])
                            
                            for opt_level in self.opt_levels:
                                if opt_level in backend_map[backend_alias]:
                                    opt_data = backend_map[backend_alias][opt_level]
                                    if "duration" in opt_data:
                                        transpile_time = opt_data["duration"]
                                        transpile_times.append(transpile_time)
                                        solver_times["transpile"].append(transpile_time)
                                        backend_times.append(transpile_time)
        
        summary = {
            "build_times": {
                "count": len(build_times),
                "total": sum(build_times),
                "avg": sum(build_times) / len(build_times) if build_times else 0,
                "min": min(build_times) if build_times else 0,
                "max": max(build_times) if build_times else 0,
            },
            "transpile_times": {
                "count": len(transpile_times),
                "total": sum(transpile_times),
                "avg": sum(transpile_times) / len(transpile_times) if transpile_times else 0,
                "min": min(transpile_times) if transpile_times else 0,
                "max": max(transpile_times) if transpile_times else 0,
            },
            "by_solver": {},
            "by_backend": {}
        }
        
        # Per-solver statistics
        for solver_name, solver_times in timing_by_solver.items():
            build_list = solver_times["build"]
            transpile_list = solver_times["transpile"]
            summary["by_solver"][solver_name] = {
                "build_avg": sum(build_list) / len(build_list) if build_list else 0,
                "transpile_avg": sum(transpile_list) / len(transpile_list) if transpile_list else 0,
                "build_count": len(build_list),
                "transpile_count": len(transpile_list)
            }
        
        # Per-backend statistics
        for backend_alias, backend_times in timing_by_backend.items():
            summary["by_backend"][backend_alias] = {
                "avg": sum(backend_times) / len(backend_times) if backend_times else 0,
                "count": len(backend_times),
                "min": min(backend_times) if backend_times else 0,
                "max": max(backend_times) if backend_times else 0,
            }
        
        return summary
        
    def print_progress_summary(self) -> None:
        """Print a formatted progress summary to console."""
        summary = self.get_progress_summary()
        
        print("Experiment Progress Summary:")
        print(f"Puzzles: {summary['puzzle_progress']} ({summary['puzzle_completion_rate']:.1%})")
        print(f"Configurations: {summary['config_progress']} ({summary['config_completion_rate']:.1%})")
        print()
        
    def print_timing_summary(self) -> None:
        """Print a formatted timing summary to console."""
        timing = self.get_timing_summary()
        
        print("Timing Summary:")
        print("Circuit Building:")
        print(f"  Count: {timing['build_times']['count']}")
        print(f"  Total: {timing['build_times']['total']:.2f}s")
        print(f"  Average: {timing['build_times']['avg']:.2f}s")
        if timing['build_times']['count'] > 0:
            print(f"  Range: {timing['build_times']['min']:.2f}s - {timing['build_times']['max']:.2f}s")
        
        print("\nTranspilation:")
        print(f"  Count: {timing['transpile_times']['count']}")
        print(f"  Total: {timing['transpile_times']['total']:.2f}s")
        print(f"  Average: {timing['transpile_times']['avg']:.2f}s")
        if timing['transpile_times']['count'] > 0:
            print(f"  Range: {timing['transpile_times']['min']:.2f}s - {timing['transpile_times']['max']:.2f}s")
        
        if timing['by_solver']:
            print("\nBy Solver:")
            for solver_name, stats in timing['by_solver'].items():
                print(f"  {solver_name}:")
                print(f"    Build: {stats['build_avg']:.2f}s avg ({stats['build_count']} circuits)")
                print(f"    Transpile: {stats['transpile_avg']:.2f}s avg ({stats['transpile_count']} transpilations)")
        
        if timing['by_backend']:
            print("\nBy Backend:")
            for backend_alias, stats in timing['by_backend'].items():
                print(f"  {backend_alias}:")
                print(f"    Average: {stats['avg']:.2f}s ({stats['count']} transpilations)")
                if stats['count'] > 0:
                    print(f"    Range: {stats['min']:.2f}s - {stats['max']:.2f}s")
        print()


def timed_transpile(qs: QSudoku, backend_alias: str, opt_level: int) -> tuple[Any, float]:
    """
    Standalone function for timed transpilation (for backward compatibility).
    
    Args:
        qs: QSudoku instance with active solver
        backend_alias: Backend alias for transpilation  
        opt_level: Optimization level
        
    Returns:
        Tuple of (transpilation_result, elapsed_seconds)
    """
    start_time = time.time()
    result = qs.transpile(backend_alias, opt_level)
    elapsed = time.time() - start_time
    return result, elapsed
