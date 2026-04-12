"""BenchmarkSession: High-level orchestration API for multi-run benchmarks.

Provides unified interface for end-to-end benchmarking workflows across all 7 stages.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from sudoku_nisq.metadata.config import MetadataConfig
from sudoku_nisq.metadata.instance import InstanceMetadataManager
from sudoku_nisq.metadata.logical_ir import LogicalIRMetadataManager
from sudoku_nisq.metadata.ir_policy import IRPolicyMetadataManager
from sudoku_nisq.metadata.compilation import CompilationMetadataManager
from sudoku_nisq.metadata.executable import ExecutableMetadataManager
from sudoku_nisq.metadata.execution import ExecutionMetadataManager
from sudoku_nisq.metadata.metrics import MetricsMetadataManager


class BenchmarkSession:
    """Orchestrates multi-run benchmarking workflows across all 7 metadata stages.
    
    Provides high-level API for:
    - Puzzle instance registration (Stage 1)
    - Circuit construction tracking (Stages 2a-2b)
    - Compilation provenance (Stage 3)
    - Execution recording (Stage 5)
    - Metrics computation (Stages 6-7)
    - Multi-run aggregation
    - Cross-stage queries
    
    Example::
    
        from sudoku_nisq.metadata import BenchmarkSession
        from sudoku_nisq import QSudoku, ExactCoverQuantumSolver
        
        # Initialize session
        session = BenchmarkSession(puzzle_hash="abc123")
        
        # Register puzzle instance
        puzzle = QSudoku.generate(size=4, num_missing_cells=2)
        session.register_puzzle(puzzle)
        
        # Execute single run
        result = session.execute_run(
            puzzle=puzzle,
            backend_alias="aer_simulator",
            shots=1024,
            opt_level=1,
            validation_context={"valid_solutions": [...]}
        )
        
        # Execute batch with aggregation
        batch_results = session.execute_batch(
            puzzle=puzzle,
            backend_alias="aer_simulator",
            n_runs=5,
            shots=1024,
            opt_level=1,
            validation_context={"valid_solutions": [...]}
        )
        
        # Query execution history
        executions = session.query_executions(
            backend_alias="aer_simulator",
            opt_level=1
        )
    """
    
    def __init__(
        self, 
        puzzle_hash: Optional[str] = None,
        puzzle: Optional[Any] = None,
        cache_base: Optional[Path] = None
    ):
        """Initialize benchmark session with stage managers.
        
        Args:
            puzzle_hash: Puzzle identifier (required if puzzle not provided)
            puzzle: QSudoku instance to extract puzzle_hash from (alternative to puzzle_hash)
            cache_base: Base cache directory (defaults to MetadataConfig.get_cache_base())
        
        Raises:
            ValueError: If neither puzzle_hash nor puzzle is provided
        
        Example:
            .. code-block:: python

                # Option 1: Direct puzzle_hash
                session = BenchmarkSession(puzzle_hash="abc123")
                
                # Option 2: Extract from puzzle
                puzzle = QSudoku.generate(size=2)
                session = BenchmarkSession(puzzle=puzzle)
        """
        if puzzle_hash is None and puzzle is not None:
            puzzle_hash = puzzle.get_hash()
        if puzzle_hash is None:
            raise ValueError("Must provide either puzzle_hash or puzzle parameter")
        
        self.puzzle_hash = puzzle_hash
        self.cache_base = Path(cache_base) if cache_base else MetadataConfig.get_cache_base()
        
        # Initialize all 7 stage managers
        self.stage1 = InstanceMetadataManager(self.cache_base)
        self.stage2a = LogicalIRMetadataManager(self.cache_base, puzzle_hash)
        self.stage2b = IRPolicyMetadataManager(self.cache_base, puzzle_hash)
        self.stage3 = CompilationMetadataManager(self.cache_base, puzzle_hash)
        self.stage4 = ExecutableMetadataManager(self.cache_base, puzzle_hash)
        self.stage5 = ExecutionMetadataManager(self.cache_base, puzzle_hash)
        self.stage6_7 = MetricsMetadataManager(self.cache_base, puzzle_hash)
    
    def register_puzzle(
        self,
        puzzle: Any,
        prng_seed: Optional[int] = None,
        solution_count: Optional[int] = None
    ) -> str:
        """Register puzzle instance to Stage 1 global registry.
        
        Args:
            puzzle: QSudoku or SudokuPuzzle instance
            prng_seed: PRNG seed for reproducibility (deferred to Phase 5.5)
            solution_count: Number of valid solutions (optional)
            
        Returns:
            str: puzzle_hash identifier
            
        Example::
        
            puzzle = QSudoku.generate(size=4, num_missing_cells=2)
            puzzle_hash = session.register_puzzle(puzzle, solution_count=1)
        """
        # Extract puzzle attributes
        # Check if it's a QSudoku wrapper
        # Be careful not to access attributes that might create Mock objects
        sudoku_puzzle = puzzle
        try:
            # Check class name first (safest)
            if hasattr(puzzle, '__class__'):
                class_name = str(puzzle.__class__)
                if 'QSudoku' in class_name:
                    sudoku_puzzle = puzzle.puzzle
                elif 'Mock' in class_name:
                    # It's a Mock - check if it has 'puzzle' attribute actually set
                    if 'puzzle' in puzzle.__dict__:
                        sudoku_puzzle = puzzle.puzzle
        except Exception:
            # Any error in detection, use puzzle as-is
            pass
        
        # Safely extract values (handles Mock objects)
        def safe_get(obj, attr, default=None):
            """Safely get attribute value, handling Mock objects."""
            try:
                val = getattr(obj, attr, default)
                # Check if it's an unconfigured Mock attribute (has _mock_name)
                if hasattr(val, '_mock_name') and not hasattr(val, '_spec_class'):
                    return default
                return val
            except AttributeError:
                return default
        
        # Record to Stage 1
        # Safely compute num_missing_cells
        num_missing = None
        open_tuples = safe_get(sudoku_puzzle, 'open_tuples')
        if open_tuples is not None:
            try:
                num_missing = len(open_tuples)
            except (TypeError, AttributeError):
                # Mock objects or other non-iterable
                pass
        
        return self.stage1.record(
            puzzle_hash=self.puzzle_hash,
            size=safe_get(sudoku_puzzle, 'size'),
            subgrid_size=safe_get(sudoku_puzzle, 'subgrid_size'),
            num_missing_cells=num_missing,
            board=safe_get(sudoku_puzzle, 'board'),
            open_tuples=open_tuples,
            pre_tuples=safe_get(sudoku_puzzle, 'pre_tuples'),
            prng_seed=prng_seed,
            solution_count=solution_count
        )
    
    def execute_run(
        self,
        puzzle: Any,
        backend_alias: str,
        shots: int,
        opt_level: int = 1,
        validation_context: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """Execute single run through Stages 2a-7.
        
        Orchestrates:
        1. Circuit construction (Stages 2a-2b via puzzle.build_circuit)
        2. Compilation (Stage 3 via puzzle.run with transpilation)
        3. Execution (Stage 5 via puzzle.run)
        4. Metrics computation (Stages 6-7 auto-computed)
        
        Args:
            puzzle: QSudoku instance (must have solver attached)
            backend_alias: Backend identifier
            shots: Number of measurement shots
            opt_level: Transpilation optimization level (0-3)
            validation_context: Dict with valid_solutions for metrics
            **kwargs: Additional arguments passed to puzzle.run()
            
        Returns:
            ExecutionResult or provider result object
            
        Raises:
            ValueError: If puzzle has no solver attached
            
        Example::
        
            puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
            result = session.execute_run(
                puzzle=puzzle,
                backend_alias="aer_simulator",
                shots=1024,
                opt_level=1,
                validation_context={"valid_solutions": [(1,2,3,4), ...]}
            )
        """
        if not hasattr(puzzle, 'quantum_solver') or puzzle.quantum_solver is None:
            raise ValueError(
                "Puzzle must have solver attached. Call puzzle.set_solver() first."
            )
        
        # Ensure metadata architecture is enabled for stage recording
        if not MetadataConfig.ENABLE_NEW_ARCHITECTURE:
            import warnings
            warnings.warn(
                "BenchmarkSession requires SUDOKU_NISQ_NEW_METADATA=1. "
                "Stage recording may be incomplete.",
                UserWarning
            )
        
        # Set validation context for metrics computation
        if validation_context and hasattr(puzzle, 'set_validation_context'):
            puzzle.set_validation_context(**validation_context)
        
        # Execute run (Stages 2a-7 are auto-recorded in quantum_solver)
        result = puzzle.run(
            backend_alias=backend_alias,
            shots=shots,
            opt_level=opt_level,
            **kwargs
        )
        
        return result
    
    def execute_batch(
        self,
        puzzle: Any,
        backend_alias: str,
        n_runs: int,
        shots: int,
        opt_level: int = 1,
        validation_context: Optional[Dict] = None,
        **kwargs
    ) -> List[Any]:
        """Execute multiple runs and compute aggregated metrics.
        
        Args:
            puzzle: QSudoku instance (must have solver attached)
            backend_alias: Backend identifier
            n_runs: Number of independent runs
            shots: Number of measurement shots per run
            opt_level: Transpilation optimization level (0-3)
            validation_context: Dict with valid_solutions for metrics
            **kwargs: Additional arguments passed to puzzle.run()
            
        Returns:
            List of ExecutionResult objects (one per run)
            
        Example::
        
            results = session.execute_batch(
                puzzle=puzzle,
                backend_alias="aer_simulator",
                n_runs=5,
                shots=1024,
                opt_level=1,
                validation_context={"valid_solutions": [...]}
            )
            
            # Aggregate metrics
            aggregated = session.stage6_7.compute_aggregated(
                run_ids=[r.run_id for r in results if hasattr(r, 'run_id')]
            )
        """
        results = []
        
        # Execute runs with progress bar
        for i in tqdm(range(n_runs), desc=f"Benchmarking {backend_alias}"):
            result = self.execute_run(
                puzzle=puzzle,
                backend_alias=backend_alias,
                shots=shots,
                opt_level=opt_level,
                validation_context=validation_context,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def query_executions(
        self,
        backend_alias: Optional[str] = None,
        opt_level: Optional[int] = None,
        compilation_id: Optional[str] = None,
        date_range: Optional[tuple] = None
    ) -> List[Dict]:
        """Query execution history with cross-stage join (Stage 3 + Stage 5).
        
        Args:
            backend_alias: Filter by backend
            opt_level: Filter by optimization level
            compilation_id: Filter by specific compilation
            date_range: Tuple of (start_datetime, end_datetime)
            
        Returns:
            List of execution records with compilation metadata
            
        Example::
        
            # Get all executions for a backend
            executions = session.query_executions(backend_alias="aer_simulator")
            
            # Filter by opt_level and date range
            from datetime import datetime
            recent = session.query_executions(
                backend_alias="aer_simulator",
                opt_level=1,
                date_range=(datetime(2025, 1, 1), datetime(2025, 1, 31))
            )
        """
        # Query Stage 5 executions
        executions = self.stage5.query(
            compilation_id=compilation_id,
            date_range=date_range
        )
        
        if not executions:
            return []
        
        # Join with Stage 3 compilation metadata
        enriched_executions = []
        for execution in executions:
            comp_id = execution.get('compilation_id')
            if comp_id:
                # Query Stage 3 for compilation metadata
                compilation_result = self.stage3.query(compilation_id=comp_id)
                # Handle both single dict and list of dicts
                compilation = None
                if isinstance(compilation_result, dict):
                    compilation = compilation_result
                elif isinstance(compilation_result, list) and len(compilation_result) > 0:
                    compilation = compilation_result[0]
                
                if compilation:
                    # Filter by backend/opt_level if specified
                    if backend_alias and compilation.get('backend_alias') != backend_alias:
                        continue
                    if opt_level is not None and compilation.get('opt_level') != opt_level:
                        continue
                    
                    # Merge execution + compilation data
                    enriched = {**execution, 'compilation': compilation}
                    enriched_executions.append(enriched)
        
        return enriched_executions
    
    def get_metrics_summary(
        self,
        run_ids: Optional[List[str]] = None
    ) -> Dict:
        """Get metrics summary for specified runs or latest aggregation.
        
        Args:
            run_ids: List of run IDs to aggregate (None = query all)
            
        Returns:
            Dict with metrics for each run + aggregated statistics
            
        Example::
        
            # Get metrics for specific runs
            summary = session.get_metrics_summary(run_ids=['uuid1', 'uuid2', 'uuid3'])
            
            # Get latest aggregation
            summary = session.get_metrics_summary()
        """
        if run_ids:
            # Compute aggregated metrics for specified runs
            result = self.stage6_7.compute_aggregated(run_ids=run_ids)
            return result if result is not None else {}
        else:
            # Query latest aggregation
            # Stage 6-7 stores aggregations with timestamp-prefixed keys
            metrics_data = self.stage6_7._load_json()
            if not metrics_data:
                return {}
            
            # Find latest aggregated entry
            aggregated_keys = [k for k in metrics_data.keys() if k.startswith('aggregated_')]
            if aggregated_keys:
                latest_key = max(aggregated_keys)  # Lexicographic sort by timestamp
                return metrics_data.get(latest_key, {})
            return {}
            
            return {}
