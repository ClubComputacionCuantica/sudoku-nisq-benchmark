"""Main Benchmark class for staged quantum hardware comparison."""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from dataclasses import asdict

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
from sudoku_nisq.backends import BackendManager
from sudoku_nisq.metadata_manager import MetadataManager
from sudoku_nisq.benchmark.results import (
    LogicalAnalysis,
    BackendFeasibility,
    TranspilationReport,
    ExecutionResults,
    BackendExecution,
    FeasibilityStatus,
)


class Benchmark:
    """Staged benchmarking pipeline for quantum Sudoku solvers.
    
    Provides a three-stage workflow:
    1. Stage 0: Logical circuit analysis (no hardware)
    2. Stage 1: Transpilation and feasibility checking
    3. Stage 2: Hardware execution (opt-in with confirmation)
    
    Example:
        >>> bench = Benchmark(size=4, missing_cells=6)
        >>> bench.add_backend("aer")
        >>> 
        >>> # Stage 0: Analyze logical circuit
        >>> logical = bench.analyze_logical_circuits()
        >>> 
        >>> # Stage 1: Check if it fits
        >>> transpiled = bench.transpile_all(opt_level=2)
        >>> transpiled.print_summary()
        >>> 
        >>> # Stage 2: Run hardware
        >>> if transpiled.get_feasible_backends():
        >>>     results = bench.run_hardware(shots=1024)
    """
    
    def __init__(
        self,
        puzzle: Optional[SudokuPuzzle] = None,
        size: int = 9,
        missing_cells: int = 20,
        solver_class: Optional[Type] = None,
        encoding: str = "simple",
        cache_base: str = ".benchmark_cache",
        **solver_kwargs
    ):
        """Initialize benchmark with puzzle and configuration.
        
        Args:
            puzzle: Existing SudokuPuzzle instance (if None, generates new puzzle)
            size: Grid size for puzzle generation (e.g., 9 for 9×9)
            missing_cells: Number of empty cells in generated puzzle
            solver_class: Quantum solver class (default: ExactCoverQuantumSolver)
            encoding: Encoding strategy ('simple', 'pattern', etc.)
            cache_base: Base directory for caching circuits and results
            **solver_kwargs: Additional arguments passed to solver constructor
        """
        # Create or use existing puzzle
        if puzzle is None:
            self.puzzle = QSudoku.generate(
                subgrid_size=int(size**0.5) if size in [4, 9, 16] else 3,
                num_missing_cells=missing_cells,
                cache_base=cache_base
            )
        else:
            self.puzzle = puzzle if isinstance(puzzle, QSudoku) else QSudoku(puzzle, cache_base=cache_base)
        
        # Set up solver
        if solver_class is None:
            from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
            solver_class = ExactCoverQuantumSolver
        
        self.solver_class = solver_class
        self.encoding = encoding
        self.solver_kwargs = solver_kwargs
        self.cache_base = Path(cache_base)
        
        # Backend configuration
        self.backends_config: List[Dict[str, Any]] = []
        
        # Stage results (cached)
        self._logical_analysis: Optional[LogicalAnalysis] = None
        self._transpilation_report: Optional[TranspilationReport] = None
        self._execution_results: Optional[ExecutionResults] = None
        
        # Backend manager reference
        self._backend_manager = BackendManager.inst()
    
    def add_backend(
        self,
        alias: str,
        label: Optional[str] = None,
        opt_level: int = 1,
        **kwargs
    ) -> None:
        """Register a backend for benchmarking.
        
        Args:
            alias: Backend alias in BackendManager
            label: Human-readable label (defaults to alias)
            opt_level: Transpilation optimization level
            **kwargs: Additional backend-specific parameters
        """
        self.backends_config.append({
            "alias": alias,
            "label": label or alias,
            "opt_level": opt_level,
            **kwargs
        })
    
    # ========================================================================
    # STAGE 0: Logical Circuit Analysis
    # ========================================================================
    
    def analyze_logical_circuits(self, force_rebuild: bool = False) -> LogicalAnalysis:
        """Stage 0: Build and analyze logical circuits without hardware constraints.
        
        Builds the quantum circuit using the configured solver and encoding,
        extracts resource metrics (qubits, gates, depth), and returns analysis.
        
        Args:
            force_rebuild: If True, rebuild circuit even if cached
            
        Returns:
            LogicalAnalysis object with circuit metrics
        """
        # Return cached result if available
        if self._logical_analysis is not None and not force_rebuild:
            return self._logical_analysis
        
        print("="*70)
        print("STAGE 0: Logical Circuit Analysis")
        print("="*70)
        
        # Set up solver
        self.puzzle.set_solver(
            self.solver_class,
            encoding=self.encoding,
            **self.solver_kwargs
        )
        
        # Build circuit
        print(f"Building circuit with {self.solver_class.__name__}...")
        circuit = self.puzzle.build_circuit()
        
        # Get solver reference
        solver = self.puzzle._solver
        
        # Extract metrics
        gate_counts = solver.get_gate_counts() or {}
        
        # Get basic metrics
        n_qubits = getattr(solver, 'u_size', 0) + getattr(solver, 's_size', 0) + getattr(solver, 'b', 0)
        n_gates = sum(gate_counts.values()) if gate_counts else 0
        depth = gate_counts.get('depth', 0) if isinstance(gate_counts, dict) else 0
        
        # Grover iterations (if applicable)
        estimated_iterations = None
        if hasattr(solver, 'num_solutions') and hasattr(solver, 's_size'):
            try:
                import math
                estimated_iterations = math.floor(
                    (math.pi / 4) * math.sqrt((2 ** solver.s_size) / solver.num_solutions)
                )
            except Exception:
                pass
        
        # Determine SDK type
        sdk_type = "pytket"  # Default
        if hasattr(solver, 'main_circuit'):
            if solver.main_circuit is not None:
                circuit_type = type(solver.main_circuit).__name__
                if 'Qiskit' in circuit_type or 'QuantumCircuit' in circuit_type:
                    sdk_type = "qiskit"
                elif 'Circuit' in circuit_type:
                    sdk_type = "pytket"
        
        # Create analysis object
        self._logical_analysis = LogicalAnalysis(
            solver_name=self.solver_class.__name__,
            encoding=self.encoding,
            puzzle_size=self.puzzle.puzzle.subgrid_size ** 2,  # Convert subgrid to grid size
            missing_cells=len(self.puzzle.puzzle.open_tuples),
            n_qubits=n_qubits,
            n_gates=n_gates,
            depth=depth,
            gate_breakdown=gate_counts,
            sdk_type=sdk_type,
            estimated_iterations=estimated_iterations,
        )
        
        # Print summary
        print(f"\n[OK] Circuit built successfully")
        print(f"  Solver: {self._logical_analysis.solver_name}")
        print(f"  Encoding: {self._logical_analysis.encoding}")
        print(f"  Puzzle: {self._logical_analysis.puzzle_size}×{self._logical_analysis.puzzle_size} ({self._logical_analysis.missing_cells} empty cells)")
        print(f"  Qubits: {self._logical_analysis.n_qubits}")
        print(f"  Gates: {self._logical_analysis.n_gates}")
        print(f"  Depth: {self._logical_analysis.depth}")
        print(f"  SDK: {self._logical_analysis.sdk_type}")
        if estimated_iterations:
            print(f"  Grover iterations: {estimated_iterations}")
        print()
        
        return self._logical_analysis
    
    # ========================================================================
    # STAGE 1: Transpilation & Feasibility
    # ========================================================================
    
    def transpile_all(
        self,
        opt_level: Optional[int] = None,
        force_rebuild: bool = False
    ) -> TranspilationReport:
        """Stage 1: Transpile for all backends and check feasibility.
        
        Compiles the circuit for each configured backend, checks if it fits,
        and generates a feasibility report with warnings and recommendations.
        
        Args:
            opt_level: Override optimization level for all backends (default: use per-backend config)
            force_rebuild: If True, rebuild transpiled circuits even if cached
            
        Returns:
            TranspilationReport with per-backend feasibility analysis
        """
        # Ensure Stage 0 is complete
        if self._logical_analysis is None:
            self.analyze_logical_circuits()
        
        print("="*70)
        print("STAGE 1: Transpilation & Feasibility Check")
        print("="*70)
        print()
        
        backend_reports: Dict[str, BackendFeasibility] = {}
        
        for backend_config in self.backends_config:
            alias = backend_config["alias"]
            config_opt_level = opt_level if opt_level is not None else backend_config.get("opt_level", 1)
            
            print(f"Analyzing backend: {alias} (opt_level={config_opt_level})")
            
            try:
                # Get backend from manager
                try:
                    backend = self._backend_manager.get(alias)
                except ValueError as e:
                    # Backend not registered
                    backend_reports[alias] = BackendFeasibility(
                        backend_alias=alias,
                        status=FeasibilityStatus.UNAVAILABLE,
                        device_qubits=None,
                        required_qubits=self._logical_analysis.n_qubits,
                        transpiled_depth=None,
                        transpiled_gates=None,
                        optimization_level=config_opt_level,
                        qubit_utilization=0.0,
                        feasibility_score=0.0,
                        warnings=[f"Backend not registered: {e}"],
                        notes="Backend must be initialized first"
                    )
                    continue
                
                # Try to get device info
                device_qubits = None
                try:
                    # Try pytket backend interface
                    if hasattr(backend, 'backend_info'):
                        info = backend.backend_info
                        if isinstance(info, dict):
                            device_qubits = info.get('n_qubits')
                    if device_qubits is None and hasattr(backend, 'device_state'):
                        device_state = backend.device_state(backend.device_name)
                        if hasattr(device_state, 'n_qubits'):
                            device_qubits = device_state.n_qubits
                except Exception:
                    pass
                
                # Transpile
                try:
                    transpile_result = self.puzzle._solver.transpile_and_analyze(
                        backend=backend,
                        backend_alias=alias,
                        opt_level=config_opt_level,
                        force_overwrite=force_rebuild
                    )
                    
                    required_qubits = transpile_result.get('n_qubits', self._logical_analysis.n_qubits)
                    transpiled_depth = transpile_result.get('depth', 0)
                    transpiled_gates = transpile_result.get('n_gates', 0)
                    
                except Exception as e:
                    # Transpilation failed
                    backend_reports[alias] = BackendFeasibility(
                        backend_alias=alias,
                        status=FeasibilityStatus.ERROR,
                        device_qubits=device_qubits,
                        required_qubits=self._logical_analysis.n_qubits,
                        transpiled_depth=None,
                        transpiled_gates=None,
                        optimization_level=config_opt_level,
                        qubit_utilization=0.0,
                        feasibility_score=0.0,
                        warnings=[f"Transpilation failed: {str(e)}"],
                        notes="Check backend compatibility and circuit structure"
                    )
                    continue
                
                # Compute feasibility metrics
                warnings = []
                
                # Check if circuit fits
                if device_qubits is not None:
                    qubit_utilization = required_qubits / device_qubits
                    
                    if required_qubits > device_qubits:
                        status = FeasibilityStatus.TOO_LARGE
                        feasibility_score = 0.0
                        warnings.append(f"Circuit requires {required_qubits} qubits but device has only {device_qubits}")
                    elif qubit_utilization > 0.8:
                        status = FeasibilityStatus.TIGHT
                        feasibility_score = 0.5
                        warnings.append(f"High qubit utilization ({qubit_utilization:.1%}) may increase error rates")
                    else:
                        status = FeasibilityStatus.FEASIBLE
                        # Score based on utilization (lower is better up to a point)
                        feasibility_score = 0.9 if qubit_utilization < 0.5 else 0.7
                else:
                    # Unknown device size
                    status = FeasibilityStatus.METRICS_AFTER_EXECUTION
                    qubit_utilization = 0.0
                    feasibility_score = 0.5
                    warnings.append("Device qubit count unknown; metrics available after execution")
                
                # Check depth (if we have coherence time info)
                if transpiled_depth and transpiled_depth > 1000:
                    warnings.append(f"High circuit depth ({transpiled_depth}) may exceed coherence time")
                    feasibility_score *= 0.9
                
                # Estimate runtime
                estimated_runtime = None
                if transpiled_depth:
                    # Rough estimate: ~10μs per gate on superconducting qubits
                    runtime_seconds = transpiled_depth * 10e-6 * 1.5  # Add overhead
                    if runtime_seconds < 1:
                        estimated_runtime = f"~{int(runtime_seconds*1000)}ms"
                    else:
                        estimated_runtime = f"~{int(runtime_seconds)}s"
                
                backend_reports[alias] = BackendFeasibility(
                    backend_alias=alias,
                    status=status,
                    device_qubits=device_qubits,
                    required_qubits=required_qubits,
                    transpiled_depth=transpiled_depth,
                    transpiled_gates=transpiled_gates,
                    optimization_level=config_opt_level,
                    qubit_utilization=qubit_utilization,
                    feasibility_score=feasibility_score,
                    warnings=warnings,
                    estimated_runtime=estimated_runtime,
                )
                
                print(f"  Status: {status.value}")
                print(f"  Required qubits: {required_qubits}" + (f"/{device_qubits}" if device_qubits else ""))
                print(f"  Transpiled depth: {transpiled_depth}")
                print(f"  Feasibility score: {feasibility_score:.2f}")
                if warnings:
                    for warning in warnings:
                        print(f"  ⚠  {warning}")
                print()
                
            except Exception as e:
                # Unexpected error
                backend_reports[alias] = BackendFeasibility(
                    backend_alias=alias,
                    status=FeasibilityStatus.ERROR,
                    device_qubits=None,
                    required_qubits=self._logical_analysis.n_qubits,
                    transpiled_depth=None,
                    transpiled_gates=None,
                    optimization_level=config_opt_level,
                    qubit_utilization=0.0,
                    feasibility_score=0.0,
                    warnings=[f"Unexpected error: {str(e)}"],
                )
                print(f"  [ERROR] Error: {e}\\n")
        
        self._transpilation_report = TranspilationReport(backend_reports)
        return self._transpilation_report
    
    # ========================================================================
    # STAGE 2: Hardware Execution
    # ========================================================================
    
    def run_hardware(
        self,
        backends: Optional[List[str]] = None,
        shots: int = 1024,
        confirm: bool = True,
        use_mitigation: bool = False,
        **execution_kwargs
    ) -> ExecutionResults:
        """Stage 2: Execute on real quantum hardware (opt-in).
        
        Runs the circuit on selected backends with optional confirmation prompt
        to prevent accidental hardware usage and costs.
        
        Args:
            backends: List of backend aliases to run (None = all feasible backends)
            shots: Number of measurement shots per execution
            confirm: If True, prompt for confirmation before execution
            use_mitigation: Apply error mitigation (ZNE)
            **execution_kwargs: Additional arguments for execution
            
        Returns:
            ExecutionResults with per-backend execution data
        """
        # Ensure Stage 1 is complete
        if self._transpilation_report is None:
            self.transpile_all()
        
        # Determine which backends to run
        if backends is None:
            backends = self._transpilation_report.get_feasible_backends()
            if not backends:
                print("⚠️  No feasible backends found. Aborting hardware execution.")
                return ExecutionResults({})
        
        # Confirmation prompt
        if confirm:
            print("\n" + "="*70)
            print("⚠️  HARDWARE EXECUTION CONFIRMATION")
            print("="*70)
            print("\nYou are about to execute on REAL QUANTUM HARDWARE:")
            for alias in backends:
                report = self._transpilation_report[alias]
                print(f"\n  • {alias}:")
                print(f"    - Shots: {shots}")
                print(f"    - Est. time: {report.estimated_runtime or 'unknown'}")
                if report.warnings:
                    print(f"    - Warnings: {len(report.warnings)}")
            
            print("\n" + "-"*70)
            response = input("Continue? [y/N]: ").strip().lower()
            if response not in ('y', 'yes'):
                print("Execution cancelled.")
                return ExecutionResults({})
        
        print("\n" + "="*70)
        print("STAGE 2: Hardware Execution")
        print("="*70)
        print()
        
        backend_results: Dict[str, BackendExecution] = {}
        
        for alias in backends:
            print(f"Executing on {alias}...")
            start_time = time.time()
            
            try:
                # Get backend
                backend = self._backend_manager.get(alias)
                
                # Get optimization level from config
                opt_level = next(
                    (cfg['opt_level'] for cfg in self.backends_config if cfg['alias'] == alias),
                    1
                )
                
                # Run on hardware
                result = self.puzzle._solver.run(
                    backend=backend,
                    backend_alias=alias,
                    shots=shots,
                    optimisation_level=opt_level,
                    use_zne=use_mitigation,
                    **execution_kwargs
                )
                
                execution_time = time.time() - start_time
                
                # Extract counts
                counts = result.get_counts() if hasattr(result, 'get_counts') else result.get('counts', {})
                
                # Compute success rate
                success_rate = None
                mitigated_rate = None
                top_solutions = []
                
                if counts and hasattr(self.puzzle._solver, '_is_valid_solution'):
                    total = sum(counts.values())
                    valid_count = sum(
                        count for bitstring, count in counts.items()
                        if self.puzzle._solver._is_valid_solution(bitstring)
                    )
                    success_rate = valid_count / total if total > 0 else 0.0
                    
                    # Top solutions
                    top_solutions = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    # Check for mitigated result
                    if hasattr(result, '_mitigated_success_prob'):
                        mitigated_rate = result._mitigated_success_prob
                
                backend_results[alias] = BackendExecution(
                    backend_alias=alias,
                    status="COMPLETED",
                    job_id=getattr(result, 'job_id', None),
                    execution_time=execution_time,
                    shots=shots,
                    counts=counts,
                    success_rate=success_rate,
                    top_solutions=top_solutions,
                    mitigated_success_rate=mitigated_rate,
                )
                
                print(f"  [OK] Completed in {execution_time:.1f}s")
                if success_rate is not None:
                    print(f"    Success rate: {success_rate:.1%}")
                if mitigated_rate:
                    print(f"    Mitigated: {mitigated_rate:.1%}")
                print()
                
            except Exception as e:
                execution_time = time.time() - start_time
                backend_results[alias] = BackendExecution(
                    backend_alias=alias,
                    status="FAILED",
                    execution_time=execution_time,
                    shots=shots,
                    error_message=str(e),
                )
                print(f"  [FAIL] Failed: {e}\n")
        
        self._execution_results = ExecutionResults(backend_results)
        return self._execution_results
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def save_results(self, filepath: str) -> None:
        """Save all benchmark results to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        results = {
            'logical_analysis': self._logical_analysis.to_dict() if self._logical_analysis else None,
            'transpilation_report': self._transpilation_report.to_dict() if self._transpilation_report else None,
            'execution_results': self._execution_results.to_dict() if self._execution_results else None,
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[OK] Results saved to {filepath}")
    
    @classmethod
    def quick_compare(
        cls,
        backends: List[str],
        size: int = 4,
        missing_cells: int = 6,
        shots: int = 1024,
        **kwargs
    ) -> ExecutionResults:
        """Quick one-liner to compare backends.
        
        Args:
            backends: List of backend aliases to compare
            size: Puzzle grid size
            missing_cells: Number of empty cells
            shots: Shots per backend
            **kwargs: Additional arguments for Benchmark constructor
            
        Returns:
            ExecutionResults from hardware execution
        """
        bench = cls(size=size, missing_cells=missing_cells, **kwargs)
        for backend in backends:
            bench.add_backend(backend)
        
        # Run all stages
        bench.analyze_logical_circuits()
        bench.transpile_all()
        return bench.run_hardware(shots=shots, confirm=False)
