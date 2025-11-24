"""Result data classes for staged benchmarking pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime


class FeasibilityStatus(Enum):
    """Status of hardware feasibility check."""
    FEASIBLE = "FEASIBLE"
    TOO_LARGE = "TOO_LARGE"
    TIGHT = "TIGHT"
    METRICS_AFTER_EXECUTION = "METRICS_AFTER_EXECUTION"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


@dataclass
class LogicalAnalysis:
    """Results from Stage 0: Logical circuit analysis.
    
    Contains metrics for the logical (pre-transpilation) quantum circuit,
    independent of any specific hardware backend.
    
    Attributes:
        solver_name: Name of the quantum solver (e.g., 'ExactCoverQuantumSolver')
        encoding: Encoding strategy used ('simple', 'pattern', etc.)
        puzzle_size: Grid size of the Sudoku puzzle
        missing_cells: Number of empty cells in the puzzle
        n_qubits: Number of qubits required by the logical circuit
        n_gates: Total gate count in the logical circuit
        depth: Circuit depth (critical path length)
        gate_breakdown: Dictionary mapping gate types to counts
        sdk_type: SDK used to build the circuit ('qiskit', 'pytket', 'braket')
        estimated_iterations: Number of Grover iterations (if applicable)
        circuit_built_at: Timestamp when circuit was built
    """
    solver_name: str
    encoding: str
    puzzle_size: int
    missing_cells: int
    n_qubits: int
    n_gates: int
    depth: int
    gate_breakdown: Dict[str, int]
    sdk_type: str
    estimated_iterations: Optional[int] = None
    circuit_built_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def fits_on(self, backend_qubits: int) -> bool:
        """Check if logical circuit fits on backend with given qubit count.
        
        Args:
            backend_qubits: Number of qubits available on the backend
            
        Returns:
            True if the circuit fits, False otherwise
        """
        return self.n_qubits <= backend_qubits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'solver_name': self.solver_name,
            'encoding': self.encoding,
            'puzzle_size': self.puzzle_size,
            'missing_cells': self.missing_cells,
            'n_qubits': self.n_qubits,
            'n_gates': self.n_gates,
            'depth': self.depth,
            'gate_breakdown': self.gate_breakdown,
            'sdk_type': self.sdk_type,
            'estimated_iterations': self.estimated_iterations,
            'circuit_built_at': self.circuit_built_at,
        }


@dataclass
class BackendFeasibility:
    """Feasibility analysis for a single backend (Stage 1).
    
    Attributes:
        backend_alias: Alias of the backend being analyzed
        status: Overall feasibility status
        device_qubits: Total qubits available on the device (None if unknown)
        required_qubits: Qubits needed after transpilation and routing
        transpiled_depth: Circuit depth after transpilation (None if not yet transpiled)
        transpiled_gates: Gate count after transpilation (None if not yet transpiled)
        optimization_level: Transpilation optimization level used
        qubit_utilization: Fraction of device qubits used (0.0-1.0)
        feasibility_score: Overall feasibility score (0.0-1.0, higher is better)
        warnings: List of warning messages about potential issues
        estimated_runtime: Estimated execution time as human-readable string
        notes: Additional notes about the backend or transpilation
    """
    backend_alias: str
    status: FeasibilityStatus
    device_qubits: Optional[int]
    required_qubits: int
    transpiled_depth: Optional[int]
    transpiled_gates: Optional[int]
    optimization_level: int
    qubit_utilization: float
    feasibility_score: float
    warnings: List[str] = field(default_factory=list)
    estimated_runtime: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'backend_alias': self.backend_alias,
            'status': self.status.value,
            'device_qubits': self.device_qubits,
            'required_qubits': self.required_qubits,
            'transpiled_depth': self.transpiled_depth,
            'transpiled_gates': self.transpiled_gates,
            'optimization_level': self.optimization_level,
            'qubit_utilization': self.qubit_utilization,
            'feasibility_score': self.feasibility_score,
            'warnings': self.warnings,
            'estimated_runtime': self.estimated_runtime,
            'notes': self.notes,
        }


class TranspilationReport:
    """Container for Stage 1 transpilation results across all backends.
    
    Provides methods to analyze and query feasibility across multiple backends.
    """
    
    def __init__(self, backend_reports: Dict[str, BackendFeasibility]):
        """Initialize with per-backend feasibility reports.
        
        Args:
            backend_reports: Dictionary mapping backend aliases to their feasibility reports
        """
        self.backends = backend_reports
        self.timestamp = datetime.now().isoformat()
    
    def get_feasible_backends(self, min_score: float = 0.5) -> List[str]:
        """Return backends that are feasible to run on.
        
        Args:
            min_score: Minimum feasibility score required (default: 0.5)
            
        Returns:
            List of backend aliases that meet feasibility criteria
        """
        return [
            alias for alias, report in self.backends.items()
            if report.status in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.TIGHT)
            and report.feasibility_score >= min_score
        ]
    
    def get_infeasible_backends(self) -> List[str]:
        """Return backends that cannot run the circuit.
        
        Returns:
            List of backend aliases that are too small or unavailable
        """
        return [
            alias for alias, report in self.backends.items()
            if report.status in (FeasibilityStatus.TOO_LARGE, FeasibilityStatus.UNAVAILABLE, FeasibilityStatus.ERROR)
        ]
    
    def print_summary(self) -> None:
        """Print a formatted summary table of transpilation results."""
        if not self.backends:
            print("No backends analyzed.")
            return
        
        print("\n" + "="*80)
        print("TRANSPILATION FEASIBILITY REPORT")
        print("="*80)
        
        # Header
        print(f"\n{'Backend':<20} {'Status':<12} {'Qubits':<15} {'Depth':<10} {'Score':<10}")
        print("-"*80)
        
        # Rows
        for alias, report in self.backends.items():
            status_symbol = {
                FeasibilityStatus.FEASIBLE: "✓ READY",
                FeasibilityStatus.TIGHT: "⚠ TIGHT",
                FeasibilityStatus.TOO_LARGE: "✗ TOO_LARGE",
                FeasibilityStatus.METRICS_AFTER_EXECUTION: "? PENDING",
                FeasibilityStatus.UNAVAILABLE: "✗ UNAVAIL",
                FeasibilityStatus.ERROR: "✗ ERROR",
            }.get(report.status, "? UNKNOWN")
            
            qubits_str = f"{report.required_qubits}/{report.device_qubits}" if report.device_qubits else f"{report.required_qubits}/?"
            depth_str = str(report.transpiled_depth) if report.transpiled_depth else "?"
            score_str = f"{report.feasibility_score:.2f}"
            
            print(f"{alias:<20} {status_symbol:<12} {qubits_str:<15} {depth_str:<10} {score_str:<10}")
            
            # Print warnings if any
            if report.warnings:
                for warning in report.warnings:
                    print(f"  ⚠  {warning}")
        
        print("\n" + "="*80)
        
        # Summary
        feasible = self.get_feasible_backends()
        infeasible = self.get_infeasible_backends()
        
        print(f"Feasible backends: {len(feasible)}")
        if feasible:
            print(f"  → {', '.join(feasible)}")
        
        if infeasible:
            print(f"Infeasible backends: {len(infeasible)}")
            print(f"  → {', '.join(infeasible)}")
        
        print()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'backends': {
                alias: report.to_dict()
                for alias, report in self.backends.items()
            }
        }
    
    def __getitem__(self, backend_alias: str) -> BackendFeasibility:
        """Access backend report by alias."""
        return self.backends[backend_alias]
    
    def __contains__(self, backend_alias: str) -> bool:
        """Check if backend is in the report."""
        return backend_alias in self.backends
    
    def __iter__(self):
        """Iterate over backend aliases."""
        return iter(self.backends)
    
    def items(self):
        """Iterate over (alias, report) pairs."""
        return self.backends.items()


@dataclass
class BackendExecution:
    """Results from hardware execution on a single backend (Stage 2).
    
    Attributes:
        backend_alias: Alias of the backend
        status: Execution status ('COMPLETED', 'FAILED', 'PENDING')
        job_id: Job identifier from the backend
        execution_time: Actual wall-clock execution time in seconds
        shots: Number of shots executed
        counts: Raw measurement counts
        success_rate: Fraction of valid solutions (0.0-1.0)
        top_solutions: List of (bitstring, probability) tuples for top solutions
        mitigated_success_rate: Success rate after error mitigation (if applied)
        timestamp: ISO timestamp of execution
        error_message: Error message if execution failed
    """
    backend_alias: str
    status: str
    job_id: Optional[str] = None
    execution_time: Optional[float] = None
    shots: Optional[int] = None
    counts: Optional[Dict[str, int]] = None
    success_rate: Optional[float] = None
    top_solutions: List[tuple] = field(default_factory=list)
    mitigated_success_rate: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'backend_alias': self.backend_alias,
            'status': self.status,
            'job_id': self.job_id,
            'execution_time': self.execution_time,
            'shots': self.shots,
            'counts': self.counts,
            'success_rate': self.success_rate,
            'top_solutions': self.top_solutions,
            'mitigated_success_rate': self.mitigated_success_rate,
            'timestamp': self.timestamp,
            'error_message': self.error_message,
        }


class ExecutionResults:
    """Container for Stage 2 hardware execution results.
    
    Aggregates execution results from multiple backends and provides
    analysis and comparison capabilities.
    """
    
    def __init__(self, backend_results: Dict[str, BackendExecution]):
        """Initialize with per-backend execution results.
        
        Args:
            backend_results: Dictionary mapping backend aliases to execution results
        """
        self.backends = backend_results
        self.timestamp = datetime.now().isoformat()
    
    def get_successful_runs(self) -> List[str]:
        """Return backends that completed successfully.
        
        Returns:
            List of backend aliases with completed executions
        """
        return [
            alias for alias, result in self.backends.items()
            if result.status == "COMPLETED"
        ]
    
    def get_failed_runs(self) -> List[str]:
        """Return backends that failed execution.
        
        Returns:
            List of backend aliases with failed executions
        """
        return [
            alias for alias, result in self.backends.items()
            if result.status == "FAILED"
        ]
    
    def print_summary(self) -> None:
        """Print a formatted summary of execution results."""
        if not self.backends:
            print("No execution results available.")
            return
        
        print("\n" + "="*80)
        print("HARDWARE EXECUTION RESULTS")
        print("="*80)
        
        # Header
        print(f"\n{'Backend':<20} {'Status':<12} {'Success Rate':<15} {'Time (s)':<12} {'Shots':<10}")
        print("-"*80)
        
        # Rows
        for alias, result in self.backends.items():
            status_str = "✓ OK" if result.status == "COMPLETED" else "✗ FAILED"
            success_str = f"{result.success_rate:.1%}" if result.success_rate is not None else "N/A"
            time_str = f"{result.execution_time:.1f}" if result.execution_time is not None else "N/A"
            shots_str = str(result.shots) if result.shots else "N/A"
            
            print(f"{alias:<20} {status_str:<12} {success_str:<15} {time_str:<12} {shots_str:<10}")
            
            if result.mitigated_success_rate and result.success_rate:
                improvement = result.mitigated_success_rate - result.success_rate
                print(f"  → Mitigated: {result.mitigated_success_rate:.1%} ({improvement:+.1%})")
            
            if result.error_message:
                print(f"  ✗ Error: {result.error_message}")
        
        print("\n" + "="*80)
        print()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'backends': {
                alias: result.to_dict()
                for alias, result in self.backends.items()
            }
        }
    
    def __getitem__(self, backend_alias: str) -> BackendExecution:
        """Access backend result by alias."""
        return self.backends[backend_alias]
    
    def __contains__(self, backend_alias: str) -> bool:
        """Check if backend is in the results."""
        return backend_alias in self.backends
