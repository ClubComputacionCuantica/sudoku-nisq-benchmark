"""
Lightweight quantum interface for general exact cover problems.

QExactCover provides a minimal API for running quantum algorithms on arbitrary
exact cover instances, independent of Sudoku structure.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

from sudoku_nisq.exact_cover_problem import ExactCoverProblem
from sudoku_nisq.metadata_manager import MetadataManager
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

if TYPE_CHECKING:
    try:
        from qiskit_aer import AerSimulator
    except ImportError:  # pragma: no cover
        pass


class QExactCover:
    """
    Lightweight quantum interface for general exact cover problems.
    
    QExactCover provides a minimal wrapper for running quantum exact cover algorithms
    on arbitrary problem instances. Unlike QSudoku (which provides extensive puzzle
    management, multi-backend support, and visualization), QExactCover focuses on
    demonstrating quantum algorithm capabilities on smaller, generic exact cover instances.
    
    This is useful for:
    - Benchmarking quantum algorithms on problems smaller than minimal Sudoku (2×2)
    - Testing exact cover formulations without Sudoku constraints
    - Exploring the space of all finite exact cover instances
    - Research on general exact cover quantum algorithms
    
    Attributes:
        problem: The ExactCoverProblem instance to solve
        solver: The quantum solver (ExactCoverQuantumSolver)
        _metadata: MetadataManager for caching circuits
    
    Example:
        >>> # Small example problem
        >>> problem = ExactCoverProblem.create_small_example()
        >>> qec = QExactCover(problem)
        >>> circuit = qec.build_circuit()
        >>> result = qec.run_aer(shots=512)
        >>> print(result['counts'])
    """
    
    def __init__(self, problem: ExactCoverProblem, cache_base: Optional[str] = None):
        """
        Initialize QExactCover with an exact cover problem.
        
        Args:
            problem: ExactCoverProblem instance defining universe and subsets
            cache_base: Base directory for caching circuits (optional)
        
        Example:
            >>> universe = [0, 1, 2, 3]
            >>> subsets = {'S_0': [0, 1], 'S_1': [2, 3]}
            >>> problem = ExactCoverProblem(universe, subsets)
            >>> qec = QExactCover(problem)
        """
        self.problem = problem
        
        # Initialize metadata manager
        self._metadata = MetadataManager(
            cache_base=Path(cache_base) if cache_base else Path(".quantum_solver_cache"),
            puzzle_hash=self.problem.get_hash()
        )
        
        # Initialize solver with the exact cover problem
        self.solver = ExactCoverQuantumSolver(
            exact_cover_problem=problem,
            metadata_manager=self._metadata,
            encoding="simple"  # Only simple encoding for generic problems
        )
    
    @property
    def universe(self):
        """Get the universe of the exact cover problem."""
        return self.problem.universe
    
    @property
    def subsets(self):
        """Get the subsets of the exact cover problem."""
        return self.problem.subsets
    
    @property
    def num_solutions(self):
        """Get the expected number of solutions."""
        return self.problem.num_solutions
    
    def build_circuit(self, sdk: str = "qiskit") -> Any:
        """
        Build the quantum circuit for solving the exact cover problem.
        
        Args:
            sdk: SDK to use for circuit construction ("qiskit" or "pytket")
        
        Returns:
            Quantum circuit in the specified SDK format
        
        Example:
            >>> qec = QExactCover(problem)
            >>> circuit = qec.build_circuit(sdk="qiskit")
            >>> print(f"Circuit uses {circuit.num_qubits} qubits")
        """
        return self.solver.build_main_circuit(sdk=sdk)
    
    def run_aer(self, shots: int = 1024, memory: bool = False, 
                opt_level: int = 0) -> Dict[str, Any]:
        """
        Run the circuit on Qiskit Aer simulator.
        
        Args:
            shots: Number of measurement shots
            memory: Whether to return individual shot results
            opt_level: Transpiler optimization level (0-3)
        
        Returns:
            Dictionary containing:
                - 'counts': Measurement counts dictionary
                - 'memory': Individual shot results (if memory=True)
                - 'metadata': Execution metadata
        
        Example:
            >>> qec = QExactCover(problem)
            >>> qec.build_circuit()
            >>> result = qec.run_aer(shots=512)
            >>> print(f"Top result: {max(result['counts'], key=result['counts'].get)}")
        """
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        # Build circuit if not already built
        if self.solver.main_circuit is None:
            self.build_circuit(sdk="qiskit")
        
        # Create Aer simulator
        simulator = AerSimulator()
        
        # Transpile circuit
        transpiled = transpile(
            self.solver.main_circuit,
            backend=simulator,
            optimization_level=opt_level
        )
        
        # Run simulation
        job = simulator.run(transpiled, shots=shots, memory=memory)
        result = job.result()
        
        # Extract results
        counts = result.get_counts()
        output = {
            'counts': counts,
            'metadata': {
                'shots': shots,
                'backend': 'AerSimulator',
                'opt_level': opt_level,
                'success': result.success
            }
        }
        
        if memory:
            output['memory'] = result.get_memory()
        
        return output
    
    def report_resources(self) -> Dict[str, Any]:
        """
        Get resource estimates for the quantum circuit.
        
        Returns:
            Dictionary with:
                - 'problem': Problem size metrics (u_size, s_size)
                - 'estimated': Theoretical resource estimates
                - 'actual': Actual circuit resources (if built)
        
        Example:
            >>> qec = QExactCover(problem)
            >>> resources = qec.report_resources()
            >>> print(f"Requires {resources['estimated']['n_qubits']} qubits")
        """
        resources = {
            'problem': {
                'universe_size': len(self.problem.universe),
                'num_subsets': len(self.problem.subsets),
                'num_solutions': self.problem.num_solutions
            },
            'estimated': self.solver.resource_estimation()
        }
        
        # Add actual circuit metrics if built
        if self.solver.main_circuit is not None:
            if hasattr(self.solver.main_circuit, 'num_qubits'):
                # Qiskit circuit
                resources['actual'] = {
                    'n_qubits': self.solver.main_circuit.num_qubits,
                    'depth': self.solver.main_circuit.depth(),
                    'size': self.solver.main_circuit.size()
                }
            elif hasattr(self.solver.main_circuit, 'n_qubits'):
                # PyTKET circuit
                resources['actual'] = {
                    'n_qubits': self.solver.main_circuit.n_qubits,
                    'depth': self.solver.main_circuit.depth(),
                    'n_gates': self.solver.main_circuit.n_gates
                }
        
        return resources
    
    @staticmethod
    def create_small_example(cache_base: Optional[str] = None) -> 'QExactCover':
        """
        Create QExactCover with a small example problem.
        
        Args:
            cache_base: Optional cache directory
        
        Returns:
            QExactCover instance with small example problem
        
        Example:
            >>> qec = QExactCover.create_small_example()
            >>> print(f"Problem has {len(qec.universe)} elements")
        """
        problem = ExactCoverProblem.create_small_example()
        return QExactCover(problem, cache_base=cache_base)
