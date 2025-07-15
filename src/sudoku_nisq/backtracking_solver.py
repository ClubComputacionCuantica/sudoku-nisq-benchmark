from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class BacktrackingQuantumSolver(QuantumSolver):
    """
    Solve Sudoku using quantum backtracking algorithm encoded in a quantum circuit.
    """

    def __init__(self, sudoku=None, **kwargs):
        """
        Initialize the BacktrackingQuantumSolver instance.
        
        Args:
            sudoku: The Sudoku puzzle to solve.
            **kwargs: Additional parameters passed to QuantumSolver base class
        """
        super().__init__(sudoku=sudoku, **kwargs)

    def _build_circuit(self) -> Circuit:
        circuit = Circuit()
        return circuit
    
    def resource_estimation(self):
        pass