from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class BacktrackingQuantumSolver(QuantumSolver):
    """
    Solve Sudoku using quantum backtracking algorithm encoded in a quantum circuit.
    """

    def __init__(self, puzzle, metadata_manager, encoding=None, store_transpiled=True, **kwargs):
        """
        Initialize the BacktrackingQuantumSolver instance.
        
        Args:
            puzzle: The SudokuPuzzle instance to solve.
            metadata_manager: The MetadataManager instance for caching.
            encoding: Encoding strategy name (optional).
            store_transpiled: Whether to save transpiled circuits to disk (default True).
            **kwargs: Additional parameters passed to QuantumSolver base class
        """
        super().__init__(
            puzzle=puzzle, 
            metadata_manager=metadata_manager, 
            encoding=encoding, 
            store_transpiled=store_transpiled,
            **kwargs
        )

    def _build_circuit(self) -> Circuit:
        circuit = Circuit()
        return circuit
    
    def resource_estimation(self):
        pass