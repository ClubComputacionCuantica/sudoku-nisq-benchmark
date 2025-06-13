from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class BacktrackingQuantumSolver(QuantumSolver):
    """
    Solve Sudoku using A. Montanaro's quantum backtracking algorithm encoded in a quantum circuit.
    """

    def __init__(self):
        super().__init__()

    def get_circuit(self) -> Circuit:
        circuit = Circuit()
        return circuit