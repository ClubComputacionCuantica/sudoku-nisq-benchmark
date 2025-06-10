from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class BacktrackingQuantumSolver(QuantumSolver):
    """
    Solve Sudoku using A. Montanaro's quantum backtracking algorithm encoded in a quantum circuit.
    """

    def __init__(self, problem=None):
        super().__init__()
        self.problem = problem

    def get_circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: encode problem & set up quantum walk or amplitude amplification
        return circuit
