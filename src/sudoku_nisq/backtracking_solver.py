from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class BacktrackingQuantumSolver(QuantumSolver):
    """
    Solve Sudoku using quantum backtracking algorithm encoded in a quantum circuit.
    """

    def __init__(self,sudoku=None):
        super().__init__(sudoku=sudoku)

    def build_circuit(self) -> Circuit:
        circuit = Circuit()
        return circuit
    
    def resource_estimation(self):
        return {
            "n_qubits": None,
            "MCX_gates": None,
            "n_gates": None,
            "depth": None,
            "error": "Not implemented"
        }