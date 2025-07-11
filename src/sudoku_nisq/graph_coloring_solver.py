from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class GraphColoringQuantumSolver(QuantumSolver):
    """
    Solve Sudoku using graph coloring Grover-based algorithm encoded in a quantum circuit.
    """

    def __init__(self):
        super().__init__()

    def get_circuit(self) -> Circuit:
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