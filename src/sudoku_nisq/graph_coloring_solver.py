from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class GraphColoringQuantumSolver(QuantumSolver):
    """
    Solver for graph-coloring using a quantum algorithm.
    """

    def __init__(self):
        super().__init__()

    def get_circuit(self) -> Circuit:
        circuit = Circuit()
        
        return circuit
