from pytket import Circuit
from sudoku_nisq.quantum_solver import QuantumSolver

class GraphColoringQuantumSolver(QuantumSolver):
    """
    Solver for graph-coloring using a quantum algorithm.
    """

    def __init__(self, graph=None, num_colors: int = 2):
        super().__init__()
        self.graph = graph
        self.num_colors = num_colors

    def get_circuit(self) -> Circuit:
        circuit = Circuit()
        # TODO: allocate qubits and encode coloring constraints
        return circuit
    
