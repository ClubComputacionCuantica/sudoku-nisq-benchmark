import math
import mpmath
from copy import deepcopy
from pytket import Circuit, Qubit, OpType
from sudoku_nisq.exact_cover_encoding import ExactCoverEncoding
from pytket.circuit.display import render_circuit_jupyter as draw
from qiskit_ibm_runtime import QiskitRuntimeService
from pytket.extensions.qiskit import IBMQBackend
from pytket.passes import FlattenRegisters

from sudoku_nisq.quantum_solver import QuantumSolver

class ExactCoverQuantumSolver(QuantumSolver):
    """
    Solve Exact Cover (and thus Sudoku) via a Grover-based encoding.
    """

    def __init__(self, sudoku=None, num_solutions=None, simple=True, pattern=False,
                 universe=None, subsets=None):
        super().__init__()
        # … your full init from before …
        # set self.universe, self.subsets, self.num_solutions, etc.

    def get_circuit(self) -> Circuit:
        # … build & return self.main_circuit …
        return self.main_circuit

    def draw_circuit(self):
        draw(self.main_circuit)

    # (All your other methods: aer_simulation, counts_plot,
    # init_ibm, flatten_reg, set_ibm_backend, ibm_transpile, etc.)
