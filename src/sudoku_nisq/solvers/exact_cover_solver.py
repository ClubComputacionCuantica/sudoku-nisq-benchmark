import math
import mpmath
from typing import Literal

from sudoku_nisq.quantum_solver import QuantumSolver
from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding

class ExactCoverQuantumSolver(QuantumSolver):
    """
    Quantum solver that constructs circuits to solve Sudoku puzzles using exact cover
    formulation with Grover's algorithm.
    
    Based on: J. -R. Jiang and Y. -J. Wang, "Quantum Circuit Based on Grover's Algorithm 
    to Solve Exact Cover Problem," 2023 VTS Asia Pacific Wireless Communications 
    Symposium (APWCS), Tainan city, Taiwan, 2023.
    """
    
    def __init__(self, puzzle=None, metadata_manager=None, encoding: Literal["simple", "pattern"] = "simple",
                 num_solutions=None, universe=None, subsets=None, **kwargs):
        """Initialize the ExactCoverQuantumSolver with Sudoku puzzle and configuration."""
        # Initialize the base class with all parameters
        super().__init__(
            puzzle=puzzle,
            metadata_manager=metadata_manager,
            encoding=encoding,
            **kwargs
        )
                    
        # Initialize encoding
        enc = ExactCoverEncoding(puzzle)
        if universe is not None:
            self.universe = universe
        else:
            # Select correct universe depending on puzzle size
            if hasattr(enc, 'universe'):
                self.universe = enc.universe
            else:
                self.universe = enc.universe2x2
        
        # Determine which encoding to use
        if encoding == "simple":
            self.subsets = subsets if subsets is not None else enc.simple_subsets
        elif encoding == "pattern":
            self.subsets = subsets if subsets is not None else enc.pattern_subsets
        else:
            raise ValueError(f"Unknown encoding {encoding!r}")
        
        # Set number of solutions
        if num_solutions is None:
            self.num_solutions = puzzle.num_solutions if hasattr(puzzle, 'num_solutions') else 1
        else:
            self.num_solutions = num_solutions
            
        self.u_size = len(self.universe)        # Total elements to cover
        self.s_size = len(self.subsets)         # Number of subsets
        self.b = math.ceil(math.log2(self.s_size)) if self.s_size > 1 else 1  # Bits for counting

    def _build_sdk_circuit(self, sdk_type: str):
        """Build exact cover circuit using the specified SDK."""
        if sdk_type == "pytket":
            from sudoku_nisq.circuits.exact_cover.pytket_impl import build_exact_cover_circuit
            return build_exact_cover_circuit(self)
        elif sdk_type == "qiskit":
            from sudoku_nisq.circuits.exact_cover.qiskit_impl import build_exact_cover_circuit
            return build_exact_cover_circuit(self)
        elif sdk_type == "braket":
            # For now, fallback to pytket
            from sudoku_nisq.circuits.exact_cover.pytket_impl import build_exact_cover_circuit
            return build_exact_cover_circuit(self)
        else:
            raise ValueError(f"Unsupported SDK type: {sdk_type}")

    def resource_estimation(self):
        """Estimate quantum computational resources required for the exact cover algorithm."""
        s_size = self.s_size
        num_solutions = self.num_solutions

        # Set the decimal precision
        mpmath.mp.dps = 50  # Adjust as needed for precision

        # Compute logarithms to avoid large numbers
        ln_2 = mpmath.log(2)
        ln_pi_over_4 = mpmath.log(mpmath.pi / 4)
        ln_num_solutions = mpmath.log(num_solutions)

        # Calculate ln_a
        ln_a = (s_size * ln_2 - ln_num_solutions) / 2

        # Calculate ln_num_iterations
        ln_num_iterations = ln_pi_over_4 + ln_a

        # Compute num_iterations without overflow
        num_iterations = int(mpmath.floor(mpmath.exp(ln_num_iterations)))

        # Calculate the number of qubits
        num_qubits = self.s_size + self.u_size * self.b + 1
        
        # Gate counts
        superpos_gates = self.s_size
        prepare_anc_gates = 2
        counter_gates = 0
        for s in self.subsets:
            counter_gates += len(self.subsets[s]) * self.b
        oracle_gates = 1 + 2 * ((self.u_size - 1) * self.b)
        diffuser_gates = 1 + 4 * self.s_size
        MCX_gates = num_iterations * (oracle_gates + 2 * counter_gates)
        total_gates = (superpos_gates + prepare_anc_gates +
                    MCX_gates + num_iterations * diffuser_gates)
        return {
            "n_qubits": num_qubits,
            "MCX_gates": MCX_gates,
            "n_gates": total_gates,
            "depth": None  # Depth is not calculated here
        }

    def _is_valid_solution(self, bitstring: str) -> bool:
        """Check if a bitstring represents a valid exact cover solution.
        
        A bitstring is valid if the subsets it selects cover all universe 
        elements exactly once. Each bit position corresponds to a subset,
        where '1' means the subset is selected.
        
        Args:
            bitstring (str): Binary string where bit i indicates whether 
                subset S_i is selected (MSB is S_0 in standard Qiskit ordering).
        
        Returns:
            bool: True if the selected subsets form a valid exact cover.
        
        Example:
            For bitstring "110", subsets S_0 and S_1 are selected.
            Valid if S_0 ∪ S_1 covers all universe elements exactly once.
        """
        # Convert bitstring to selected subset indices
        # Qiskit uses big-endian: leftmost bit is qubit 0
        selected_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
        
        # Collect all universe elements covered by selected subsets
        covered_elements = []
        for idx in selected_indices:
            subset_key = f'S_{idx}'
            if subset_key in self.subsets:
                covered_elements.extend(self.subsets[subset_key])
        
        # Check two conditions:
        # 1. Each element appears exactly once (no duplicates)
        # 2. All universe elements are covered
        return (len(covered_elements) == len(set(covered_elements)) and 
                set(covered_elements) == set(self.universe))