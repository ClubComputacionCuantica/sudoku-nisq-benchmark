import math
import mpmath
from copy import deepcopy
from pytket import Circuit, Qubit, OpType
from sudoku_nisq.exact_cover_encoding import ExactCoverEncoding

from sudoku_nisq.quantum_solver import QuantumSolver

class ExactCoverQuantumSolver(QuantumSolver):
    """
    Solve Exact Cover via a Grover-based encoding.
    """
    def __init__(self, sudoku=None, num_solutions=None, simple=True, pattern=False,
                 universe=None, subsets=None):
        """
        Initialize the ExactCoverQuantumSolver instance.

        Parameters:
        - sudoku: The Sudoku puzzle to solve.
        - simple (bool): Whether to use simple encoding.
        - pattern (bool): Whether to use pattern encoding.
        - num_solutions (int): The number of expected solutions.
        """
        # Initialize the base class
        super().__init__()
        
        if sudoku is not None:
            
            # Initialize encoding
            encoding = ExactCoverEncoding(sudoku)
            self.universe = encoding.universe
            
            # Determine which encoding to use
            if simple is True:
                subsets = encoding.simple_subsets
            if pattern is True:
                subsets = encoding.pattern_subsets
            
            self.subsets = subsets

        if sudoku is None:
            self.universe = universe
            self.subsets = subsets
        
        if num_solutions is None:
            self.num_solutions = sudoku.count_solutions()
        else:
            self.num_solutions = num_solutions
            
        self.u_size = len(self.universe)        # Total elements to cover
        self.s_size = len(subsets)              # Number of subsets
        self.b = math.ceil(math.log2(self.s_size))  # Qubits for counting

    def find_resources(self, num_iterations=None):
        """
        Calculate the resources needed for the Exact Cover problem.
        Parameters:
        - num_iterations (int): Number of Grover iterations. If None, it is calculated automatically.
        Returns:
        - num_qubits (int): Number of qubits required.
        - MCX_gates (int): Number of multi-controlled X gates required.
        - total_gates (int): Total number of gates required.
        """
        # If num_iterations is not specified, calculate it based on the size of the problem
        if num_iterations is None:
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

        return num_qubits, MCX_gates, total_gates

    def get_circuit(self):
        """
        Builds and returns the full quantum circuit for the Exact Cover problem.

        Returns:
        - self.main_circuit (Circuit): The constructed quantum circuit.
        
        """
        # Returns the fully assembled circuit
                # Initialize circuits
        self.main_circuit = Circuit()
        self.oracle = Circuit()
        self.diffuser = Circuit()
        self.count_circuit = Circuit()
        self.count_circuit_dag = Circuit()
        self.aux_circ = Circuit()

        # Generate a register for the subsets
        self.s_qubits = [Qubit("S",i) for i in range(self.s_size)]

        # Add subset qubits to the main circuit
        for q in self.s_qubits:
            self.main_circuit.add_qubit(q)

        # Apply Hadamard to each qubit in the main circuit
        for q in self.s_qubits:
            self.main_circuit.H(q)

        # Add the subset register to the counting, diffuser and auxiliary circuits
        for q in self.s_qubits:
            self.count_circuit.add_qubit(q)
            self.diffuser.add_qubit(q)
            self.aux_circ.add_qubit(q)
        
        '''
        For each element u_i in U, we add qubits U_i[0], ... , U_i[b] for implementing
        the counter of the element u_i to store the number of subsets covering it
        '''
        self.u_qubits = []
        for i in range(self.u_size):
            label = f"U_{i}"
            u_label_qubits = [Qubit(label,j) for j in range(self.b)]
            self.u_qubits.extend(u_label_qubits)

        # Add the U_{i} registers to the main, counting, oracle and auxiliary circuits
        for q in self.u_qubits:
            self.main_circuit.add_qubit(q)
            self.count_circuit.add_qubit(q)
            self.oracle.add_qubit(q)
            self.aux_circ.add_qubit(q)

        # Add the ancilla
        self.anc = Qubit("anc")
        self.main_circuit.add_qubit(self.anc)
        self.oracle.add_qubit(self.anc)
        self.aux_circ.add_qubit(self.anc)
        self.main_circuit.add_gate(OpType.X, [self.anc])
        self.main_circuit.add_gate(OpType.H, [self.anc])
        
        self._assemble_full_circuit_w_meas()
        return self.main_circuit

    def _build_counter(self):
        """
        Constructs the counting circuit that counts the number of subsets covering each element.

        For each element u_i in U, we have a set of qubits U_i[0], ..., U_i[b-1] used to count
        the number of subsets covering u_i.

        The counting is performed using controlled increment operations based on the S qubits
        and the subsets they represent.
        """
    
        ## Generate lists for creating the MCX gates for the counters
        # The following section takes a subset and creates a list of the qubits that will be
        # controls and targets for the MCX gate for generating the counters
        
        # Initialize list to store the MCX gate qubit lists
        all_lists = []  # This will store all generated lists
        j = 0  # Index for the S qubit corresponding to the S_j subset
        for subset in self.subsets:
            q_list = []
            for elementU in self.subsets[subset]:
                S_list = []  # Generate a list to contain all lists of qubits to add a MCX
                S_list.append(Qubit("S", j))
                # Access register corresponding to the element u_i in subset S_subset
                i = self.universe.index(elementU)
                label = f"U_{i}"
                register = [q for q in self.count_circuit.qubits if q.reg_name.startswith(label)]
                for q in register:
                    S_list.append(q)
                    q_list.append(deepcopy(S_list))
            all_lists.append(q_list)
            j += 1
        
        # We reverse the list because of the construction in the previous step leaves them in 
        # reverse order
        reversed_lists = []
        for element in all_lists:
            reversed_element = element[::-1]
            reversed_lists.append(reversed_element)
        
        # Add the MCX gates to the counting circuit
        for element in reversed_lists:
            for q_list in element:
                self.count_circuit.add_gate(OpType.CnX, q_list)

        # Create the dagger (inverse) of the counting circuit
        self.count_circuit_dag = self.count_circuit.dagger()

    def _build_oracle(self):
        """
        Constructs the diffuser circuit used in Grover's algorithm.
        """
        
        # Apply X gates to all U qubits except those corresponding to zero
        for q in self.u_qubits:
            if q.index[0] != 0:
                    self.oracle.X(q)
                    
        # Prepare the list of qubits for the multi-controlled X gate
        oracle_qubits_list = []
        for q in self.u_qubits:
            oracle_qubits_list.append(q)
        oracle_qubits_list.append(self.anc)
        self.oracle.add_gate(OpType.CnX, oracle_qubits_list)
        
        # Apply X gates again to revert the qubits
        for q in self.u_qubits:
            if q.index[0] != 0:
                    self.oracle.X(q)

    def _build_diffuser(self):
        """
        Constructs the diffuser circuit used in Grover's algorithm.
        """
        diffuser_qubits_list = []
        for q in self.s_qubits:
            self.diffuser.H(q)
            self.diffuser.X(q)
            diffuser_qubits_list.append(q)
        
        self.diffuser.add_gate(OpType.CnZ, diffuser_qubits_list)
        
        for q in self.s_qubits:
            self.diffuser.X(q)
            self.diffuser.H(q)

    def _assemble_aux_circ(self):
        """
        Assembles the auxiliary circuit which includes the counter and oracle.
        """
        self._build_counter()
        self._build_oracle()
        self.aux_circ.append(self.count_circuit)
        self.aux_circ.append(self.oracle)
        self.aux_circ.append(self.count_circuit_dag)

    def _assemble_full_circuit_w_meas(self, num_iterations = None):
        """
        Assembles the full circuit including auxiliary circuits and measurements.

        Parameters:
        - num_iterations (int): Number of Grover iterations. If None, it is calculated automatically.
        """
        self._assemble_aux_circ()
        self._build_diffuser()
        
        if num_iterations is None:
            num_iterations = math.floor((math.pi / 4) * math.sqrt((2 ** self.s_size) / self.num_solutions))
        
        # Append sub-circuits to the main circuit
        for i in range(num_iterations):
            self.main_circuit.append(self.aux_circ)
            self.main_circuit.append(self.diffuser)

        c_bits = self.main_circuit.add_c_register("c", self.s_size)
        for q in self.s_qubits:
            self.main_circuit.Measure(q, c_bits[q.index[0]])