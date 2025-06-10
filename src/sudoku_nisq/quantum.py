import math
import mpmath
from copy import deepcopy
from pytket import Circuit, Qubit, OpType

from sudoku_nisq.exact_cover_encoding import ExactCoverEncoding
from pytket.circuit.display import render_circuit_jupyter as draw
from qiskit_ibm_runtime import QiskitRuntimeService
from pytket.extensions.qiskit import IBMQBackend

from pytket.passes import FlattenRegisters

class ExactCoverQuantumSolver:
    """
    This module defines the ExactCoverQuantumSolver class, which constructs and simulates or
    runs a quantum circuit through a backend to solve any Exact Cover problem using 
    using an algorithm based on Grover's algorithm: 
        J. -R. Jiang and Y. -J. Wang, "Quantum Circuit Based on Grover’s Algorithm to Solve Exact Cover Problem," 2023
        VTS Asia Pacific Wireless Communications Symposium (APWCS), Tainan city, Taiwan, 2023, pp. 1-5, doi: 10.1109/APWCS60142.2023.10234054.

    Usage:
    - Initialize the solver with the problem instance.
    - Build the circuit using get_circuit().
    - Simulate the circuit using aer_simulation().
    - Analyze and plot the results using counts_plot().

    Example:
    # Define your problem instance
            sudoku = Sudoku()   This also allows to select a simple encoding or an encoding in patterns
        or general exact cover form
            U = [A, B, C]
            S = {'S_0': [U[2]], 'S_1': [U[0], U[2]], 'S_2': [U[0]], 'S_3': [U[1]], 'S_4': [U[0], U[1]]}
    
    # Initialize the solver
    solver = ExactCoverQuantumSolver(sudoku, simple=True, num_solutions=1)

    # Build the quantum circuit
    circuit = solver.get_circuit()
    
    # Draw the quantum circuit
    
    solver.draw_circuit()

    # Simulate the circuit
    results = solver.aer_simulation(shots=1024) -> self.sim_counts

    # Plot the results
    solver.counts_plot()
    
    """
    def __init__(self, sudoku = None, num_solutions = None, simple=True, pattern=False, universe = None, subsets = None):
        """
        Initialize the ExactCoverQuantumSolver instance.

        Parameters:
        - sudoku: The Sudoku puzzle to solve.
        - simple (bool): Whether to use simple encoding.
        - pattern (bool): Whether to use pattern encoding.
        - num_solutions (int): The number of expected solutions.
        """
        if sudoku is not None:
            print(sudoku)
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
        self.main_circuit = None
        self.backend = None
        
    def find_resources(self, num_iterations=None):
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
    
    def draw_circuit(self):
        draw(self.main_circuit)    
        
    def aer_simulation(self,shots=1024):
        """
        Simulates the circuit using AerBackend from pytket.

        Parameters:
        - shots (int): Number of shots for the simulation.

        Returns:
        - self.sim_counts (dict): The simulation counts.
        """
        # Check the number of qubits in the circuit
        num_qubits = self.main_circuit.n_qubits
        if num_qubits > 30:
            raise ValueError(f"Quantum resources exceed 30 qubits ({num_qubits} qubits used). Simulation aborted.")
        
        from pytket.extensions.qiskit import AerBackend
        from pytket.passes import FlattenRegisters
        
        # Compile and simulate the circuit
        backend = AerBackend()
        flatten = FlattenRegisters()
        flatten.apply(self.main_circuit)
        compiled_circ = backend.get_compiled_circuit(self.main_circuit)

        handle = backend.process_circuit(compiled_circ, n_shots=shots)
        self.sim_counts = backend.get_result(handle).get_counts()
        
        return self.sim_counts

    def counts_plot(self,counts = None):
        """
        Plots the probabilities of the measured states.

        Parameters:
        - counts (dict): The counts from simulation. If None, uses self.sim_counts.
        """
        if counts is None:
            counts = self.sim_counts
            
        import matplotlib.pyplot as plt
        from pytket.utils import probs_from_counts
        
        data = probs_from_counts(counts)
        # Extract keys and values
        keys = list(data.keys())
        values = list(data.values())

        # Convert keys to string representation for plotting
        keys_str = [str(key) for key in keys]

        # Create the bar plot
        plt.figure(figsize=(7, 5))
        plt.bar(keys_str, values, color='royalblue')

        # Add title and labels
        plt.title('Amplitudes')
        plt.xlabel('states')
        plt.ylabel('Amplitude')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def init_ibm(self, ibm_token):
        """
        Returns IBM backends available to your account
        """
        QiskitRuntimeService.save_account(channel="ibm_quantum", token=ibm_token, overwrite=True)
        available_backend_list = IBMQBackend.available_devices()
        return available_backend_list
    
    def flatten_reg(self):
        if self.main_circuit is None:
            self.get_circuit()
        flatten = FlattenRegisters()
        flatten.apply(self.main_circuit)
        print("Flattened once")
    
    def set_ibm_backend(self,ibm_backend="ibm_brisbane"):
        self.backend = IBMQBackend(ibm_backend)
    
    def ibm_transpile(self, ibm_backend=None, optimisation_level: int =0, flatten=None):
        if flatten is not None:
            self.flatten_reg()
        if self.backend is None:
            if ibm_backend is None:
                self.set_ibm_backend() # Use default backend
            else:
                self.set_ibm_backend(ibm_backend)
                
        return self.backend.get_compiled_circuit(self.main_circuit, optimisation_level=optimisation_level)
    
    def find_transp_resources(self,ibm_backend,optimisation_levels: list =[0,1,2],flatten=None):
        resources = []
        if flatten is not None:
            self.flatten_reg()
        for level in optimisation_levels:
            circ = self.ibm_transpile(ibm_backend=ibm_backend,optimisation_level=level)
            resources.append([circ.n_qubits, circ.n_gates, circ.depth()])
        return resources
        
    def ibm_handle(self,transpiled_circuit,n_shots=1024):
        # Pending
        if self.backend is None:
            raise RuntimeError("Circuit must be transpiled through ibm_transpile method")
        result_handle = self.backend.process_circuit(transpiled_circuit, n_shots=n_shots)
        return result_handle  # Returns the handle for further processing
            
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
    
    