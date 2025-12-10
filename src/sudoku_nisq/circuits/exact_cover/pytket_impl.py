# mypy: ignore-errors
import math
from copy import deepcopy
from pytket import Circuit, Qubit, OpType

# ---------------------------------------------------------------------------
# Gate Counter
# ---------------------------------------------------------------------------

class GateCounter:
    """Tracks gate counts during circuit construction.
    
    Uses clear naming convention:
    - Single qubit: H, X, Measure
    - Multi-controlled X: CX (1 control), CCX (2 controls), C3X (3 controls), etc.
    - Multi-controlled Z: CZ (1 control), CCZ (2 controls), C3Z (3 controls), etc.
    """
    
    def __init__(self):
        self.counts = {}
    
    def increment(self, gate_name: str, count: int = 1):
        """Increment the count for a specific gate type."""
        self.counts[gate_name] = self.counts.get(gate_name, 0) + count
    
    def add_counter(self, other: 'GateCounter', multiplier: int = 1):
        """Add counts from another counter, optionally multiplied."""
        for gate_name, count in other.counts.items():
            self.increment(gate_name, count * multiplier)
    
    def to_dict(self) -> dict:
        """Return the gate counts as a dictionary."""
        return self.counts.copy()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_MULTI_CONTROLS = 1000000000  # choose a threshold that fits your use case

def _compute_grover_iterations(solver) -> int:
    """Compute Grover iteration count with basic safety checks."""
    num_solutions = getattr(solver, "num_solutions", None)

    if num_solutions is None:
        raise ValueError(
            "solver.num_solutions is None. "
            "Grover iteration formula requires a known, positive number of solutions."
        )
    if num_solutions <= 0:
        raise ValueError(
            f"Grover iteration formula undefined for num_solutions={num_solutions}. "
            "Instance appears to have no solutions."
        )

    return math.floor(
        (math.pi / 4) * math.sqrt((2 ** solver.s_size) / num_solutions)
    )


def _add_multi_control_gate(circ: Circuit, op_type: OpType, qubits, max_controls=MAX_MULTI_CONTROLS, counter: GateCounter | None = None):
    """
    Add a multi-controlled gate (CnX, CnZ, etc.) with a sanity check on the
    number of control qubits.

    qubits: [control_0, control_1, ..., control_n, target]
    counter: Optional GateCounter to track gate usage
    """
    n_controls = len(qubits) - 1
    if n_controls <= max_controls:
        circ.add_gate(op_type, qubits)
        
        # Track gate in counter if provided
        if counter is not None:
            if op_type == OpType.CnX:
                if n_controls == 1:
                    counter.increment("CX")
                elif n_controls == 2:
                    counter.increment("CCX")
                else:
                    counter.increment(f"C{n_controls}X")
            elif op_type == OpType.CnZ:
                if n_controls == 1:
                    counter.increment("CZ")
                elif n_controls == 2:
                    counter.increment("CCZ")
                else:
                    counter.increment(f"C{n_controls}Z")
    else:
        raise ValueError(
            f"{op_type.name} with {n_controls} controls may be too large or "
            "inefficient. Consider redesigning the circuit or providing a "
            "custom decomposition."
        )

# ---------------------------------------------------------------------------
# Main circuit construction
# ---------------------------------------------------------------------------


def build_exact_cover_circuit(solver, decompose_cnz: bool = True):
    """Build exact cover circuit using PyTKET SDK.
    
    Args:
        solver: Solver instance with problem parameters
        decompose_cnz: If True (default), count CnZ gates as their H+MCX+H 
                       decomposition for consistency with Qiskit. If False,
                       count CnZ as a single gate operation.
    
    Returns:
        tuple: (circuit, gate_counts_dict) where gate_counts_dict contains
               the count of each fundamental gate type used.
    """
    # Initialize gate counter
    counter = GateCounter()
    
    # Initialize circuits
    main_circuit = Circuit()
    oracle = Circuit()
    diffuser = Circuit()
    count_circuit = Circuit()
    count_circuit_dag = Circuit()
    aux_circ = Circuit()

    # Generate a register for the subsets
    s_qubits = [Qubit("S", i) for i in range(solver.s_size)]

    # Add subset qubits to the main circuit
    for q in s_qubits:
        main_circuit.add_qubit(q)

    # Apply Hadamard to each qubit in the main circuit
    for q in s_qubits:
        main_circuit.H(q)
        counter.increment("H")

    # Add the subset register to the counting, diffuser and auxiliary circuits
    for q in s_qubits:
        count_circuit.add_qubit(q)
        diffuser.add_qubit(q)
        aux_circ.add_qubit(q)
    
    # For each element u_i in U, add qubits U_i[0], ... , U_i[b]
    u_qubits = []
    for i in range(solver.u_size):
        label = f"U_{i}"
        u_label_qubits = [Qubit(label, j) for j in range(solver.b)]
        u_qubits.extend(u_label_qubits)

    # Add the U_{i} registers to the main, counting, oracle and auxiliary circuits
    for q in u_qubits:
        main_circuit.add_qubit(q)
        count_circuit.add_qubit(q)
        oracle.add_qubit(q)
        aux_circ.add_qubit(q)

    # Add the ancilla
    anc = Qubit("anc")
    main_circuit.add_qubit(anc)
    oracle.add_qubit(anc)
    aux_circ.add_qubit(anc)
    main_circuit.add_gate(OpType.X, [anc])
    counter.increment("X")
    main_circuit.add_gate(OpType.H, [anc])
    counter.increment("H")
    
    # Build sub-circuits with gate counting
    count_counter = GateCounter()
    _build_counter_pytket(solver, count_circuit, s_qubits, u_qubits, count_counter)
    count_circuit_dag = count_circuit.dagger()

    oracle_counter = GateCounter()
    _build_oracle_pytket(oracle, solver, u_qubits, anc, oracle_counter)
    
    diffuser_counter = GateCounter()
    _build_diffuser_pytket(diffuser, s_qubits, diffuser_counter, decompose_cnz=decompose_cnz)

    # Assemble auxiliary circuit
    aux_circ.append(count_circuit)
    aux_circ.append(oracle)
    aux_circ.append(count_circuit_dag)

    # Calculate Grover iterations (with safety checks)
    num_iterations = _compute_grover_iterations(solver)

    # Append sub-circuits to the main circuit
    for _ in range(num_iterations):
        main_circuit.append(aux_circ)
        main_circuit.append(diffuser)
    
    # Aggregate gate counts: COUNT + ORACLE + COUNT† per iteration, plus DIFFUSER per iteration
    # aux_circ contains: COUNT, ORACLE, COUNT_dagger
    counter.add_counter(count_counter, multiplier=num_iterations)  # COUNT
    counter.add_counter(oracle_counter, multiplier=num_iterations)  # ORACLE
    counter.add_counter(count_counter, multiplier=num_iterations)  # COUNT† (same gates as COUNT)
    counter.add_counter(diffuser_counter, multiplier=num_iterations)  # DIFFUSER

    # Add measurements
    c_bits = main_circuit.add_c_register("c", solver.s_size)
    for q in s_qubits:
        main_circuit.Measure(q, c_bits[q.index[0]])
        counter.increment("Measure")
        
    return main_circuit, counter.to_dict()

def _build_counter_pytket(solver, count_circuit, s_qubits, u_qubits, counter: GateCounter | None = None):
    """Build counting circuit using PyTKET.
    
    Args:
        counter: Optional GateCounter to track gate usage
    """
    all_lists = []  # This will store all generated lists
    j = 0  # Index for the S qubit corresponding to the S_j subset

    for subset in solver.subsets:
        q_list = []
        for elementU in solver.subsets[subset]:
            S_list = []
            S_list.append(Qubit("S", j))
            # Access register corresponding to the element u_i in subset S_subset
            i = solver.universe.index(elementU)
            label = f"U_{i}"
            # Use equality rather than startswith for safety
            register = [q for q in count_circuit.qubits if q.reg_name == label]
            for q in register:
                S_list.append(q)
                q_list.append(deepcopy(S_list))
        all_lists.append(q_list)
        j += 1

    # Reverse the lists because of the construction in the previous step
    reversed_lists = []
    for element in all_lists:
        reversed_element: list = element[::-1]  # type: ignore[assignment]
        reversed_lists.append(reversed_element)

    # Add the MCX gates to the counting circuit
    for element in reversed_lists:
        for q_list in element:
            _add_multi_control_gate(count_circuit, OpType.CnX, q_list, counter=counter)

def _build_oracle_pytket(oracle, solver, u_qubits, anc, counter: GateCounter | None = None):
    """Build oracle circuit using PyTKET.
    
    Args:
        counter: Optional GateCounter to track gate usage
    """
    # Apply X gates to all U qubits except those corresponding to zero
    x_count = 0
    for q in u_qubits:
        if q.index[0] != 0:
            oracle.X(q)
            x_count += 1
    
    if counter is not None:
        counter.increment("X", x_count)

    # Prepare the list of qubits for the multi-controlled X gate
    oracle_qubits_list = list(u_qubits) + [anc]
    _add_multi_control_gate(oracle, OpType.CnX, oracle_qubits_list, counter=counter)

    # Apply X gates again to revert the qubits
    if counter is not None:
        counter.increment("X", x_count)

def _build_diffuser_pytket(diffuser, s_qubits, counter: GateCounter | None = None, decompose_cnz: bool = True):
    """Build diffuser circuit using PyTKET.
    
    Args:
        counter: Optional GateCounter to track gate usage
        decompose_cnz: If True (default), count CnZ as H+MCX+H decomposition 
                       for consistency with Qiskit. If False, count as CnZ.
    """
    diffuser_qubits_list = []
    for q in s_qubits:
        diffuser.H(q)
        diffuser.X(q)
        diffuser_qubits_list.append(q)
    
    if counter is not None:
        counter.increment("H", len(s_qubits))
        counter.increment("X", len(s_qubits))

    # Handle CnZ gate counting based on decompose_cnz flag
    if decompose_cnz and counter is not None:
        # Count as H+MCX+H decomposition for consistency with Qiskit
        n_controls = len(diffuser_qubits_list) - 1
        counter.increment("H", 2)  # H gates wrapping the MCX
        if n_controls == 1:
            counter.increment("CX")
        elif n_controls == 2:
            counter.increment("CCX")
        else:
            counter.increment(f"C{n_controls}X")
        _add_multi_control_gate(diffuser, OpType.CnZ, diffuser_qubits_list, counter=None)
    else:
        # Count CnZ as a single gate operation
        _add_multi_control_gate(diffuser, OpType.CnZ, diffuser_qubits_list, counter=counter)

    for q in s_qubits:
        diffuser.X(q)
        diffuser.H(q)
    
    if counter is not None:
        counter.increment("X", len(s_qubits))
        counter.increment("H", len(s_qubits))