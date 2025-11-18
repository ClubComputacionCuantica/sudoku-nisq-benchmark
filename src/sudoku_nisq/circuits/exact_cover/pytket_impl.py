import math
from copy import deepcopy
from pytket import Circuit, Qubit, OpType

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


def _add_multi_control_gate(circ: Circuit, op_type: OpType, qubits, max_controls=MAX_MULTI_CONTROLS):
    """
    Add a multi-controlled gate (CnX, CnZ, etc.) with a sanity check on the
    number of control qubits.

    qubits: [control_0, control_1, ..., control_n, target]
    """
    n_controls = len(qubits) - 1
    if n_controls <= max_controls:
        circ.add_gate(op_type, qubits)
    else:
        raise ValueError(
            f"{op_type.name} with {n_controls} controls may be too large or "
            "inefficient. Consider redesigning the circuit or providing a "
            "custom decomposition."
        )

# ---------------------------------------------------------------------------
# Main circuit construction
# ---------------------------------------------------------------------------


def build_exact_cover_circuit(solver):
    """Build exact cover circuit using PyTKET SDK."""
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
    main_circuit.add_gate(OpType.H, [anc])
    
    # Build sub-circuits
    _build_counter_pytket(solver, count_circuit, s_qubits, u_qubits)
    count_circuit_dag = count_circuit.dagger()

    _build_oracle_pytket(oracle, solver, u_qubits, anc)
    _build_diffuser_pytket(diffuser, s_qubits)

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

    # Add measurements
    c_bits = main_circuit.add_c_register("c", solver.s_size)
    for q in s_qubits:
        main_circuit.Measure(q, c_bits[q.index[0]])
        
    return main_circuit

def _build_counter_pytket(solver, count_circuit, s_qubits, u_qubits):
    """Build counting circuit using PyTKET."""
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
        reversed_element = element[::-1]
        reversed_lists.append(reversed_element)

    # Add the MCX gates to the counting circuit
    for element in reversed_lists:
        for q_list in element:
            _add_multi_control_gate(count_circuit, OpType.CnX, q_list)

def _build_oracle_pytket(oracle, solver, u_qubits, anc):
    """Build oracle circuit using PyTKET."""
    # Apply X gates to all U qubits except those corresponding to zero
    for q in u_qubits:
        if q.index[0] != 0:
            oracle.X(q)

    # Prepare the list of qubits for the multi-controlled X gate
    oracle_qubits_list = list(u_qubits) + [anc]
    _add_multi_control_gate(oracle, OpType.CnX, oracle_qubits_list)

    # Apply X gates again to revert the qubits
    for q in u_qubits:
        if q.index[0] != 0:
            oracle.X(q)

def _build_diffuser_pytket(diffuser, s_qubits):
    """Build diffuser circuit using PyTKET."""
    diffuser_qubits_list = []
    for q in s_qubits:
        diffuser.H(q)
        diffuser.X(q)
        diffuser_qubits_list.append(q)

    _add_multi_control_gate(diffuser, OpType.CnZ, diffuser_qubits_list)

    for q in s_qubits:
        diffuser.X(q)
        diffuser.H(q)