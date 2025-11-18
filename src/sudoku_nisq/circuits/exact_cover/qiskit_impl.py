from __future__ import annotations

import math
from copy import deepcopy
from typing import Iterable

try:  # Local import guard so upstream users can still use pytket only.
    from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
except ImportError as e:  # pragma: no cover - raised if user lacks qiskit
    raise ImportError(
        "Qiskit not available. Install with: pip install qiskit"
    ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_registers(regs: Iterable[QuantumRegister]) -> list:
    """Flatten iterable of QuantumRegister into a list of qubits."""
    return [q for reg in regs for q in reg]


def _compute_grover_iterations(solver) -> int:
    """Compute Grover iteration count with basic safety checks.

    Uses the same formula as the PyTKET implementation. Requires a positive
    `solver.num_solutions` value.
    """
    num_solutions = getattr(solver, "num_solutions", None)
    if num_solutions is None:
        raise ValueError(
            "solver.num_solutions is None; Grover iteration formula requires a known, positive number of solutions."
        )
    if num_solutions <= 0:
        raise ValueError(
            f"Grover iteration formula undefined for num_solutions={num_solutions}."
        )
    return math.floor((math.pi / 4) * math.sqrt((2 ** solver.s_size) / num_solutions))


# ---------------------------------------------------------------------------
# Subcircuit builders (native Qiskit)
# ---------------------------------------------------------------------------

def _build_counter_qiskit(solver, S_reg: QuantumRegister, U_regs: list[QuantumRegister]) -> QuantumCircuit:
    """Build the counting subcircuit.

    Reproduces the cascaded multi-controlled-X construction from the deprecated
    Qiskit implementation: for each subset S_j and each universe element u_i in
    that subset, we build growing control lists over the bits of register U_i.

    Qubit order in the returned circuit: [S..., U_0..., U_1..., ..., U_{u-1}...]
    """
    count = QuantumCircuit(S_reg, *U_regs, name="COUNT")

    all_lists: list[list[list]] = []
    j = 0
    # Optional speed-up map (not stored globally)
    universe_index = {u: i for i, u in enumerate(solver.universe)}
    for subset_key in solver.subsets:
        per_subset_lists: list[list] = []
        for elementU in solver.subsets[subset_key]:
            i = universe_index[elementU]
            # Build incremental lists: [S_j, U_i[0]], [S_j, U_i[0], U_i[1]], ...
            growing = [S_reg[j]]
            for q in U_regs[i]:
                growing.append(q)
                per_subset_lists.append(deepcopy(growing))
        # Reverse to match previous implementation ordering
        per_subset_lists = list(reversed(per_subset_lists))
        all_lists.append(per_subset_lists)
        j += 1

    for per_subset in all_lists:
        for q_list in per_subset:
            if len(q_list) == 1:  # single target, no controls
                count.x(q_list[0])
            else:
                count.mcx(q_list[:-1], q_list[-1])  # controls, target

    return count


def _build_oracle_qiskit(U_regs: list[QuantumRegister], anc_reg: QuantumRegister) -> QuantumCircuit:
    """Build the oracle subcircuit.

    Flips all U register qubits except index 0, performs a big MCX onto ancilla,
    then uncomputes the flips. Order: [U_0..., U_1..., ..., anc]
    """
    oracle = QuantumCircuit(*U_regs, anc_reg, name="ORACLE")

    # Flip all non-zero index bits in each U_i register
    for reg in U_regs:
        for bit_index, q in enumerate(reg):
            if bit_index != 0:
                oracle.x(q)

    controls = _flatten_registers(U_regs)
    target = anc_reg[0]
    if controls:  # typical case
        oracle.mcx(controls, target)
    else:  # degenerate edge case: no controls (empty universe)
        oracle.x(target)

    # Uncompute flips
    for reg in U_regs:
        for bit_index, q in enumerate(reg):
            if bit_index != 0:
                oracle.x(q)

    return oracle


def _build_diffuser_qiskit(S_reg: QuantumRegister) -> QuantumCircuit:
    """Standard Grover diffuser on subset register S."""
    diff = QuantumCircuit(S_reg, name="DIFFUSER")

    for q in S_reg:
        diff.h(q)
        diff.x(q)

    # Multi-controlled Z via H-mapped MCX if more than one qubit
    if len(S_reg) == 0:
        return diff  # nothing to do
    if len(S_reg) == 1:
        diff.h(S_reg[0])
        diff.z(S_reg[0])
        diff.h(S_reg[0])
    else:
        diff.h(S_reg[-1])
        diff.mcx(S_reg[:-1], S_reg[-1])  # implements C^(n-1)Z
        diff.h(S_reg[-1])

    for q in S_reg:
        diff.x(q)
        diff.h(q)

    return diff


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_exact_cover_circuit(solver):
    """Build and return the full Grover search circuit for the exact cover instance.

    The solver must provide: s_size, u_size, b, subsets (dict-like), universe (list),
    and num_solutions. Measurements are added onto a classical register `c` of size
    s_size for sampling subset selections that solve the exact cover.
    """
    # Registers
    S = QuantumRegister(solver.s_size, "S")
    U_regs = [QuantumRegister(solver.b, f"U_{i}") for i in range(solver.u_size)]
    anc = QuantumRegister(1, "anc")
    c_bits = ClassicalRegister(solver.s_size, "c")

    main = QuantumCircuit(S, *U_regs, anc, c_bits, name="MAIN")

    # Superposition over subsets
    for q in S:
        main.h(q)

    # Prepare ancilla in |-> state
    main.x(anc[0])
    main.h(anc[0])

    # Build subcircuits
    count_circ = _build_counter_qiskit(solver, S, U_regs)
    count_gate = count_circ.to_gate(label="COUNT")
    count_inv_gate = count_gate.inverse()

    oracle_circ = _build_oracle_qiskit(U_regs, anc)
    oracle_gate = oracle_circ.to_gate(label="ORACLE")

    diffuser_circ = _build_diffuser_qiskit(S)
    diffuser_gate = diffuser_circ.to_gate(label="DIFFUSER")

    # Determine Grover iterations
    try:
        num_iterations = _compute_grover_iterations(solver)
    except ValueError:
        # Fallback: 0 iterations (edge case) if invalid num_solutions
        num_iterations = 0

    # Precompute qubit ordering lists for appends
    S_list = list(S)
    U_flat = _flatten_registers(U_regs)
    count_order = S_list + U_flat
    oracle_order = U_flat + [anc[0]]

    # Grover iterations: COUNT → ORACLE → COUNT† → DIFFUSER
    for _ in range(num_iterations):
        main.append(count_gate, count_order)
        main.append(oracle_gate, oracle_order)
        main.append(count_inv_gate, count_order)
        main.append(diffuser_gate, S_list)

    # Measure subset register
    main.measure(S, c_bits)

    return main