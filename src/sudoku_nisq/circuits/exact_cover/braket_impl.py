from __future__ import annotations

import math
from copy import deepcopy
from typing import Iterable, List

try:
    # Amazon Braket SDK
    from braket.circuits import Circuit
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Amazon Braket SDK not available. Install with: pip install amazon-braket-sdk"
    ) from e


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

def _flatten_registers(regs: Iterable[Iterable[int]]) -> list[int]:
    """Flatten iterable of 'registers' (each is a list of qubit indices) into one list."""
    return [q for reg in regs for q in reg]


def _compute_grover_iterations(solver) -> int:
    """Compute Grover iteration count with basic safety checks.

    Same formula as before, just reused in Braket-land.
    Requires a positive `solver.num_solutions`.
    """
    num_solutions = getattr(solver, "num_solutions", None)
    if num_solutions is None:
        raise ValueError(
            "solver.num_solutions is None; Grover iteration formula requires "
            "a known, positive number of solutions."
        )
    if num_solutions <= 0:
        raise ValueError(
            f"Grover iteration formula undefined for num_solutions={num_solutions}."
        )
    return math.floor((math.pi / 4) * math.sqrt((2 ** solver.s_size) / num_solutions))


# ---------------------------------------------------------------------------
# Subcircuit builders (Braket)
# ---------------------------------------------------------------------------

def _build_counter_braket(
    solver,
    S_qubits: List[int],
    U_regs: List[List[int]],
    counter: GateCounter = None,
) -> Circuit:
    """Build the counting subcircuit for Braket.

    Qubit order in the subcircuit is implicit via indices:
    [S..., U_0..., U_1..., ..., U_{u-1}...]
    
    Args:
        counter: Optional GateCounter to track gate usage
    """

    count = Circuit()

    all_lists: list[list[list[int]]] = []
    # Map universe element -> index
    universe_index = {u: i for i, u in enumerate(solver.universe)}

    for j, subset_key in enumerate(solver.subsets):
        per_subset_lists: list[list[int]] = []
        for elementU in solver.subsets[subset_key]:
            i = universe_index[elementU]
            # Build incremental control lists: [S_j, U_i[0]], [S_j, U_i[0], U_i[1]], ...
            growing = [S_qubits[j]]
            for q in U_regs[i]:
                growing.append(q)
                per_subset_lists.append(deepcopy(growing))
        # Reverse to match previous ordering convention
        per_subset_lists = list(reversed(per_subset_lists))
        all_lists.append(per_subset_lists)

    for per_subset in all_lists:
        for q_list in per_subset:
            if len(q_list) == 1:
                # Single-target X
                count.x(q_list[0])
                if counter is not None:
                    counter.increment("X")
            else:
                controls = q_list[:-1]
                target = q_list[-1]
                # Multi-controlled X using Braket gate modifiers
                # NOTE: Multi-control via `control=` is only guaranteed on local simulators.
                count.x(target, control=controls)
                if counter is not None:
                    n_controls = len(controls)
                    if n_controls == 1:
                        counter.increment("CX")
                    elif n_controls == 2:
                        counter.increment("CCX")
                    else:
                        counter.increment(f"C{n_controls}X")

    return count


def _build_oracle_braket(
    U_regs: List[List[int]],
    anc_qubit: int,
    counter: GateCounter = None,
) -> Circuit:
    """Build the oracle subcircuit:

    - Flip all non-zero index bits in each U_i register
    - Apply a big multi-controlled X on ancilla
    - Uncompute the flips
    
    Args:
        counter: Optional GateCounter to track gate usage
    """

    oracle = Circuit()

    # Flip all non-zero index bits in each U_i register
    x_count = 0
    for reg in U_regs:
        for bit_index, q in enumerate(reg):
            if bit_index != 0:
                oracle.x(q)
                x_count += 1
    
    if counter is not None:
        counter.increment("X", x_count)

    controls = _flatten_registers(U_regs)
    if controls:
        oracle.x(anc_qubit, control=controls)
        if counter is not None:
            n_controls = len(controls)
            if n_controls == 1:
                counter.increment("CX")
            elif n_controls == 2:
                counter.increment("CCX")
            else:
                counter.increment(f"C{n_controls}X")
    else:
        # Degenerate case: empty universe
        oracle.x(anc_qubit)
        if counter is not None:
            counter.increment("X")

    # Uncompute flips
    if counter is not None:
        counter.increment("X", x_count)
    for reg in U_regs:
        for bit_index, q in enumerate(reg):
            if bit_index != 0:
                oracle.x(q)

    return oracle


def _build_diffuser_braket(S_qubits: List[int], counter: GateCounter = None, decompose_cnz: bool = True) -> Circuit:
    """Standard Grover diffuser on subset register S in Braket.
    
    Args:
        counter: Optional GateCounter to track gate usage
        decompose_cnz: If True (default), count CnZ as H+MCX+H decomposition 
                       for consistency with Qiskit. If False, count as CnZ.
    """

    diff = Circuit()

    # First layer: H then X on each qubit
    for q in S_qubits:
        diff.h(q)
        diff.x(q)
    
    if counter is not None:
        counter.increment("H", len(S_qubits))
        counter.increment("X", len(S_qubits))

    n = len(S_qubits)
    if n == 0:
        # Nothing to do
        return diff

    if n == 1:
        q = S_qubits[0]
        diff.h(q)
        diff.z(q)
        diff.h(q)
        if counter is not None:
            counter.increment("H", 2)
            counter.increment("Z")
    else:
        last = S_qubits[-1]
        controls = S_qubits[:-1]

        diff.h(last)
        # Multi-controlled Z implemented as H-mapped multi-controlled X
        diff.x(last, control=controls)
        diff.h(last)
        
        if counter is not None:
            if decompose_cnz:
                # Count as H+MCX+H decomposition for consistency with Qiskit
                n_controls = len(controls)
                counter.increment("H", 2)  # H gates wrapping the MCX
                if n_controls == 1:
                    counter.increment("CX")
                elif n_controls == 2:
                    counter.increment("CCX")
                else:
                    counter.increment(f"C{n_controls}X")
            else:
                # Count CnZ as a single gate operation
                n_controls = len(controls)
                if n_controls == 1:
                    counter.increment("CZ")
                elif n_controls == 2:
                    counter.increment("CCZ")
                else:
                    counter.increment(f"C{n_controls}Z")

    # Final layer: X then H on each qubit
    for q in S_qubits:
        diff.x(q)
        diff.h(q)
    
    if counter is not None:
        counter.increment("X", len(S_qubits))
        counter.increment("H", len(S_qubits))

    return diff


# ---------------------------------------------------------------------------
# Public builder (Braket)
# ---------------------------------------------------------------------------

def build_exact_cover_circuit(solver, decompose_cnz: bool = True):
    """Build the full Grover search circuit for the exact cover instance on Braket.

    Required solver fields:
        - s_size: size of subset register S
        - u_size: number of universe elements
        - b: number of bits per universe-element register U_i
        - subsets: dict-like describing subsets (same as Qiskit version)
        - universe: list giving the universe, used to index U registers
        - num_solutions: (estimated) number of valid exact covers

    Qubit layout (logical indices in the Braket Circuit):

        S register: qubits [0, ..., s_size - 1]
        U_0:       [s_size, ..., s_size + b - 1]
        U_1:       [s_size + b, ..., s_size + 2b - 1]
        ...
        U_{u-1}:   [s_size + (u_size-1)*b, ..., s_size + u_size*b - 1]
        ancilla:   [s_size + u_size * b]

    Measurements:
        We add `measure` on the S register. When you run the circuit with
        `device.run(circuit, shots=...)`, you can interpret the measured bits
        at these S-qubit indices as the subset-selection string.
    
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
    
    # ------------------------------------------------------------------
    # Allocate logical qubit indices
    # ------------------------------------------------------------------
    s_size = solver.s_size
    u_size = solver.u_size
    b = solver.b

    S_qubits = list(range(s_size))

    U_regs: list[list[int]] = []
    for i in range(u_size):
        start = s_size + i * b
        U_regs.append(list(range(start, start + b)))

    anc_qubit = s_size + u_size * b

    # Total number of qubits is implicit; Braket grows the circuit as needed.
    main = Circuit()

    # ------------------------------------------------------------------
    # Initial state preparation
    # ------------------------------------------------------------------

    # Superposition over subsets S
    main.h(S_qubits)
    counter.increment("H", len(S_qubits))

    # Ancilla in |-> = (|0> - |1>) / sqrt(2)
    main.x(anc_qubit)
    counter.increment("X")
    main.h(anc_qubit)
    counter.increment("H")

    # ------------------------------------------------------------------
    # Build subcircuits
    # ------------------------------------------------------------------

    count_counter = GateCounter()
    count_circ = _build_counter_braket(solver, S_qubits, U_regs, count_counter)
    
    oracle_counter = GateCounter()
    oracle_circ = _build_oracle_braket(U_regs, anc_qubit, oracle_counter)
    
    diffuser_counter = GateCounter()
    diffuser_circ = _build_diffuser_braket(S_qubits, diffuser_counter, decompose_cnz=decompose_cnz)

    # Inverse of the counting circuit via adjoint
    count_inv_circ = count_circ.adjoint()

    # ------------------------------------------------------------------
    # Grover iteration count
    # ------------------------------------------------------------------
    try:
        num_iterations = _compute_grover_iterations(solver)
    except ValueError:
        # Fallback: 0 iterations if num_solutions invalid / unknown
        num_iterations = 0

    # ------------------------------------------------------------------
    # Grover loop: COUNT → ORACLE → COUNT† → DIFFUSER
    # ------------------------------------------------------------------
    for _ in range(num_iterations):
        main.add_circuit(count_circ)
        main.add_circuit(oracle_circ)
        main.add_circuit(count_inv_circ)
        main.add_circuit(diffuser_circ)  # operates only on S_qubits
    
    # Aggregate gate counts: COUNT + ORACLE + COUNT† + DIFFUSER per iteration
    counter.add_counter(count_counter, multiplier=num_iterations)  # COUNT
    counter.add_counter(oracle_counter, multiplier=num_iterations)  # ORACLE
    counter.add_counter(count_counter, multiplier=num_iterations)  # COUNT† (same gates as COUNT)
    counter.add_counter(diffuser_counter, multiplier=num_iterations)  # DIFFUSER

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    # Braket doesn't have classical registers; we just mark S_qubits for measurement.
    # When using device.run(...).result().measurement_counts, interpret S bits by index.
    main.measure(S_qubits)
    counter.increment("Measure", len(S_qubits))

    return main, counter.to_dict()
