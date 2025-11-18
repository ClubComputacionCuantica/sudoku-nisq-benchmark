import math
import mpmath
from copy import deepcopy
from typing import Literal

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import ZGate

from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding
from sudoku_nisq.qiskit_quantum_solver import QiskitQuantumSolver

class ExactCoverQiskitQuantumSolver(QiskitQuantumSolver):
    """
    Qiskit 2.2 translation of the pytket-based ExactCover solver.

    Subcircuits:
        - Counter: increments per-universe-element counters controlled on S_j.
        - Oracle: marks states where all counters == 1 (per your original construction).
        - Diffuser: standard Grover diffuser over S-register.

    Usage is identical to your original class.
    """

    def __init__(
        self,
        puzzle=None,
        metadata_manager=None,
        encoding: Literal["simple", "pattern"] = "simple",
        num_solutions=None,
        universe=None,
        subsets=None,
        **kwargs,
    ):
        super().__init__(
            puzzle=puzzle,
            metadata_manager=metadata_manager,
            encoding=encoding,
            **kwargs,
        )

        enc = ExactCoverEncoding(puzzle)
        self.universe = enc.universe

        if encoding == "simple":
            self.subsets = enc.simple_subsets
        elif encoding == "pattern":
            self.subsets = enc.pattern_subsets
        else:
            raise ValueError(f"Unknown encoding {encoding!r}")

        self.num_solutions = (
            puzzle.num_solutions if num_solutions is None else num_solutions
        )

        self.u_size = len(self.universe)          # |U|
        self.s_size = len(self.subsets)           # |S|
        # counter bits per universe element
        self.b = math.ceil(math.log2(self.s_size)) if self.s_size > 0 else 1

        # Placeholders to mirror original structure
        self.main_circuit: QuantumCircuit | None = None
        self.oracle: QuantumCircuit | None = None
        self.diffuser: QuantumCircuit | None = None
        self.count_circuit: QuantumCircuit | None = None
        self.count_circuit_dag: QuantumCircuit | None = None
        self.aux_circ: QuantumCircuit | None = None

    # ---------- Helpers ----------

    @staticmethod
    def _flatten_registers(reg_list):
        return [q for reg in reg_list for q in reg]

    def _build_counter_circuit(self, S_reg: QuantumRegister, U_regs: list[QuantumRegister]) -> QuantumCircuit:
        """
        Reproduces the pytket counter using cascaded multi-controlled X gates.

        Qubit order in the returned circuit: [S..., U_0..., U_1..., ..., U_{u-1}...]
        """
        count = QuantumCircuit(S_reg, *U_regs, name="COUNT")

        # Build lists of (controls..., target) exactly like the pytket logic
        all_lists = []
        j = 0
        for subset_key in self.subsets:
            q_list_for_subset = []
            for elementU in self.subsets[subset_key]:
                i = self.universe.index(elementU)
                S_list = [S_reg[j]]
                # iterate over U_i register bits
                for q in U_regs[i]:
                    S_list.append(q)
                    q_list_for_subset.append(deepcopy(S_list))
            # reverse the per-subset list as in the original code
            q_list_for_subset = list(reversed(q_list_for_subset))
            all_lists.append(q_list_for_subset)
            j += 1

        # Add MCX gates: controls are all but last, target is last
        for per_subset in all_lists:
            for q_list in per_subset:
                if len(q_list) == 1:
                    # degenerate case: just X on the target (no controls)
                    count.x(q_list[0])
                else:
                    count.mcx(q_list[:-1], q_list[-1], mode="noancilla")

        return count

    def _build_oracle_circuit(self, U_regs: list[QuantumRegister], anc_reg: QuantumRegister) -> QuantumCircuit:
        """
        Oracle: X on all U bits except index 0 in each U_i,
        then one big MCX onto ancilla, then undo the X.
        Qubit order: [U_0..., U_1..., ..., U_{u-1}..., anc]
        """
        oracle = QuantumCircuit(*U_regs, anc_reg, name="ORACLE")

        # Flip all U bits with index != 0
        for reg in U_regs:
            for bit_idx, q in enumerate(reg):
                if bit_idx != 0:
                    oracle.x(q)

        # Big MCX: controls are all U bits, target is anc
        controls = self._flatten_registers(U_regs)
        target = anc_reg[0]
        if controls:
            oracle.mcx(controls, target, mode="noancilla")
        else:
            # No controls? Just X on anc (edge case)
            oracle.x(target)

        # Uncompute X flips
        for reg in U_regs:
            for bit_idx, q in enumerate(reg):
                if bit_idx != 0:
                    oracle.x(q)

        return oracle

    def _build_diffuser_circuit(self, S_reg: QuantumRegister) -> QuantumCircuit:
        """
        Standard Grover diffuser on S.
        Qubit order: [S...]
        """
        diff = QuantumCircuit(S_reg, name="DIFFUSER")

        # H, X on all S
        for q in S_reg:
            diff.h(q)
            diff.x(q)

        # Multi-controlled Z (on the all-ones state after X) via H-mapped MCX
        if len(S_reg) == 1:
            # For a single qubit: H X Z X H  ==  H X (Z) X H == reflection
            diff.h(S_reg[0])
            diff.z(S_reg[0])
            diff.h(S_reg[0])
        else:
            diff.h(S_reg[-1])
            diff.mcx(S_reg[:-1], S_reg[-1], mode="noancilla")  # implements C^(n-1)Z
            diff.h(S_reg[-1])

        # Undo X, then H
        for q in S_reg:
            diff.x(q)
            diff.h(q)

        return diff

    # ---------- Public methods (same signatures) ----------

    def resource_estimation(self):
        s_size = self.s_size
        num_solutions = self.num_solutions

        mpmath.mp.dps = 50
        ln_2 = mpmath.log(2)
        ln_pi_over_4 = mpmath.log(mpmath.pi / 4)
        ln_num_solutions = mpmath.log(num_solutions)

        ln_a = (s_size * ln_2 - ln_num_solutions) / 2
        ln_num_iterations = ln_pi_over_4 + ln_a
        num_iterations = int(mpmath.floor(mpmath.exp(ln_num_iterations)))

        num_qubits = self.s_size + self.u_size * self.b + 1

        superpos_gates = self.s_size
        prepare_anc_gates = 2
        counter_gates = 0
        for s in self.subsets:
            counter_gates += len(self.subsets[s]) * self.b
        oracle_gates = 1 + 2 * ((self.u_size - 1) * self.b)
        diffuser_gates = 1 + 4 * self.s_size
        MCX_gates = num_iterations * (oracle_gates + 2 * counter_gates)
        total_gates = (
            superpos_gates
            + prepare_anc_gates
            + MCX_gates
            + num_iterations * diffuser_gates
        )
        return {
            "n_qubits": num_qubits,
            "MCX_gates": MCX_gates,
            "n_gates": total_gates,
            "depth": None,
        }

    # ---------- Circuit construction ----------

    def _build_circuit(self):
        """
        Build the full circuit (main) with Grover iterations and measurements.
        Returns: QuantumCircuit
        """
        # Registers
        S = QuantumRegister(self.s_size, "S")  # subsets
        U_regs = [QuantumRegister(self.b, f"U_{i}") for i in range(self.u_size)]
        anc = QuantumRegister(1, "anc")
        c = ClassicalRegister(self.s_size, "c")

        # Main circuit with all regs
        self.main_circuit = QuantumCircuit(S, *U_regs, anc, c, name="MAIN")

        # Superposition on S
        for q in S:
            self.main_circuit.h(q)

        # Prepare ancilla |->  (X then H)
        self.main_circuit.x(anc[0])
        self.main_circuit.h(anc[0])

        # Build subcircuits (as gates) with qubit orders we will map onto MAIN
        count = self._build_counter_circuit(S, U_regs)
        self.count_circuit = count
        self.count_circuit_dag = count.inverse()

        oracle = self._build_oracle_circuit(U_regs, anc)
        self.oracle = oracle

        diffuser = self._build_diffuser_circuit(S)
        self.diffuser = diffuser

        # Turn them into gates for easy repetition
        count_gate = count.to_gate(label="COUNT")
        count_dg_gate = count_gate.inverse()
        oracle_gate = oracle.to_gate(label="ORACLE")
        diffuser_gate = diffuser.to_gate(label="DIFFUSER")

        # Default #iterations if not provided
        num_iterations = math.floor(
            (math.pi / 4) * math.sqrt((2 ** self.s_size) / self.num_solutions)
        )

        # Prepare fixed qubit orders for appends
        S_list = list(S)
        U_flat = self._flatten_registers(U_regs)
        AUX_order_for_count = S_list + U_flat
        AUX_order_for_oracle = U_flat + [anc[0]]

        # One Grover iteration = COUNT → ORACLE → COUNT† → DIFFUSER
        for _ in range(num_iterations):
            self.main_circuit.append(count_gate, AUX_order_for_count)
            self.main_circuit.append(oracle_gate, AUX_order_for_oracle)
            self.main_circuit.append(count_dg_gate, AUX_order_for_count)
            self.main_circuit.append(diffuser_gate, S_list)

        # Measure S into classical bits
        self.main_circuit.measure(S, c)

        return self.main_circuit

    # Parity with your original method names
    def _assemble_aux_circ(self):
        """
        (Kept for API parity) Builds and stores an AUX circuit = COUNT + ORACLE + COUNT†
        as a circuit acting on [S, all U, anc]. Not needed for main build since we append gates directly.
        """
        S_aux = QuantumRegister(self.s_size, "S")
        U_aux = [QuantumRegister(self.b, f"U_{i}") for i in range(self.u_size)]
        anc_aux = QuantumRegister(1, "anc")
        aux = QuantumCircuit(S_aux, *U_aux, anc_aux, name="AUX")

        count = self._build_counter_circuit(S_aux, U_aux).to_gate(label="COUNT")
        oracle = self._build_oracle_circuit(U_aux, anc_aux).to_gate(label="ORACLE")
        count_dg = count.inverse()

        aux.append(count, list(S_aux) + self._flatten_registers(U_aux))
        aux.append(oracle, self._flatten_registers(U_aux) + [anc_aux[0]])
        aux.append(count_dg, list(S_aux) + self._flatten_registers(U_aux))

        self.aux_circ = aux
        return aux

    def _assemble_full_circuit_w_meas(self, num_iterations=None):
        """
        API-compatible wrapper that just calls _build_circuit(); the Qiskit
        version integrates measurements already.
        """
        return self._build_circuit()
