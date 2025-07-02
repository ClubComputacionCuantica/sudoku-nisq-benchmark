from pytket import Circuit, Qubit, OpType
from itertools import combinations
import math

from sudoku_nisq.quantum_solver import QuantumSolver
from sudoku_nisq.board import Board

class GraphColoringQuantumSolver(QuantumSolver):
    """
    Grover-based quantum solver for the graph coloring problem.

    This solver recasts a graph coloring task as a constraint satisfaction problem similar to Sudoku,
    using vertex-pair constraints that enforce different colors. It then constructs a quantum circuit
    using Grover's algorithm to search for valid color assignments.
    """
    def __init__(self, sudoku=None, num_solutions: int = None, tuples: list = None, field_values: dict = None, 
                 subunit_height: int = None, subunit_width: int = None):
        """
        Initialize the graph-coloring quantum solver with either a Sudoku-based structure or manual parameters.

        Parameters
        ----------
        sudoku : Sudoku, optional
            A Sudoku instance used to derive graph structure and preset values.
        num_solutions : int, optional
            Number of solutions expected. Influences the number of Grover iterations.
        tuples : list of tuple of int, optional
            List of (i, j) pairs indicating graph edges—i.e., vertices that must have different colors.
        field_values : dict of int to int, optional
            Mapping from vertex index to its fixed color value (if any).
        subunit_height : int, optional
            Used to determine the color space size when not using a Sudoku instance.
        subunit_width : int, optional
            Used alongside subunit_height to compute color space size.
        """
        super().__init__()

        # Construct from a Sudoku object from Board, if provided
        if sudoku is not None:
            n = sudoku.subgrid_size
            sudoku_board = Board(
                unit_height=n,
                unit_width=n,
                grid_height=n,
                grid_width=n,
                init_value=-1
            )
            # Process initial values from the Sudoku format
            clean = [(i, j, -1 if v == 0 else v) for i, j, v in sudoku.pre_tuples]
            positions = [(i, j) for i, j, v in clean]
            values = [v for _, _, v in clean]
            sudoku_board.update_board(values, positions)
            self.tuples = sudoku_board.get_open_indexed_tuples()

            # Flatten 2D positions to 1D keys for field_values
            field_values = {
                i * sudoku.board_size + j: v
                for i, j, v in sudoku.pre_tuples
                if v != 0
            }
            
            self.subunit_height = n
            self.subunit_width = n
        else:
            # Use explicitly provided tuples and dimensions
            self.tuples = tuples
            self.subunit_height = subunit_height
            self.subunit_width = subunit_width

        # Set number of solutions to guide Grover iterations
        self.num_solutions = (
            sudoku.count_solutions() if num_solutions is None and sudoku is not None else num_solutions or 1
        )

        # Normalize vertex indices to compact representation for encoding
        self.normalized_tuples = self._get_normalized_tuples()
        self.normalized_field_values = self._get_normalized_field_values(field_values)
    
    def find_resources(self) -> Dict[str, Any]:
        return {
            "n_qubits": None,
            "MCX_gates": None,
            "n_gates": None,
            "depth": None,
            "error": "Not implemented"
        }
    
    def get_circuit(self) -> Circuit:
        """
        Build and return a Grover-based quantum circuit that searches for a valid graph coloring.

        This method performs the following steps:
        1. **Compute register sizes**  
            - `unique`: number of distinct vertices  
            - `color_size`: bits needed to encode each vertexs color  
            - `in_bits`: total input qubits = unique x color_size  
            - `cmp_bits`: one ancilla per edge/constraint  
            - `unknown_count`: number of qubits in superposition  
            - `iterations`: Grover iteration count
        2. **Allocate qubits and classical bits**  
            - Input qubits (`in_qubits`)  
            - Comparison ancillas (`cmp_qubits`)  
            - One output ancilla (`out_q`) initialized to |−⟩ for phase kickback  
            - Classical register (`c_bits`) to store measurement results  
        3. **Initialize qubits**  
            - Fixed-color qubits set via X gates according to `normalized_field_values`  
            - Unknown-color qubits put into H superposition and tracked in `unknown_qubit_list`  
        4. **Grover loop** (repeat `iterations` times)  
            a. Apply the graph-coloring oracle (`_graph_coloring_oracle`) marking invalid assignments  
            b. Apply the diffusion operator over the unknown qubits (`_diffuser`)  
        5. **Measurement**  
            - Measure each input qubit into the corresponding classical bit  

        Returns
        -------
        Circuit
            A pytket Circuit implementing the full Grover search for valid colorings.
        """
        # Determine dimensions
        unique = len({f for tup in self.tuples for f in tup})
        color_size = self._get_color_size()
        in_bits = unique * color_size
        cmp_bits = len(self.tuples)
        class_bits = in_bits
        unknown_count = (unique - len(self.normalized_field_values)) * color_size
        print(unknown_count, unique, len(self.normalized_field_values), color_size)
        print(f"Graph coloring problem with {unique} vertices, {color_size} color bits each(total {in_bits} input bits), "
            f"{cmp_bits} constraints, {unknown_count} unknown qubits, {self.num_solutions} solutions.")
        if unknown_count <= 0:
            raise ValueError("No unknown qubits to search; all colors are fixed.")
        
        iterations = self._get_grover_iterations(unknown_count)

        # Build circuit & registers
        circ = Circuit()
        in_qubits = [Qubit("in", i) for i in range(in_bits)]
        for q in in_qubits:
            circ.add_qubit(q)
        cmp_qubits = [Qubit("cmp", i) for i in range(cmp_bits)]
        for q in cmp_qubits:
            circ.add_qubit(q)
        out_q = Qubit("out", 0)
        circ.add_qubit(out_q)
        c_bits = circ.add_c_register("c", class_bits)

        # Initialize out to |1> -> H
        circ.X(out_q)
        circ.H(out_q)

        # Initialize inputs and collect unknown qubits
        inits = self._get_qubit_inits(in_bits, color_size)
        unknown_qubit_list = []
        for idx, val in inits.items():
            q = in_qubits[idx]
            if val is None:
                circ.H(q)
                unknown_qubit_list.append(q)
            elif val == 1:
                circ.X(q)
            # else val == 0: leave in |0>

        # Grover iterations
        for _ in range(iterations):
            self._graph_coloring_oracle(circ, in_qubits, cmp_qubits, out_q, color_size)
            self._diffuser(circ, unknown_qubit_list, out_q)

        # Measurement
        for i, q in enumerate(in_qubits):
            circ.Measure(q, c_bits[i])
        self.main_circuit = circ.copy()
        return circ
    
    def _get_grover_iterations(self, unknown_count: int) -> int:
        """
        Returns the number of Grover iterations
        j = floor((π/4) * sqrt(N / M)),
        where N = 2**unknown_count and M = self.num_solutions.
        """
        N = 2**unknown_count
        M = self.num_solutions
        return math.floor((math.pi / 4) * math.sqrt(N / M))
    
    def _flipper(self, circ: Circuit, in_qubits: list, cmp_qubits: list, idx: int, color_size: int):
        """
        Adds gate patterns to mark heuristic color‐constraint violations for one edge.

        Targets the constraint that two connected vertices, x and y
        (given by self.normalized_tuples[idx]), must not share the same color.
        Each vertexs color is stored as a binary string across `color_size` qubits
        in `in_qubits`. The ancilla qubit cmp_qubits[idx] is flipped for certain
        1‐bit patterns across those two blocks—but it is not a strict equality check.

        Operation:
        ----------
        1. Pairwise CNOTs:
        - For each bit position i in [0, color_size):
            * CNOT(control=in_qubits[x_block + i], target=ancilla)
            * CNOT(control=in_qubits[y_block + i], target=ancilla)
        - Cumulatively flips the ancilla once for every 1‐bit in either color register,
            entangling parity information of all bits.

        2. Multi‐Controlled Xs:
        - Let dist = |y − x| * color_size.
        - For r = 2 … color_size:
            * Generate all size‐r combinations of qubit indices from both x and y blocks
                that do *not* include any pair separated by exactly `dist` (i.e. aligned bits).
            * For each such combo, apply a multi‐controlled X (CnX) using those r qubits
                as controls onto the ancilla.
        - These further flip the ancilla when specific higher‐order bit patterns
            (combinations of r ones) occur across the two registers.

        Net effect:
        -----------
        The ancilla cmp_qubits[idx] ends up in |1⟩ for a variety of overlapping
        bit‐patterns across x and ys color encodings. It therefore serves as a
        heuristic “violation marker” during the Grover oracle, but is *not* a
        direct bitwise equality comparator.

        Parameters
        ----------
        circ : Circuit
            The pytket Circuit being constructed.
        in_qubits : list[Qubit]
            All input qubits, grouped in blocks of `color_size` per vertex.
        cmp_qubits : list[Qubit]
            Ancilla qubits used to flag each edge; this method uses cmp_qubits[idx].
        idx : int
            Which pair of vertices (edge) to check, from self.normalized_tuples.
        color_size : int
            Number of qubits encoding each vertexs color (i.e., ⌈log₂(num_colors)⌉).
        """
        # Determine x, y and their positions
        x, y = self.normalized_tuples[idx]
        start_x, start_y = x * color_size, y * color_size
        # integer positions for combination logic
        x_pos = list(range(start_x, start_x + color_size))
        y_pos = list(range(start_y, start_y + color_size))
        all_pos = x_pos + y_pos
        # corresponding Qubit handles
        x_qubits = [in_qubits[p] for p in x_pos]
        y_qubits = [in_qubits[p] for p in y_pos]
        anc = cmp_qubits[idx]

        # pairwise CNOTs onto ancilla
        for qx, qy in zip(x_qubits, y_qubits):
            circ.add_gate(OpType.CnX, [qx, anc])
            circ.add_gate(OpType.CnX, [qy, anc])

        # multi-controlled X for valid combos of positions
        dist = abs(y - x) * color_size
        for r in range(2, color_size + 1):
            for combo in self._generate_valid_combinations(all_pos, r, dist):
                controls = [in_qubits[p] for p in combo]
                circ.add_gate(OpType.CnX, controls + [anc])

    def _graph_coloring_oracle(self, circ: Circuit, in_qubits: list, cmp_qubits: list, out_qubit: Qubit, color_size: int):
        """
        Constructs the Grover oracle for the graph coloring constraints.

        This oracle marks all assignments that violate any edge-coloring constraint
        between adjacent vertices. It uses the following steps:

        1. **Compute Constraint Flags**  
        For each edge (tuple of vertices) in `self.normalized_tuples`, call
        `_flipper`, which flips the corresponding ancilla in `cmp_qubits[idx]`
        if those two vertices may share the same color.

        2. **Mark Invalid Solutions**  
        Apply a multi-controlled NOT (`CnX`) from _all_ comparison ancillas
        onto the single `out_qubit`. If _any_ ancilla is |1⟩ (constraint violated),
        the `out_qubit` flips, marking the state as “bad” for Grover.

        3. **Uncompute Constraint Flags**  
        Re-run `_flipper` in reverse to reset all comparison ancillas back to |0⟩,
        so they dont carry garbage into subsequent operations.

        Parameters
        ----------
        circ : Circuit
            The pytket circuit under construction.
        in_qubits : list of Qubit
            Qubits encoding the color bits for each vertex.
        cmp_qubits : list of Qubit
            Ancilla qubits, one per edge, used to flag constraint violations.
        out_qubit : Qubit
            Single-qubit flag that is flipped if _any_ edge has a conflict.
        color_size : int
            Number of bits used to encode each vertexs color.

        Notes
        -----
        - This method implements the oracle of Grovers algorithm: it flips the phase
        (via the `out_qubit`) of all states that do _not_ satisfy the coloring constraints.
        - The `cmp_qubits` register must start all in |0⟩ and end all in |0⟩.
        """
        # apply flipper for each tuple
        for idx in range(len(self.normalized_tuples)):
            self._flipper(circ, in_qubits, cmp_qubits, idx, color_size)
        # mark solution on out_qubit
        circ.add_gate(OpType.CnX, cmp_qubits + [out_qubit])
        # uncompute
        for idx in range(len(self.normalized_tuples)):
            self._flipper(circ, in_qubits, cmp_qubits, idx, color_size)

    def _diffuser(self, circ: Circuit, target_qubits: list[Qubit], out_qubit: Qubit):
        """
        Perform the Grover diffuser (inversion-about-the-mean) over a subset of qubits.

        This routine implements the standard Grover diffusion operator,
        but only on those input qubits that were initialized into superposition
        (i.e. the “unknown” bits). It uses the output qubit as the phase-flip ancilla.

        Steps:
        1. Apply H then X to each target qubit to prepare for phase inversion.
        2. Use a multi-controlled X (CnX) gate with all targets as controls
        and the out_qubit as target—this flips the global phase of the
        |00…0⟩ state in the transformed basis.
        3. Uncompute by applying X then H on each target qubit to return to
        the computational basis.

        Parameters
        ----------
        circ : Circuit
            The pytket Circuit being built.
        target_qubits : list[Qubit]
            List of qubits (in superposition) on which to perform the diffusion.
        out_qubit : Qubit
            Ancilla qubit used to mark the all-zero state during the phase flip.
        """
        # Apply H and X to each target qubit
        for q in target_qubits:
            circ.H(q)
            circ.X(q)
        # Multi-controlled X with all targets controlling the out_qubit
        circ.add_gate(OpType.CnX, target_qubits + [out_qubit])
        # Uncompute X and H
        for q in target_qubits:
            circ.X(q)
            circ.H(q)
    
    def _generate_valid_combinations(self, elements, size, forbidden_diff):
        """
        Generate all valid combinations of a given size from the input elements,
        where no two elements in any combination differ by exactly `forbidden_diff`.

        This utility is used to avoid certain binary patterns (e.g., bitstrings)
        that represent conflicting variable assignments in the quantum oracle.

        Parameters
        ----------
        elements : list of int
            The list of integer positions (e.g., qubit indices) to select combinations from.
        size : int
            The number of elements in each combination.
        forbidden_diff : int
            A specific absolute difference that should not appear between any two elements in a combination.

        Returns
        -------
        list of tuple of int
            All valid combinations (of the given size) where no pair of integers differs by forbidden_diff.
        """
        def ok(combo):
            # Check all pairs in the combination
            for a, b in combinations(combo, 2):
                # If any pair has an absolute difference equal to the forbidden value, reject it
                if abs(a - b) == forbidden_diff:
                    return False
            return True

        # Return only combinations that pass the filter
        return [c for c in combinations(elements, size) if ok(c)]

    def _get_normalized_tuples(self) -> list:
        field_set = sorted({f for tup in self.tuples for f in tup})
        mapping = {orig: i for i, orig in enumerate(field_set)}
        return [(mapping[a], mapping[b]) for a, b in self.tuples]

    def _get_normalized_field_values(self, field_values: dict) -> dict:
        field_set = sorted({f for tup in self.tuples for f in tup})
        mapping = {orig: i for i, orig in enumerate(field_set)}
        return {mapping[k]: v for k, v in field_values.items() if k in mapping}

    def _get_color_size(self) -> int:
        sub_size = self.subunit_height * self.subunit_width - 1
        return len(bin(sub_size)) - 2

    def _padded_binary(self, dec: int, size: int) -> str:
        b = bin(dec)[2:]
        return b.rjust(size, '0')

    def _get_qubit_inits(self, total: int, size: int) -> dict:
        mapping = {}
        for i in range(total // size):
            val = self.normalized_field_values.get(i)
            for j in range(size):
                idx = i * size + j
                if val is None:
                    mapping[idx] = None  # superposition
                else:
                    bit = self._padded_binary(val, size)[j]
                    mapping[idx] = int(bit)
        return mapping