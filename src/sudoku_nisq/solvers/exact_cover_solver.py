import math
import mpmath
from typing import Literal, Optional, List, Dict, Any

from sudoku_nisq.quantum_solver import QuantumSolver
from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding
from sudoku_nisq.utils.memory_tracker import MemoryTracker
from sudoku_nisq.exact_cover_problem import ExactCoverProblem

class ExactCoverQuantumSolver(QuantumSolver):
    """
    Quantum solver that constructs circuits to solve Sudoku puzzles using exact cover
    formulation with Grover's algorithm.
    
    Based on: J. -R. Jiang and Y. -J. Wang, "Quantum Circuit Based on Grover's Algorithm 
    to Solve Exact Cover Problem," 2023 VTS Asia Pacific Wireless Communications 
    Symposium (APWCS), Tainan city, Taiwan, 2023.
    """
    
    def __init__(self, puzzle=None, metadata_manager=None, encoding: Literal["simple", "pattern"] = "simple",
                 num_solutions=None, universe=None, subsets=None, 
                 exact_cover_problem: Optional[ExactCoverProblem] = None, **kwargs):
        """
        Initialize the ExactCoverQuantumSolver.
        
        Supports three modes:
        1. Sudoku mode: Provide puzzle, encoding is applied
        2. Generic mode: Provide exact_cover_problem (ExactCoverProblem instance)
        3. Direct mode: Provide universe and subsets directly
        
        Args:
            puzzle: SudokuPuzzle instance (for Sudoku mode)
            metadata_manager: MetadataManager for caching
            encoding: Encoding type for Sudoku ("simple" or "pattern")
            num_solutions: Expected number of solutions
            universe: Direct universe list (for generic/direct mode)
            subsets: Direct subsets dict (for generic/direct mode)
            exact_cover_problem: ExactCoverProblem instance (for generic mode)
            **kwargs: Additional options (decompose_cnz, track_memory)
        """
        
        # Extract gate counting options before passing to parent
        self.decompose_cnz = kwargs.pop('decompose_cnz', True)  # Default: decompose for consistency
        
        # Extract memory tracking option (advanced/dev feature)
        self.track_memory = kwargs.pop('track_memory', False)  # Default: disabled for production use
        
        # Determine mode: Generic vs Sudoku
        if exact_cover_problem is not None:
            # Generic mode: Use ExactCoverProblem
            self._is_generic = True
            self.universe = exact_cover_problem.universe
            self.subsets = exact_cover_problem.subsets
            self.num_solutions = exact_cover_problem.num_solutions or 1
            self._problem_hash = exact_cover_problem.get_hash()  # Store hash for caching
            
            # For generic mode, puzzle parameter is optional
            # We still initialize parent but with puzzle=None
            super().__init__(
                puzzle=None,
                metadata_manager=metadata_manager,
                encoding=encoding,
                **kwargs
            )
            
        elif universe is not None and subsets is not None:
            # Direct mode: Use provided universe/subsets
            self._is_generic = True
            self.universe = universe
            self.subsets = subsets
            self.num_solutions = num_solutions or 1
            
            # Generate hash for caching
            from sudoku_nisq.exact_cover_problem import ExactCoverProblem
            temp_problem = ExactCoverProblem(universe=universe, subsets=subsets)
            self._problem_hash = temp_problem.get_hash()
            
            super().__init__(
                puzzle=None,
                metadata_manager=metadata_manager,
                encoding=encoding,
                **kwargs
            )
            
        elif puzzle is not None:
            # Sudoku mode: Use ExactCoverEncoding
            self._is_generic = False
            
            # Initialize the base class with puzzle
            super().__init__(
                puzzle=puzzle,
                metadata_manager=metadata_manager,
                encoding=encoding,
                **kwargs
            )
            
            # Initialize encoding
            enc = ExactCoverEncoding(puzzle)
            
            # Select correct universe depending on puzzle size
            if hasattr(enc, 'universe'):
                self.universe = enc.universe
            else:
                self.universe = enc.universe2x2
            
            # Determine which encoding to use
            if encoding == "simple":
                self.subsets = enc.simple_subsets
            elif encoding == "pattern":
                self.subsets = enc.pattern_subsets
            else:
                raise ValueError(f"Unknown encoding {encoding!r}")
            
            # Set number of solutions
            self.num_solutions = puzzle.num_solutions if hasattr(puzzle, 'num_solutions') else 1
            
        else:
            raise ValueError(
                "Must provide one of: "
                "(1) puzzle for Sudoku mode, "
                "(2) exact_cover_problem for generic mode, or "
                "(3) both universe and subsets for direct mode"
            )
        
        # Common initialization for all modes
        self.u_size = len(self.universe)        # Total elements to cover
        self.s_size = len(self.subsets)         # Number of subsets
        self.b = math.ceil(math.log2(self.s_size)) if self.s_size > 1 else 1  # Bits for counting
        
        # Gate counting (populated when circuit is built)
        self.gate_counts = None
        
        # Memory tracking (populated when circuit is built)
        self.memory_usage = None

    def _build_sdk_circuit(self, sdk_type: str):
        """Build exact cover circuit using the specified SDK.
        
        Returns:
            Circuit object in the specified SDK format. Gate counts are stored
            in self.gate_counts and memory usage in self.memory_usage (if enabled) as side effects.
        """
        # Track memory during circuit construction (optional, for advanced/dev use)
        if self.track_memory:
            mem_tracker = MemoryTracker()
            mem_tracker.start()
            mem_tracker.snapshot('before_circuit')
        
        if sdk_type == "pytket":
            from sudoku_nisq.circuits.exact_cover.pytket_impl import build_exact_cover_circuit
            circuit, gate_counts = build_exact_cover_circuit(self, decompose_cnz=self.decompose_cnz)
            if self.track_memory:
                mem_tracker.snapshot('after_pytket_build')
            self.gate_counts = gate_counts
            if self.track_memory:
                self.memory_usage = mem_tracker.report()
            return circuit
        elif sdk_type == "qiskit":
            from sudoku_nisq.circuits.exact_cover.qiskit_impl import build_exact_cover_circuit
            circuit, gate_counts = build_exact_cover_circuit(self)
            if self.track_memory:
                mem_tracker.snapshot('after_qiskit_build')
            self.gate_counts = gate_counts
            if self.track_memory:
                self.memory_usage = mem_tracker.report()
            return circuit
        elif sdk_type == "braket":
            # For now, fallback to pytket
            from sudoku_nisq.circuits.exact_cover.pytket_impl import build_exact_cover_circuit
            circuit, gate_counts = build_exact_cover_circuit(self, decompose_cnz=self.decompose_cnz)
            if self.track_memory:
                mem_tracker.snapshot('after_braket_build')
            self.gate_counts = gate_counts
            if self.track_memory:
                self.memory_usage = mem_tracker.report()
            return circuit
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
    
    def _transpile_pytket(self, backend, opt_level: int):
        """Transpile circuit using PyTKET backend.
        
        Uses PyTKET's native backend compilation interface which applies
        hardware-specific gate decompositions and optimizations.
        
        Args:
            backend: PyTKET backend instance with get_compiled_circuit method
            opt_level: Optimization level for transpilation (0-2)
            
        Returns:
            Circuit: Transpiled circuit in PyTKET format
        """
        try:
            tcirc = backend.get_compiled_circuit(
                self.main_circuit,
                optimisation_level=opt_level
            )
            return tcirc
        except Exception as e:
            raise RuntimeError(
                f"PyTKET transpilation failed at opt_level {opt_level}: {e}"
            )
    
    def _transpile_qiskit(self, backend, opt_level: int):

            # -------------------------
            # Decoding & result helpers
            # -------------------------
            def decode_bitstring(self, bitstring: str) -> Dict[str, Any]:
                """Decode a measurement bitstring into selected subsets and assignments.

                For Sudoku mode, attempts to extract (row, col, digit) assignments from
                subset contents when available. For generic exact cover mode, returns the
                selected subset keys and their raw contents.

                Returns a dict with:
                  - 'selected_indices': list[int]
                  - 'selected_subsets': dict[str, list]
                  - 'assignments': list[tuple]  # best-effort (row,col,digit) if available
                """
                selected_indices = [i for i, b in enumerate(bitstring) if b == '1']
                selected_subsets = {f'S_{i}': self.subsets.get(f'S_{i}', []) for i in selected_indices}

                assignments = []
                # Best-effort extraction of (row, col, digit) tuples if present
                for items in selected_subsets.values():
                    for entry in items:
                        if isinstance(entry, tuple) and len(entry) == 3 and all(isinstance(x, int) for x in entry):
                            assignments.append(entry)

                return {
                    'selected_indices': selected_indices,
                    'selected_subsets': selected_subsets,
                    'assignments': assignments,
                }

            def apply_assignments(self, assignments: List[tuple]) -> Optional[list]:
                """Apply (row, col, digit) assignments to a copy of the puzzle board.

                Returns a new board (list[list[int]]) if in Sudoku mode and a puzzle is
                available; otherwise returns None.
                """
                if hasattr(self, 'puzzle') and self.puzzle is not None and hasattr(self.puzzle, 'board'):
                    # Deep-copy board
                    import copy
                    board = copy.deepcopy(self.puzzle.board)
                    for (r, c, d) in assignments:
                        if 0 <= r < len(board) and 0 <= c < len(board[r]):
                            board[r][c] = d
                    return board
                return None

            def decode_counts(self, counts: Dict[str, int], *, k_top: int = 3) -> Dict[str, Any]:
                """Convert counts to structured solution metrics and ranking.

                Returns dict with:
                  - 'solutions': [{ 'bitstring', 'prob', 'selected_indices', 'selected_subsets', 'assignments', 'board', 'is_valid' }]
                  - 'success_rate': total probability mass on valid exact covers
                  - 'topk_solution_mass': sum of probabilities of top-k valid covers (default k=3)
                  - 'precision_at_k': fraction of top-k bitstrings overall that are valid covers
                  - 'recall_at_k': fraction of all valid covers that appear among top-k overall
                  - 'distinct_solutions_observed': number of valid covers observed (count>0)
                  - 'snr': ratio of best valid prob to best invalid prob
                """
                total = sum(counts.values()) or 1

                # Probabilities per bitstring and overall ranking
                probs = { (bitstring if isinstance(bitstring, str) else ''.join(str(b) for b in bitstring)) : c / total
                          for bitstring, c in counts.items() }
                all_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

                # Build solutions list with validity flags and decoded info
                solutions = []
                for bs, p in all_sorted:
                    dec = self.decode_bitstring(bs)
                    board = self.apply_assignments(dec['assignments']) if dec['assignments'] else None
                    is_valid = False
                    try:
                        is_valid = self._is_valid_solution(bs)
                    except Exception:
                        is_valid = False
                    solutions.append({
                        'bitstring': bs,
                        'prob': p,
                        **dec,
                        'board': board,
                        'is_valid': is_valid,
                    })

                # Success probability = total mass on valid exact covers
                success_rate = sum(p for bs, p in all_sorted if next((s for s in solutions if s['bitstring'] == bs), {}).get('is_valid', False))

                # Top-k solution mass (k most probable valid covers)
                valid_sorted = [(s['bitstring'], s['prob']) for s in solutions if s['is_valid']]
                valid_sorted.sort(key=lambda kv: kv[1], reverse=True)
                topk_solution_mass = sum(p for _, p in valid_sorted[:k_top])

                # Precision@k and Recall@k over overall top-k bitstrings
                topk_all = solutions[:k_top]
                n_sol_in_topk = sum(1 for s in topk_all if s['is_valid'])
                precision_at_k = n_sol_in_topk / (k_top or 1)
                # Estimate total number of valid covers via problem enumeration if available
                try:
                    total_valid = len(self.problem.enumerate_valid_bitstrings())
                except Exception:
                    total_valid = len(valid_sorted)
                recall_at_k = (n_sol_in_topk / (total_valid or 1)) if total_valid else 0.0

                # Distinct valid solutions observed
                # Use counts-based observation (count > 0)
                observed_valid = set()
                for s in solutions:
                    if s['is_valid'] and counts.get(s['bitstring'], 0) > 0:
                        observed_valid.add(s['bitstring'])
                distinct_solutions_observed = len(observed_valid)

                # SNR between best valid and best invalid
                best_valid = max((s['prob'] for s in solutions if s['is_valid']), default=0.0)
                best_invalid = max((s['prob'] for s in solutions if not s['is_valid']), default=0.0)
                snr = (best_valid / best_invalid) if best_invalid > 0 else (float('inf') if best_valid > 0 else 0.0)

                return {
                    'solutions': solutions,
                    'success_rate': success_rate,
                    'topk_solution_mass': topk_solution_mass,
                    'precision_at_k': precision_at_k,
                    'recall_at_k': recall_at_k,
                    'distinct_solutions_observed': distinct_solutions_observed,
                    'snr': snr,
                }

    def format_result(self, result: Any) -> Dict[str, Any]:
        """Format a backend-native Result into a decoded solution structure.

        Extracts counts via result.get_counts(), decodes solutions, and attaches
        mitigated success rate if present on the Result object.
        """
        counts = result.get_counts()
        out = self.decode_counts(counts)
        if hasattr(result, '_mitigated_success_prob'):
            out['mitigated_success_rate'] = result._mitigated_success_prob
        # Attach eta metrics using compiled circuit resources if available
        try:
            from sudoku_nisq.metrics import compute_eta_metrics
            # Prefer transpiled circuit if present on result or solver
            depth = None
            gate_counts = None
            sdk = 'qiskit'
            # If result carries the compiled circuit
            compiled = getattr(result, 'compiled_circuit', None)
            if compiled is None:
                compiled = getattr(self, 'transpiled_circuit', None)
            if compiled is not None and hasattr(compiled, 'depth'):
                try:
                    depth = compiled.depth()
                except Exception:
                    pass
                try:
                    gate_counts = dict(compiled.count_ops())
                except Exception:
                    pass
            # Fallback to any stored gate_counts
            if gate_counts is None:
                gate_counts = getattr(self, 'gate_counts', None)
            eta_metrics = compute_eta_metrics(out.get('success_rate'), gate_counts=gate_counts, depth=depth, sdk=sdk)
            out.update(eta_metrics)
        except Exception:
            pass
        # Compute 95% CI for success_rate based on total shots
        try:
            from sudoku_nisq.metrics import binomial_ci_normal
            total_shots = sum(counts.values()) if isinstance(counts, dict) else None
            p_hat = out.get('success_rate')
            if total_shots:
                ci = binomial_ci_normal(p_hat, total_shots, level=0.95)
                if ci:
                    out['ci_95_low'], out['ci_95_high'] = ci
                    out['shots'] = total_shots
        except Exception:
            pass
        # Compute shots for 99% success if success_rate available
        try:
            from sudoku_nisq.metrics import shots_for_target_success
            s99 = shots_for_target_success(out.get('success_rate'), target=0.99)
            out['shots_for_99pct'] = s99
        except Exception:
            pass
        # Persist execution metrics to metadata if manager is available
        try:
            if hasattr(self, 'metadata_manager') and self.metadata_manager is not None:
                metrics_to_store = {
                    'success_rate': out.get('success_rate'),
                    'topk_solution_mass': out.get('topk_solution_mass'),
                    'precision_at_k': out.get('precision_at_k'),
                    'recall_at_k': out.get('recall_at_k'),
                    'distinct_solutions_observed': out.get('distinct_solutions_observed'),
                    'snr': out.get('snr'),
                    'eta': out.get('eta'),
                    'eta2': out.get('eta2'),
                    'G_total': out.get('G_total'),
                    'G_2q': out.get('G_2q'),
                    'depth': out.get('depth'),
                    'shots_for_99pct': out.get('shots_for_99pct'),
                    'ci_95_low': out.get('ci_95_low'),
                    'ci_95_high': out.get('ci_95_high'),
                    'shots': out.get('shots'),
                }
                # Attempt to derive backend alias and opt level from recent run context
                backend_alias = getattr(self, 'last_backend_alias', 'unknown')
                opt_level = getattr(self, 'last_opt_level', 0)
                self.metadata_manager.record_execution_metrics(
                    solver_name=self.__class__.__name__,
                    encoding=getattr(self, 'encoding', 'default'),
                    backend_alias=backend_alias,
                    opt_level=opt_level,
                    metrics=metrics_to_store,
                )
        except Exception:
            pass
        return out
        """Transpile circuit using Qiskit's preset pass manager with Target integration.
        
        Uses generate_preset_pass_manager() for full 6-stage transpilation pipeline:
        init → layout → routing → translation → optimization → scheduling.
        
        This provides access to advanced IBM backend features like dynamic circuits,
        pulse-level control, and Target-driven compilation with exact hardware constraints.
        
        Falls back to qiskit.compiler.transpile() if backend doesn't support Target
        (e.g., AerSimulator, legacy backends).
        
        Args:
            backend: Qiskit backend instance (native runtime or AerSimulator)
            opt_level: Optimization level for transpilation (0-3)
            
        Returns:
            QuantumCircuit: Transpiled circuit in Qiskit format
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.compiler import transpile
        
        # Ensure main circuit is in Qiskit format
        if not hasattr(self.main_circuit, 'qubits'):
            raise TypeError(
                f"Main circuit must be a Qiskit QuantumCircuit for Qiskit transpilation. "
                f"Got {type(self.main_circuit)}. Rebuild circuit with sdk='qiskit'."
            )
        
        try:
            # Try Target-driven transpilation with generate_preset_pass_manager
            # This is the modern approach for IBM runtime backends
            if hasattr(backend, 'target') and backend.target is not None:
                pm = generate_preset_pass_manager(
                    optimization_level=opt_level,
                    backend=backend,
                )
                tcirc = pm.run(self.main_circuit)
                return tcirc
            else:
                # Fallback to compiler.transpile for backends without Target
                # (e.g., AerSimulator, older backends)
                tcirc = transpile(
                    self.main_circuit,
                    backend=backend,
                    optimization_level=opt_level,
                )
                return tcirc
        except Exception as e:
            raise RuntimeError(
                f"Qiskit transpilation failed at opt_level {opt_level}: {e}"
            )
    
    def _transpile_braket(self, backend, opt_level: int):
        """Braket doesn't support client-side transpilation.
        
        AWS Braket performs all transpilation server-side during job execution.
        The transpiled circuit can be accessed after execution through the task result.
        
        Args:
            backend: Braket backend instance (unused)
            opt_level: Optimization level (unused)
            
        Raises:
            NotImplementedError: Always, as Braket doesn't support pre-transpilation
        """
        raise NotImplementedError(
            "AWS Braket performs transpilation server-side. "
            "Pre-transpilation is not supported. "
            "Access transpiled circuit information after execution via task.result(). "
            "Use puzzle.get_transpiled_circuit_from_task(task_result) to extract "
            "transpilation information from completed jobs."
        )