import gc
import math
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
from pathlib import Path

from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
from sudoku_nisq.backends import BackendManager

if TYPE_CHECKING:
    from sudoku_nisq.quantum_solver import QuantumSolver
    try:
        from pytket.extensions.quantinuum.backends.credential_storage import CredentialStorage  # noqa: F401
    except Exception:  # pragma: no cover
        pass

class QSudoku():
    """High-level interface for quantum Sudoku solving with integrated backend management.
    
    QSudoku provides a unified interface for solving Sudoku puzzles using quantum algorithms.
    It combines puzzle management, quantum solver integration, backend handling, and result
    visualization into a single, easy-to-use class. The design supports research workflows
    with automatic memory management and comprehensive caching.
    
    Key Features:
    - Single active solver architecture with automatic cleanup
    - Integrated quantum backend management (IBM, Quantinuum, simulators)
    - Comprehensive caching and metadata tracking
    - Built-in visualization and analysis tools
    - Factory methods for puzzle generation and loading
    
    Attributes:
        puzzle (SudokuPuzzle): The underlying Sudoku puzzle instance.
        _solver (Optional[QuantumSolver]): Currently active quantum solver instance.
        _attached_backends (Dict[str, Any]): Dictionary of attached quantum backends by alias.
        cache_base (Path): Base directory for caching circuits and metadata.
        
    Example:
        .. code-block:: python

            # Generate a new puzzle and set up quantum solving
            puzzle = QSudoku.generate(subgrid_size=3, num_missing_cells=20)

            # Set up exact cover solver
            from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
            puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

            # Initialize quantum backend
            puzzle.init_ibm(api_token="your_token", instance="your_instance", device="ibm_brisbane")

            # Build and run quantum circuit
            circuit = puzzle.build_circuit()
            result = puzzle.run("ibm_brisbane", opt_level=1, shots=1024)
            puzzle.counts_plot(result, backend_alias="IBM Brisbane")

            # Analyze resources
            resources = puzzle.report_resources()
            print(f"Circuit requires {resources['main_circuit']['n_qubits']} qubits")
    """
    def __init__(self, puzzle: SudokuPuzzle, cache_base: Optional[str] = None):
        """Initialize a QSudoku instance with puzzle and caching configuration.
        
        Sets up the quantum Sudoku solving environment by wrapping a SudokuPuzzle
        instance with quantum solver management, backend integration, and metadata
        tracking capabilities.
        
        Args:
            puzzle (SudokuPuzzle): The Sudoku puzzle instance to wrap and solve.
                Must be a valid SudokuPuzzle with defined constraints and structure.
            cache_base (Optional[str]): Base directory path for caching quantum circuits
                and metadata. If None, defaults to ".quantum_solver_cache" in the
                current working directory.
                
        Note:
            The QSudoku instance maintains a single active solver at a time. Use
            set_solver() to configure quantum algorithms and encoding strategies.
            Backend connections are managed separately via init_ibm(), init_quantinuum(),
            or attach_backend() methods.
        """
        
        self.puzzle = puzzle
        
        # Initialize solver management
        self._solver: Optional["QuantumSolver"] = None
        self._attached_backends: Dict[str, Any] = {}
        self._validation_context: Optional[Any] = None
        
        # Cache base for stage managers
        self.cache_base = Path(cache_base) if cache_base else Path(".quantum_solver_cache")
    
    @classmethod
    def generate(
        cls,
        subgrid_size: int = 3,
        num_missing_cells: int = 20,
        canonicalize: bool = False,
        cache_base: Optional[str] = None,
        *,
        size: Optional[int] = None,
    ) -> "QSudoku":
        """Factory method for generating a QSudoku instance with a random puzzle.
        
        Creates a new QSudoku instance by generating a random Sudoku puzzle with
        specified parameters. This is the recommended way to create new puzzles
        for quantum algorithm research and benchmarking.
        
        Args:
            subgrid_size (int): Backward-compatible subgrid dimension k (e.g., 3 for standard 9x9,
                2 for 4x4). Determines overall board size as k² when ``size`` is not provided.
            num_missing_cells (int): Number of cells to leave empty in the generated
                puzzle. Higher values create more challenging puzzles but may affect
                quantum algorithm performance.
            canonicalize (bool): Whether to canonicalize the generated puzzle by
                relabeling digits to ensure a standard form. Defaults to False.
            cache_base (Optional[str]): Base directory for caching quantum circuits
                and metadata. If None, uses default cache location.
            size (Optional[int]): New optional overall grid size N. If provided,
                the generator targets an N×N puzzle. Supported values: 2 (special case,
                no real subgrids) or perfect squares like 4, 9, 16 (with subgrids of
                size sqrt(N)). If both ``size`` and ``subgrid_size`` are given, ``size``
                takes precedence and must be consistent when applicable.
                
        Returns:
            QSudoku: A new QSudoku instance with the generated puzzle ready for
                quantum solver integration.
                
        Example:
            .. code-block:: python

                # Generate standard 9x9 Sudoku with 20 missing cells
                puzzle = QSudoku.generate(size=9, num_missing_cells=20)

                # Generate challenging 9x9 puzzle with canonicalization
                hard_puzzle = QSudoku.generate(
                    subgrid_size=3,
                    num_missing_cells=50,
                    canonicalize=True,
                    cache_base="./my_cache"
                )

                # Generate 4x4 mini-Sudoku for testing
                mini_puzzle = QSudoku.generate(size=4, num_missing_cells=8)
        """
        # Map overall size N to subgrid_size k when provided.
        # Special-case N=2 -> k=1 (no real subgrids); otherwise require N to be a perfect square.
        if size is not None:
            if size == 2:
                k = 1
            else:
                k = math.isqrt(size)
                if k * k != size:
                    raise ValueError(f"size must be 2 or a perfect square (e.g., 4, 9, 16); got {size}")
            if subgrid_size is not None and subgrid_size != k:
                # Enforce consistency if caller also provided subgrid_size
                subgrid_size = k

        # Generate the puzzle using SudokuPuzzle (supports optional size)
        puzzle = SudokuPuzzle.generate(
            subgrid_size=subgrid_size,
            num_missing_cells=num_missing_cells,
            canonicalize=canonicalize,
            size=size,
        )

        # Wrap the puzzle in a QSudoku instance
        return cls(puzzle=puzzle, cache_base=cache_base)

    @classmethod
    def from_board(cls, board: List[List[int]], canonicalize: bool = False, cache_base: Optional[str] = None) -> "QSudoku":
        """Factory method for creating a QSudoku instance from an existing board.
        
        Creates a QSudoku instance from a pre-defined Sudoku board configuration.
        This method is useful for loading specific puzzles, benchmark problems,
        or custom test cases into the quantum solving framework.
        
        Args:
            board (List[List[int]]): 2D list representing the Sudoku board where
                each cell contains either a digit (1 to board_size) or 0/None for
                empty cells. The board must be square with size = subgrid_size².
            canonicalize (bool): Whether to canonicalize the board by relabeling
                digits to ensure standard form. Useful for puzzle comparison and
                analysis. Defaults to False.
            cache_base (Optional[str]): Base directory for caching quantum circuits
                and metadata. If None, uses default cache location.
                
        Returns:
            QSudoku: A new QSudoku instance with the provided board ready for
                quantum algorithm application.
                
        Raises:
            ValueError: If the board format is invalid or contains inconsistent
                Sudoku constraints.
                
        Example:
            .. code-block:: python

                # Load a specific 4x4 puzzle
                board_4x4 = [
                    [1, 0, 0, 4],
                    [0, 0, 1, 0],
                    [0, 4, 0, 0],
                    [2, 0, 0, 3]
                ]
                puzzle = QSudoku.from_board(board_4x4)

                # Load and canonicalize a 9x9 benchmark puzzle
                benchmark_board = load_benchmark_puzzle("hard_01.txt")
                puzzle = QSudoku.from_board(
                    board=benchmark_board,
                    canonicalize=True,
                    cache_base="./benchmarks_cache"
                )
        """
        # Create the puzzle using SudokuPuzzle.from_board()
        puzzle = SudokuPuzzle.from_board(board=board, canonicalize=canonicalize)

        # Wrap the puzzle in a QSudoku instance
        return cls(puzzle=puzzle, cache_base=cache_base)

    @property
    def board(self) -> List[List[int]]:
        """Get the current puzzle board as a 2D list.
        
        Returns:
            List[List[int]]: 2D list representing the Sudoku board where each cell
                contains a digit (1 to board_size) or 0 for empty cells.
        """
        return self.puzzle.board
    
    @property
    def board_size(self) -> int:
        """Get the total size of the Sudoku board.
        
        Returns:
            int: Board dimension (e.g., 4 for 4x4, 9 for 9x9 Sudoku). Equal to
                subgrid_size².
        """
        return self.puzzle.board_size
    
    @property
    def subgrid_size(self) -> int:
        """Get the subgrid size of the Sudoku puzzle.
        
        Returns:
            int: Subgrid dimension (e.g., 2 for 4x4, 3 for 9x9 Sudoku). The board
                is divided into subgrid_size² subgrids, each of size subgrid_size².
        """
        return self.puzzle.subgrid_size
    
    @property
    def num_missing_cells(self) -> int:
        """Get the number of empty cells in the puzzle.
        
        Returns:
            int: Count of cells that need to be filled to complete the puzzle.
                This affects quantum algorithm complexity and resource requirements.
        """
        return self.puzzle.num_missing_cells

    @property
    def quantum_solver(self) -> Optional["QuantumSolver"]:
        """Get the currently active quantum solver instance.
        
        Returns:
            Optional[QuantumSolver]: The active solver, or None if no solver is set.
                Used by BenchmarkSession to access solver configuration and resources.
        
        Example:
            .. code-block:: python

                puzzle = QSudoku.generate(size=2)
                puzzle.set_solver(ExactCoverQuantumSolver)
                solver = puzzle.quantum_solver
                print(solver.solver_name, solver.encoding)
        """
        return self._solver

    def plot_puzzle(self) -> None:
        """Plot the current Sudoku board using matplotlib visualization.
        
        Creates a visual representation of the puzzle with grid lines and filled
        numbers. Useful for puzzle inspection and result verification.
        
        Returns:
            matplotlib.figure.Figure: The matplotlib figure object containing the plot.
            
        Example:
            .. code-block:: python

                puzzle = QSudoku.generate(subgrid_size=3, num_missing_cells=20)
                fig = puzzle.plot_puzzle()
                fig.savefig("my_puzzle.png")
        """
        return self.puzzle.plot()

    def init_ibm(self, api_token: str, instance: str, device: str, alias: Optional[str] = None) -> str:
        """Initialize and attach IBM Quantum backend to this puzzle instance.
        
        Sets up authentication with IBM Quantum Platform and registers the specified
        device for quantum circuit execution. The backend is automatically attached
        to this QSudoku instance for immediate use.
        
        Args:
            api_token (str): Your IBM Quantum Platform API token for authentication.
            instance (str): Your IBM Quantum instance CRN (Customer Resource Name),
                a long string beginning with "crn:" that identifies your specific
                IBM Quantum service instance.
            device (str): IBM Quantum device name (e.g., "ibm_brisbane", "ibm_kyiv",
                "simulator_statevector"). Must be available to your account.
            alias (Optional[str]): Custom alias for the device. If None, uses the
                device name as the alias.
                
        Returns:
            str: The alias used for the registered backend, which can be used in
                run() and transpile() calls.
                
        Raises:
            RuntimeError: If authentication fails or device is not available.
            
        Example:
            .. code-block:: python

                puzzle = QSudoku.generate(subgrid_size=3, num_missing_cells=20)

                # Initialize IBM backend
                alias = puzzle.init_ibm(
                    api_token="your_api_token_here",
                    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/...",
                    device="ibm_brisbane",
                    alias="brisbane"
                )

                # Use the backend for execution
                result = puzzle.run(alias, opt_level=1, shots=1024)
        """
        alias = BackendManager.inst().init_ibm(device=device, alias=alias, api_token=api_token, instance=instance)
        self.attach_backend(alias)
        return alias

    def init_quantinuum(self, device: str, alias: Optional[str] = None, token_store: Optional['CredentialStorage'] = None, provider: Optional[str] = None) -> str:
        """Initialize and attach Quantinuum quantum backend to this puzzle instance.
        
        Sets up authentication with Quantinuum's quantum cloud platform and registers
        the specified device for quantum circuit execution. The backend is automatically
        attached to this QSudoku instance for immediate use.
        
        Args:
            device (str): Quantinuum device name (e.g., "H1-1", "H1-2", "H2-1", "H2-2E").
                Must be available to your Quantinuum account.
            alias (Optional[str]): Custom alias for the device. If None, uses the
                device name as the alias.
            token_store (Optional[CredentialStorage]): Storage mechanism for authentication
                tokens. If None, defaults to in-memory storage requiring re-authentication
                each session.
            provider (Optional[str]): Specific Quantinuum provider identifier. Can be
                None to use the default provider.
                
        Returns:
            str: The alias used for the registered backend, which can be used in
                run() and transpile() calls.
                
        Raises:
            RuntimeError: If authentication fails or device is not available.
            
        Example:
            .. code-block:: python

                puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=8)

                # Initialize Quantinuum backend with persistent storage
                from pytket.extensions.quantinuum.backends.credential_storage import DiskCredentialStorage

                alias = puzzle.init_quantinuum(
                    device="H1-1",
                    alias="h1_device",
                    token_store=DiskCredentialStorage()
                )

                # Use the backend for execution
                result = puzzle.run(alias, opt_level=2, shots=500)
        """
        alias = BackendManager.inst().init_quantinuum(device=device, alias=alias, token_store=token_store, provider=provider)
        self.attach_backend(alias)
        return alias
    
    def init_aer(
        self,
        method: str = "automatic",
        noise_model: Any = None,
        coupling_map: Any = None,
        basis_gates: list[str] | None = None,
        device: str = "CPU",
        precision: str = "double",
        alias: Optional[str] = None,
        **backend_options
    ) -> str:
        """Initialize and attach Qiskit Aer local simulator to this puzzle instance.
        
        Configures a local Aer simulator with specified simulation method, noise model,
        and performance options. The simulator is automatically attached to this QSudoku
        instance for immediate use. Supports ideal and noisy simulation, GPU acceleration,
        and device emulation.
        
        Args:
            method (str, optional): Simulation method. Options:
                - "automatic": Auto-select based on circuit (default)
                - "statevector": Dense statevector simulation
                - "density_matrix": Density matrix simulation (supports noise)
                - "stabilizer": Clifford stabilizer simulator
                - "extended_stabilizer": Approximate Clifford+T simulator
                - "matrix_product_state": MPS/tensor network simulator
                - "unitary": Compute circuit unitary
                - "superop": Compute superoperator representation
            noise_model (NoiseModel, optional): Qiskit Aer noise model for noisy
                simulation. Can be created from real devices via NoiseModel.from_backend().
            coupling_map (list or CouplingMap, optional): Device coupling map for
                layout constraints and hardware emulation.
            basis_gates (list, optional): Basis gates for device emulation.
            device (str, optional): Compute device: "CPU" or "GPU". Defaults to "CPU".
            precision (str, optional): Float precision: "single" or "double".
                Defaults to "double".
            alias (Optional[str]): Custom alias for the simulator. If None, uses
                "aer_{method}" as the alias.
            **backend_options: Additional AerSimulator backend options (e.g.,
                max_parallel_threads, blocking_qubits, etc.).
                
        Returns:
            str: The alias used for the registered simulator, which can be used in
                run() and transpile() calls.
                
        Examples:
            Ideal statevector simulation:
            
            >>> puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=4)
            >>> alias = puzzle.init_aer(method="statevector")
            >>> result = puzzle.run(alias, opt_level=1, shots=1024)
            
            GPU-accelerated simulation:
            
            >>> alias = puzzle.init_aer(
            ...     method="statevector",
            ...     device="GPU",
            ...     precision="single",
            ...     alias="aer_gpu"
            ... )
            
            Device emulation with noise:
            
            >>> from qiskit_ibm_runtime import QiskitRuntimeService
            >>> from qiskit_aer.noise import NoiseModel
            >>> service = QiskitRuntimeService()
            >>> real_backend = service.backend("ibm_brisbane")
            >>> noise = NoiseModel.from_backend(real_backend)
            >>> alias = puzzle.init_aer(
            ...     method="density_matrix",
            ...     noise_model=noise,
            ...     coupling_map=real_backend.coupling_map,
            ...     basis_gates=real_backend.configuration().basis_gates,
            ...     alias="brisbane_sim"
            ... )
            >>> result = puzzle.run(alias, opt_level=2, shots=4096)
        """
        alias = BackendManager.inst().init_aer(
            device=method,  # device param is used as method in AerProvider
            method=method,
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            device_type=device,
            precision=precision,
            alias=alias,
            **backend_options
        )
        self.attach_backend(alias)
        return alias
    
    def run_aer_with_noise(
        self,
        shots: int = 1024,
        noise_model: Any = None,
        device_name: Optional[str] = None,
        method: str = "density_matrix",
        optimization_level: int = 1,
        **aer_options
    ) -> Dict[str, Any]:
        """Execute circuit on Aer with noise model (custom or auto-generated from device).
        
        Convenience method for running noisy simulations. Can either accept a custom
        noise model or automatically generate one from a real IBM Quantum device name.
        Uses density_matrix simulation method by default for accurate noisy simulation.
        
        Args:
            shots (int, optional): Number of measurement samples. Defaults to 1024.
            noise_model (NoiseModel, optional): Custom Qiskit Aer noise model.
                If None and device_name is provided, noise model is auto-generated
                from the specified device.
            device_name (Optional[str]): IBM Quantum device name to generate noise
                model from (e.g., "ibm_brisbane", "ibm_kyiv"). Ignored if noise_model
                is provided.
            method (str, optional): Aer simulation method. Defaults to "density_matrix"
                which is recommended for noisy simulation. Can use "automatic" for
                auto-selection or "statevector" for faster approximate noise.
            optimization_level (int, optional): Transpiler optimization level (0-3).
                Defaults to 1.
            **aer_options: Additional arguments passed to run_aer() such as device,
                precision, coupling_map, basis_gates, etc.
                
        Returns:
            Dict[str, Any]: Simulation results including raw counts, execution time,
                and metadata.
                
        Raises:
            ValueError: If no solver is set or if neither noise_model nor device_name
                is provided.
            ImportError: If qiskit-aer or required packages are not installed.
            
        Examples:
            With custom noise model:
            
            >>> from qiskit_aer.noise import NoiseModel, depolarizing_error
            >>> noise = NoiseModel()
            >>> noise.add_all_qubit_quantum_error(
            ...     depolarizing_error(0.01, 2), ['cx']
            ... )
            >>> result = puzzle.run_aer_with_noise(
            ...     shots=4096,
            ...     noise_model=noise
            ... )
            
            Auto-generate from device:
            
            >>> result = puzzle.run_aer_with_noise(
            ...     shots=8192,
            ...     device_name="ibm_brisbane",
            ...     optimization_level=2
            ... )
            
            With GPU acceleration:
            
            >>> result = puzzle.run_aer_with_noise(
            ...     shots=2048,
            ...     device_name="ibm_kyiv",
            ...     device="GPU",
            ...     precision="single"
            ... )
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        
        # Auto-generate noise model from device if needed
        if noise_model is None and device_name is not None:
            try:
                from qiskit_aer.noise import NoiseModel
                from qiskit.providers.fake_provider import GenericBackendV2
                
                # Try to get fake backend for noise model generation
                try:
                    # First try with real backend name
                    fake_backend = GenericBackendV2.from_backend_name(device_name)
                except Exception:
                    # If that fails, try common alternatives
                    raise ValueError(
                        f"Could not generate noise model from device '{device_name}'. "
                        "Provide a valid IBM device name or a custom noise_model."
                    )
                
                noise_model = NoiseModel.from_backend(fake_backend)
                
                # Also get coupling map and basis gates for realistic emulation
                if 'coupling_map' not in aer_options:
                    aer_options['coupling_map'] = fake_backend.coupling_map
                if 'basis_gates' not in aer_options:
                    aer_options['basis_gates'] = fake_backend.configuration().basis_gates
                    
            except ImportError as e:
                raise ImportError(
                    "qiskit-aer and qiskit fake providers are required for device noise models. "
                    "Install with: pip install qiskit-aer"
                ) from e
        elif noise_model is None and device_name is None:
            raise ValueError(
                "Either noise_model or device_name must be provided for noisy simulation."
            )
        
        # Run with Aer and noise
        return self._solver.run_aer(
            shots=shots,
            method=method,
            noise_model=noise_model,
            optimization_level=optimization_level,
            **aer_options
        )
    
    def set_solver(self, solver_class: Type["QuantumSolver"], encoding: Optional[str] = None, **solver_kwargs) -> Optional["QuantumSolver"]:
        """Set the active quantum solver for this puzzle with automatic memory management.
        
        Configures the quantum algorithm to use for solving the Sudoku puzzle.
        Automatically handles cleanup of any previous solver and initializes the
        new solver with the current puzzle state and metadata tracking.
        
        Args:
            solver_class (Type[QuantumSolver]): Class of the quantum solver to use
                (e.g., ExactCoverQuantumSolver, BacktrackingQuantumSolver).
            encoding (Optional[str]): Encoding strategy for quantum representation.
                Options depend on the solver (e.g., "simple", "pattern" for exact cover).
                If None, uses the solver's default encoding.
            **solver_kwargs: Additional keyword arguments passed to the solver
                constructor for algorithm-specific configuration.
                
        Returns:
            Optional[QuantumSolver]: The newly created and configured solver instance.
            
        Note:
            Only one solver can be active at a time. Setting a new solver automatically
            releases memory from the previous solver and updates metadata tracking.
            
        Example:
            .. code-block:: python

                from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
                from sudoku_nisq.backtracking_solver import BacktrackingQuantumSolver

                puzzle = QSudoku.generate(subgrid_size=3, num_missing_cells=20)

                # Set exact cover solver with simple encoding
                puzzle.set_solver(
                    ExactCoverQuantumSolver,
                    encoding="simple",
                    store_transpiled=True
                )

                # Switch to different algorithm
                puzzle.set_solver(
                    BacktrackingQuantumSolver,
                    encoding="binary",
                    max_iterations=1000
                )
        """
        # Create new solver
        new_solver = solver_class(
            puzzle=self.puzzle,
            cache_base=self.cache_base,
            encoding=encoding,
            **solver_kwargs
        )
        
        # Swap in new solver (cleanup handled internally)
        self._swap_solver(new_solver)
        
        return self._solver
    
    def drop_solver(self) -> None:
        """Explicitly release the current solver and free associated memory.
        
        Removes the active solver instance and triggers garbage collection to
        free memory. This is useful for memory management when switching between
        different algorithms or when solver is no longer needed.
        
        Note:
            After calling this method, you must call set_solver() again before
            using any solver-dependent methods like build_circuit() or run().
            
        Example:
            .. code-block:: python

                puzzle.set_solver(ExactCoverQuantumSolver)
                circuit = puzzle.build_circuit()

                # Free solver memory when done
                puzzle.drop_solver()

                # Must set solver again for further use
                puzzle.set_solver(BacktrackingQuantumSolver)
        """
        if self._solver:
            del self._solver
            self._solver = None
            gc.collect()
    
    def set_validation_context(self, valid_solutions: List[str]) -> None:
        """Configure validation context for automatic metrics computation.
        
        Sets up the validation context that enables automatic Stage 6-7 metrics
        recording during quantum execution. When a validation context is provided,
        the solver will automatically compute evaluation metrics (success probability,
        fidelity, top-k statistics) and normalization metrics (gate efficiency,
        circuit volume efficiency, shot efficiency) after each execution.
        
        Args:
            valid_solutions (List[str]): List of valid solution bitstrings to check
                against measurement outcomes. Each string should be a binary string
                matching the quantum circuit's output format (e.g., "0110" for a
                4-qubit measurement).
                
        Note:
            The validation context persists across multiple run() and run_aer() calls
            until explicitly cleared with clear_validation_context() or a new context
            is set. Metrics are recorded automatically if the new metadata architecture
            is enabled (MetadataConfig.ENABLE_NEW_ARCHITECTURE=True).
            
        Example:
            .. code-block:: python

                from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
                
                puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=2)
                puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
                
                # Get known valid solutions (from classical solver or manual entry)
                valid_solutions = ["0110", "1001"]  # Example bitstrings
                puzzle.set_validation_context(valid_solutions)
                
                # Run with automatic metrics recording
                result = puzzle.run_aer(shots=1024)
                
                # Metrics are automatically recorded to Stage 6-7 JSON files
                # Query them via MetricsMetadataManager if needed
        """
        from sudoku_nisq.metrics.data_models import ValidationContext
        
        # Create validator function
        valid_solutions_set = set(valid_solutions)
        
        def solution_validator(bitstring: str) -> bool:
            return bitstring in valid_solutions_set
        
        self._validation_context = ValidationContext(
            valid_solutions=list(valid_solutions),
            total_valid_count=len(valid_solutions),
            solution_validator=solution_validator
        )
    
    def clear_validation_context(self) -> None:
        """Remove the validation context to disable automatic metrics computation.
        
        Clears the current validation context, which disables automatic Stage 6-7
        metrics recording during quantum execution. Execution will continue normally
        but only Stage 5 execution metadata will be recorded.
        
        Example:
            .. code-block:: python

                # Set up validation for benchmarking
                puzzle.set_validation_context(["0110", "1001"])
                result1 = puzzle.run_aer(shots=1024)  # Metrics recorded
                
                # Disable metrics for quick testing
                puzzle.clear_validation_context()
                result2 = puzzle.run_aer(shots=256)  # Only execution data recorded
        """
        self._validation_context = None
    
    def _swap_solver(self, new_solver: "QuantumSolver") -> None:
        """Internal method to safely replace the current solver with cleanup.
        
        Handles the internal mechanics of solver replacement including memory
        cleanup of the previous solver and installation of the new one.
        
        Args:
            new_solver (QuantumSolver): The new solver instance to install.
            
        Note:
            This is an internal method used by set_solver(). Users should not
            call this method directly.
        """
        if self._solver:
            del self._solver
            gc.collect()
        self._solver = new_solver
    
    def build_circuit(self, sdk: str | None = None):
        """Build the main quantum circuit using the active solver.
        
        Constructs the quantum circuit that implements the selected solving algorithm
        for the current puzzle. The circuit is built according to the solver's
        encoding strategy and algorithm parameters.
        
        Args:
            sdk (str | None, optional): Explicitly select which SDK to use for circuit
                construction. Valid values are 'pytket', 'qiskit', or 'braket'.
                If None (default), SDK selection follows this priority:
                1. Use the backend's provider SDK if a backend was initialized
                2. Default to PyTKET if no backend is available
                This allows SDK comparison, testing without backends, and research workflows.
        
        Returns:
            Circuit: A quantum circuit object in the format of the selected SDK
                (pytket.Circuit, qiskit.QuantumCircuit, or braket.circuits.Circuit).
            
        Raises:
            ValueError: If no solver has been set using set_solver(), or if an invalid
                SDK name is provided.
            
        Examples:
            .. code-block:: python

                # Default: automatic SDK selection based on backend
                puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
                circuit = puzzle.build_circuit()
                
                # Explicit SDK selection (no backend needed)
                qiskit_circuit = puzzle.build_circuit(sdk="qiskit")
                pytket_circuit = puzzle.build_circuit(sdk="pytket")
                
                # Override backend's SDK for comparison
                puzzle.initialize_ibm(token="...")
                qiskit_circ = puzzle.build_circuit(sdk="qiskit")  # Use IBM's default
                pytket_circ = puzzle.build_circuit(sdk="pytket")  # Compare with PyTKET
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        return self._solver.build_main_circuit(sdk=sdk)
    
    def draw_circuit(self, circuit = None):
        """Visualize the quantum circuit using matplotlib rendering.
        
        Creates a visual representation of the quantum circuit showing gates,
        qubits, and circuit structure. Useful for algorithm analysis and
        educational purposes.
        
        Args:
            circuit (Optional[Circuit]): Specific circuit to visualize. If None,
                uses the main circuit from the current solver.
                
        Returns:
            The rendered circuit visualization object.
            
        Raises:
            ValueError: If no solver has been set and no circuit is provided.
            
        Example:
            .. code-block:: python

                puzzle.set_solver(ExactCoverQuantumSolver)
                circuit = puzzle.build_circuit()

                # Visualize the built circuit
                puzzle.draw_circuit()

                # Visualize a specific circuit
                puzzle.draw_circuit(circuit)
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        
        if circuit is None:
            circuit = self._solver.build_main_circuit()

        return self._solver.draw_circuit(circuit)

    def attach_backend(self, alias: str) -> None:
        """Attach a backend from the global registry to this puzzle instance.
        
        Associates a previously registered backend with this QSudoku instance,
        making it available for quantum circuit execution. The backend must
        already be registered in the global BackendManager registry.
        
        Args:
            alias (str): Alias of the backend to attach. Must exist in the global
                BackendManager registry.
                
        Raises:
            ValueError: If the backend alias is not found in the global registry.
            
        Example:
            .. code-block:: python

                # Assuming a backend was registered globally
                from sudoku_nisq.backends import BackendManager
                BackendManager.init_ibm(api_token, instance, "ibm_brisbane", "brisbane")

                # Attach to this puzzle instance
                puzzle.attach_backend("brisbane")
                result = puzzle.run("brisbane", opt_level=1, shots=1024)
        """
        # Fail fast validation
        backend = BackendManager.inst().get(alias)  # Raises if not found
        self._attached_backends[alias] = backend
    
    def transpile(self, backend_alias: str, opt_level: int, **kwargs):
        """Transpile the quantum circuit for a specific hardware backend and return it.
        
        Compiles the circuit generated by the active solver to be compatible
        with the target quantum hardware, applying optimizations and gate
        decompositions as needed. Returns the transpiled circuit object.
        
        Args:
            backend_alias (str): Alias of the attached backend to transpile for.
                Must have been attached via init_ibm(), init_quantinuum(), or
                attach_backend().
            opt_level (int): Optimization level for transpilation (typically 0-3).
                Higher levels may reduce circuit size but take longer to compile.
            **kwargs: Additional keyword arguments passed to the transpilation
                process (e.g., initial_layout, routing_method for Qiskit).
                
        Returns:
            Circuit or QuantumCircuit: The transpiled circuit object in the format
                appropriate for the backend's SDK (pytket.Circuit for PyTKET backends,
                qiskit.QuantumCircuit for Qiskit backends).
                
        Raises:
            ValueError: If backend is not attached, no solver is set, or Braket backend
                (which doesn't support client-side transpilation).
            
        Example:
            .. code-block:: python

                puzzle.init_ibm(api_token, instance, "ibm_brisbane", "brisbane")
                puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
                
                # Build and transpile circuit
                circuit = puzzle.build_circuit(sdk="qiskit")
                transpiled = puzzle.transpile("brisbane", opt_level=2)
                
                # Inspect transpiled circuit properties
                print(f"Original gates: {circuit.count_ops()}")
                print(f"Transpiled gates: {transpiled.count_ops()}")
        """
        if backend_alias not in self._attached_backends:
            attached = list(self._attached_backends.keys())
            raise ValueError(f"Backend '{backend_alias}' not attached. Attached: {attached}")
        
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        
        # Get transpilation results (includes metrics and caching)
        result = self._solver.transpile_and_analyze(
            self._attached_backends[backend_alias],
            backend_alias,
            opt_level,
            **kwargs
        )
        
        # If error occurred, raise it
        if "error" in result:
            raise RuntimeError(f"Transpilation failed: {result['error']}")
        
        # Load and return the transpiled circuit
        return self.get_transpiled_circuit(backend_alias, opt_level)
    
    def get_transpiled_circuit(self, backend_alias: str, opt_level: int = 0):
        """Retrieve a cached transpiled circuit for a specific backend.
        
        Loads a previously transpiled circuit from cache. The circuit must have
        been transpiled before using transpile() or run() methods.
        
        Args:
            backend_alias (str): Alias of the backend the circuit was transpiled for.
            opt_level (int, optional): Optimization level of the cached transpilation.
                Defaults to 0.
                
        Returns:
            Circuit or QuantumCircuit: The transpiled circuit in SDK-appropriate format.
                
        Raises:
            ValueError: If no solver is set or transpiled circuit doesn't exist in cache.
            FileNotFoundError: If no transpiled circuit exists for the given parameters.
            
        Example:
            .. code-block:: python

                # Transpile and cache
                puzzle.run("brisbane", opt_level=2, shots=1024)
                
                # Later, retrieve the cached transpiled circuit
                transpiled_circuit = puzzle.get_transpiled_circuit("brisbane", opt_level=2)
                
                # Visualize or analyze
                puzzle.draw_circuit(transpiled_circuit)
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        
        # Detect SDK type from backend
        backend = None
        if backend_alias in self._attached_backends:
            backend = self._attached_backends[backend_alias]
        else:
            try:
                backend = BackendManager.inst().get(backend_alias)
            except ValueError:
                # Backend not found - still try to load from cache
                pass
        
        if backend is not None:
            sdk_type = self._solver._detect_backend_sdk(backend)
            path = self._solver.transpiled_circuit_path(backend_alias, opt_level, sdk_type=sdk_type)
            
            if not path.exists():
                raise FileNotFoundError(
                    f"No transpiled circuit found for {backend_alias} at opt_level {opt_level}. "
                    f"Run transpile() or run() first."
                )
            
            # Load circuit based on SDK type
            if sdk_type == "pytket":
                return self._solver.load_circuit(path)
            elif sdk_type == "qiskit":
                return self._solver._load_qiskit_circuit(path)
            else:
                # Try loading as pytket by default
                return self._solver.load_circuit(path)
        else:
            # Backend not known - try to find any cached circuit
            # Check for pytket format first (most common)
            path_pytket = self._solver.transpiled_circuit_path(backend_alias, opt_level, sdk_type="pytket")
            if path_pytket.exists():
                return self._solver.load_circuit(path_pytket)
            
            # Try qiskit format
            path_qiskit = self._solver.transpiled_circuit_path(backend_alias, opt_level, sdk_type="qiskit")
            if path_qiskit.exists():
                return self._solver._load_qiskit_circuit(path_qiskit)
            
            # Try backward-compatible path (no SDK type)
            path_legacy = self._solver.transpiled_circuit_path(backend_alias, opt_level, sdk_type=None)
            if path_legacy.exists():
                return self._solver.load_circuit(path_legacy)
            
            # Nothing found
            raise FileNotFoundError(
                f"No transpiled circuit found for {backend_alias} at opt_level {opt_level}. "
                f"Run transpile() or run() first."
            )

    def run(self, backend_alias: str, opt_level: int, shots: int, **kwargs):
        """Execute the quantum circuit on the specified hardware backend.
        
        Runs the quantum circuit on the target quantum hardware or simulator,
        applying transpilation and collecting the specified number of measurement
        samples.
        
        Args:
            backend_alias (str): Alias of the attached backend to run on.
                Must have been attached via init_ibm(), init_quantinuum(), or
                attach_backend().
            opt_level (int): Optimization level for transpilation (typically 0-3).
                Higher levels may reduce circuit size but take longer to compile.
            shots (int): Number of measurement samples to collect. Higher values
                provide better statistics but take longer to execute.
            **kwargs: Additional keyword arguments passed to the quantum job
                submission.
                
        Returns:
            Dict[str, Any]: Dictionary containing execution results including
                raw counts, job metadata, and timing information.
                
        Raises:
            ValueError: If backend is not attached or no solver is set.
            RuntimeError: If quantum job execution fails.
            
        Example:
            .. code-block:: python

                puzzle.init_ibm(api_token, instance, "ibm_brisbane", "brisbane")
                puzzle.set_solver(ExactCoverQuantumSolver)

                # Run with different configurations
                quick_result = puzzle.run("brisbane", opt_level=0, shots=512)
                optimized_result = puzzle.run("brisbane", opt_level=2, shots=1024)

                print(f"Quick run: {len(quick_result['counts'])} outcomes")
                print(f"Optimized run: {len(optimized_result['counts'])} outcomes")
        
        TODO:
            Unify naming between ``opt_level`` (public QSudoku API) and
            ``optimisation_level`` used internally by solver backends.
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        
        # Get backend from global registry (singleton facade)
        try:
            backend = BackendManager.inst().get(backend_alias)
        except ValueError as e:
            # Fallback to attached backends for backward compatibility
            if backend_alias in self._attached_backends:
                backend = self._attached_backends[backend_alias]
            else:
                available_global = BackendManager.inst().all()
                available_attached = list(self._attached_backends.keys())
                raise ValueError(
                    f"Backend '{backend_alias}' not found in global registry or attached backends. "
                    f"Global: {available_global}, Attached: {available_attached}"
                ) from e
            
        return self._solver.run(
            backend,
            backend_alias,
            shots,
            force_run=False,
            optimisation_level=opt_level,
            validation_context=self._validation_context,
            **kwargs
        )
    
    def run_aer(self, shots: int = 1024, **kwargs):
        """Execute the quantum circuit on the local Aer simulator.
        
        Runs the logical quantum circuit on Qiskit Aer simulator without
        hardware-specific transpilation. Useful for testing and debugging
        quantum algorithms without noise or hardware limitations.
        
        Args:
            shots (int, optional): Number of measurement samples to collect.
                Defaults to 1024. Higher values provide better statistics.
            **kwargs: Additional keyword arguments passed to the Aer simulator.
                
        Returns:
            Dict[str, Any]: Dictionary containing simulation results including
                raw counts, execution time, and metadata.
                
        Raises:
            ValueError: If no solver is set.
            RuntimeError: If simulation fails.
            
        Example:
            .. code-block:: python

                puzzle.set_solver(ExactCoverQuantumSolver)

                # Quick simulation for testing
                test_result = puzzle.run_aer(shots=256)

                # High-precision simulation
                precise_result = puzzle.run_aer(shots=8192)

                print(f"Test: {len(test_result['counts'])} outcomes")
                print(f"Precise: {len(precise_result['counts'])} outcomes")
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        return self._solver.run_aer(shots, validation_context=self._validation_context, **kwargs)

    def format_result(self, result):
        """Decode and format a backend result using the active solver.

        Returns a dict with solutions, success rate, and optional mitigated rate.
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        return self._solver.format_result(result)
    
    def counts_plot(self, counts=None, backend_alias=None, shots=None, top_n=20, 
                    show_valid_only=False, figsize=(12, 6), show_summary=True):
        """
        Create a bar plot of measurement counts with Sudoku-specific enhancements.
        
        This method provides an easy interface to visualize quantum execution results
        with automatic validation highlighting and summary statistics.
        
        Args:
            counts: Dictionary of measurement outcomes, pytket Result object, or None
            backend_alias: Name of the backend used (for title display)
            shots: Total number of shots (for title and percentage calculation)
            top_n: Show only the top N most frequent outcomes (default: 20)
            show_valid_only: If True, only show outcomes that represent valid Sudoku solutions
            figsize: Figure size tuple (width, height)
            show_summary: If True, print summary statistics (default: True)
            
        Examples:
            # Plot results from Aer simulation
            result = puzzle.run_aer(shots=1024)
            puzzle.counts_plot(result, backend_alias="Aer", shots=1024)
            
            # Plot only valid solutions from hardware run
            result = puzzle.run("ibm_brisbane", opt_level=1, shots=100)
            puzzle.counts_plot(result, backend_alias="IBM Brisbane", show_valid_only=True)
            
            # Plot with custom counts dictionary
            puzzle.counts_plot(my_counts, backend_alias="Custom", shots=500, top_n=10)
            
            # Plot without summary statistics
            puzzle.counts_plot(result, backend_alias="Aer", show_summary=False)
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
            
        # Delegate to the solver's counts_plot method with QSudoku-specific defaults
        if backend_alias is None:
            backend_alias = "Unknown Backend"
            
        return self._solver.counts_plot(
            counts=counts,
            backend_alias=backend_alias, 
            shots=shots,
            top_n=top_n,
            show_valid_only=show_valid_only,
            figsize=figsize,
            show_summary=show_summary
        )
    
    def report_resources(self):
        """Return structured summary of all recorded resource data.

        The summary queries the stage-aware metadata system and includes puzzle
        info plus per-solver, per-encoding circuit metrics and backend-specific
        transpilation results. Depth values may be ``None`` when a solver's
        analytical estimation does not compute it (e.g. exact-cover estimation).
        
        Structure::

            {
                "puzzle_info": {
                    "hash": str,
                    "size": int,
                    "num_missing_cells": int
                },
                "solvers": {
                    <solver_name>: {
                        <encoding>: {
                            "main_circuit": {
                                "n_qubits": int,
                                "n_gates": int,
                                "n_mcx_gates": int,  # present only if computed
                                "depth": Optional[int]  # may be None if not estimated
                            },
                            "backends": {
                                <backend_alias>: {
                                    <opt_level>: {"n_qubits": int, "n_gates": int, "depth": int} | {"error": str}
                                }
                            }
                        }
                    }
                }
            }

        Returns:
            Dict[str, Any]: Nested dictionary as described above.

        Example:
            .. code-block:: python

                resources = puzzle.report_resources()
                print(resources["puzzle_info"])               # Basic puzzle metadata
                print(resources["solvers"].keys())            # Solver names recorded
                # Access main circuit qubits of exact cover simple encoding
                mc = resources["solvers"]["ExactCoverQuantumSolver"]["simple"]["main_circuit"]
                print(mc.get("n_qubits"), mc.get("depth"))   # Depth may be None
        """
        return self._get_resource_summary_from_stages()

    def get_hash(self) -> str:
        """Get the unique hash identifier for the Sudoku puzzle.
        
        Returns the cryptographic hash of the puzzle configuration, which
        serves as a unique identifier for caching and metadata management.
        
        Returns:
            str: Hexadecimal hash string uniquely identifying the puzzle
                configuration.
        Example:
            .. code-block:: python

                puzzle = QSudoku.generate(subgrid_size=3, num_missing_cells=20)
                puzzle_hash = puzzle.get_hash()
        """
        return self.puzzle.get_hash()
    
    def calculate_metrics(self, run_id = None):
        """Calculate or retrieve benchmarking metrics for a quantum execution.
        
        Provides programmatic access to comprehensive benchmarking metrics including
        success probability, ranking metrics, efficiency metrics, and more. Can
        either compute metrics for a fresh result or load previously computed
        metrics from Stage 6-7 JSON files.
        
        Args:
            run_id: Optional UUID of a specific execution run. If provided, loads
                metrics from Stage 6-7 JSON files. If None, returns metrics from
                the most recent run (if Stage 6-7 recording was enabled).
        
        Returns:
            MetricsResult dataclass or dict containing all computed metrics, or None if
            no metrics are available (e.g., validation context wasn't set).
        
        Raises:
            ValueError: If new metadata architecture is not enabled, or if
                requested run_id doesn't exist.
        
        Example:
            >>> from sudoku_nisq import QSudoku
            >>> from sudoku_nisq.solvers import ExactCoverQuantumSolver
            >>> 
            >>> # Generate puzzle and set up solver
            >>> puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=2)
            >>> puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
            >>> puzzle.build_circuit()
            >>> 
            >>> # Enable automatic metrics with validation context
            >>> context = puzzle.puzzle.create_validation_context('simple')
            >>> puzzle.set_validation_context(context.valid_solutions)
            >>> 
            >>> # Run and compute metrics
            >>> result = puzzle.run_aer(shots=1024)
            >>> metrics = puzzle.calculate_metrics()
            >>> 
            >>> # Access metrics
            >>> print(f"Success probability: {metrics['p_succ']:.4f}")
            >>> print(f"Valid odds: {metrics.get('valid_odds', 'N/A')}")
        
        Note:
            Requires `SUDOKU_NISQ_NEW_METADATA=1` environment variable or
            `MetadataConfig.ENABLE_NEW_ARCHITECTURE=True` to enable Stage 6-7
            recording. Also requires a validation context to be set before execution.
        
        See Also:
            - set_validation_context(): Enable automatic metrics recording
            - create_validation_context(): Generate validation context from enumeration
        """
        from sudoku_nisq.metadata.metrics import MetricsMetadataManager
        from sudoku_nisq.metadata.config import MetadataConfig
        
        if not MetadataConfig.ENABLE_NEW_ARCHITECTURE:
            raise ValueError(
                "Metrics calculation requires new metadata architecture. "
                "Set SUDOKU_NISQ_NEW_METADATA=1 or MetadataConfig.ENABLE_NEW_ARCHITECTURE=True"
            )
        
        # Initialize metrics manager
        puzzle_hash = self.get_hash()
        metrics_manager = MetricsMetadataManager(
            cache_base=".quantum_solver_cache",
            puzzle_hash=puzzle_hash
        )
        
        # Query metrics
        if run_id is not None:
            # Load specific run
            all_metrics = metrics_manager.query()
            if run_id not in all_metrics:
                raise ValueError(f"No metrics found for run_id: {run_id}")
            return all_metrics[run_id]
        else:
            # Get most recent run
            all_metrics = metrics_manager.query()
            if not all_metrics:
                return None  # No metrics available
            
            # Find most recent non-aggregated entry
            from datetime import datetime
            recent_run_id = None
            recent_timestamp = None
            
            for rid, data in all_metrics.items():
                if rid.startswith("aggregated_"):
                    continue  # Skip aggregated entries
                
                # Try to extract timestamp from data
                timestamp_str = data.get("timestamp")
                if timestamp_str:
                    try:
                        ts = datetime.fromisoformat(timestamp_str)
                        if recent_timestamp is None or ts > recent_timestamp:
                            recent_timestamp = ts
                            recent_run_id = rid
                    except (ValueError, TypeError):
                        pass
            
            if recent_run_id is None:
                return None
            
            return all_metrics[recent_run_id]    
    def _get_resource_summary_from_stages(self) -> Dict[str, Any]:
        """Query stage managers to build resource summary.
        
        Returns:
            dict: Resource summary in legacy format for backward compatibility
        """
        from sudoku_nisq.metadata import LogicalIRMetadataManager, CompilationMetadataManager
        
        puzzle_hash = self.puzzle.get_hash()
        
        # Build puzzle info
        puzzle_size = getattr(self.puzzle, 'size', None) or getattr(self.puzzle, 'board_size', 0)
        summary = {
            "puzzle_info": {
                "hash": puzzle_hash,
                "size": puzzle_size,
                "num_missing_cells": len(self.puzzle.open_tuples)
            },
            "solvers": {}
        }
        
        # Query Stage 2a for logical IR data
        try:
            stage2a = LogicalIRMetadataManager(
                cache_base=self.cache_base,
                puzzle_hash=puzzle_hash
            )
            
            ir_records = stage2a.query()
            
            # Handle both list and None return types
            if ir_records is None:
                ir_records = []
            elif not isinstance(ir_records, list):
                ir_records = [ir_records]
            
            # Group by solver/encoding
            for record in ir_records:
                solver_name = record.get('solver_name')
                encoding = record.get('encoding')
                resources = record.get('resources', {})
                
                if isinstance(solver_name, str) and solver_name:
                    solvers_dict = summary.get("solvers")
                    if not isinstance(solvers_dict, dict):
                        solvers_dict = {}
                        summary["solvers"] = solvers_dict
                    
                    if solver_name not in solvers_dict:
                        solvers_dict[solver_name] = {}
                    
                    if isinstance(encoding, str) and encoding:
                        solver_entry = solvers_dict.get(solver_name)
                        if isinstance(solver_entry, dict) and encoding not in solver_entry:
                            solver_entry[encoding] = {
                                "main_circuit": resources,
                                "backends": {}
                            }
        except Exception:
            pass
        
        # Query Stage 3 for compilation data
        try:
            stage3 = CompilationMetadataManager(
                cache_base=self.cache_base,
                puzzle_hash=puzzle_hash
            )
            
            comp_records_result = stage3.query()
            
            # Handle both list and None return types
            comp_records = []
            if comp_records_result is None:
                comp_records = []
            elif not isinstance(comp_records_result, list):
                comp_records = [comp_records_result]
            else:
                comp_records = comp_records_result
            
            # Match compilations to solvers via circuit_hash
            circuit_hash_to_solver = {}
            if ir_records:
                for record in ir_records:
                    circuit_hash = record.get('circuit_hash')
                    solver_name = record.get('solver_name')
                    encoding = record.get('encoding')
                    if circuit_hash and solver_name and encoding:
                        circuit_hash_to_solver[circuit_hash] = (solver_name, encoding)
            
            for comp in comp_records:
                circuit_hash = comp.get('circuit_hash')
                if circuit_hash and circuit_hash in circuit_hash_to_solver:
                    solver_name, encoding = circuit_hash_to_solver[circuit_hash]
                    backend_alias = comp.get('backend_alias')
                    opt_level = comp.get('opt_level')
                    resources = comp.get('resources', {})
                    
                    # Type-safe nested dict access
                    if (isinstance(solver_name, str) and solver_name and 
                        isinstance(encoding, str) and encoding):
                        solvers_dict = summary.get("solvers")
                        if not isinstance(solvers_dict, dict):
                            continue
                        
                        solver_entry = solvers_dict.get(solver_name)
                        if not isinstance(solver_entry, dict):
                            continue
                        
                        encoding_entry = solver_entry.get(encoding)
                        if not isinstance(encoding_entry, dict):
                            continue
                        
                        backends_entry = encoding_entry.get("backends")
                        if not isinstance(backends_entry, dict):
                            continue
                        
                        if isinstance(backend_alias, str) and backend_alias:
                            if backend_alias not in backends_entry:
                                backends_entry[backend_alias] = {}
                            if isinstance(opt_level, int):
                                backends_entry[backend_alias][str(opt_level)] = resources
        except Exception:
            pass
        
        return summary