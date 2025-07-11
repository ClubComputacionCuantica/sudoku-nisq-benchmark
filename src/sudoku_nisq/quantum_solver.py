from abc import ABC, abstractmethod
from pytket.passes import FlattenRegisters
from pytket.circuit.display import render_circuit_jupyter as draw
from pytket.extensions.qiskit import IBMQBackend, AerBackend
import os
import json
import glob
import logging
from pytket.circuit import Circuit
import matplotlib.pyplot as plt

# Set up logger
log = logging.getLogger(__name__)
# Only configure logging if no handlers exist (prevents duplicate logs)
if not logging.getLogger().hasHandlers():
    # Default logging config for production: INFO to stdout
    logging.basicConfig(
        level=os.environ.get("QUANTUM_SOLVER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # To enable persistent logging, set the environment variable QUANTUM_SOLVER_LOG_FILE to a file path
    log_file = os.environ.get("QUANTUM_SOLVER_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)

"""
Quantum Solver Framework for Sudoku NISQ Benchmarking

This module provides a unified interface for quantum circuit execution across different
backend providers, with specific focus on Sudoku solving algorithms. The framework
supports multiple quantum backends (IBM Quantum, Aer simulator) and includes
caching, resource analysis, and visualization capabilities.

Key Features:
    - Multi-backend support (IBM Quantum, Aer, extensible to others)
    - Intelligent caching for circuits, compilations, and execution results
    - Resource analysis and optimization level comparison
    - Puzzle-specific result tracking and visualization
    - Persistent disk-based caching with configurable storage locations

Environment Variables:
    QUANTUM_SOLVER_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
    QUANTUM_SOLVER_LOG_FILE: Path for persistent log file (optional)
    QUANTUM_SOLVER_CACHE_DIR: Base directory for cache storage (optional)

Usage Example:
    # Initialize solver with Sudoku puzzle
    solver = ExactCoverQuantumSolver(sudoku=my_puzzle)
    
    # Configure IBM backend
    available_devices = solver.init_ibm(token, instance)
    solver.add_backend("ibm_brisbane", "ibm_brisbane")
    solver.set_backend("ibm_brisbane")
    
    # Analyze resource requirements
    resources = solver.find_transpiled_resources([0, 1, 2, 3])
    
    # Execute and visualize results
    result = solver.run(shots=1024, optimisation_level=1)
    solver.counts_plot()

Inheritance Hierarchy:
    QuantumSolver (ABC)
    ├── ExactCoverQuantumSolver
    ├── GraphColoringQuantumSolver
    └── BacktrackingQuantumSolver
"""

# Logging usage:
# - Set log level via QUANTUM_SOLVER_LOG_LEVEL (e.g., DEBUG, INFO, WARNING)
# - To log to a file, set QUANTUM_SOLVER_LOG_FILE to a writable file path
# - Example: export QUANTUM_SOLVER_LOG_LEVEL=DEBUG
#            export QUANTUM_SOLVER_LOG_FILE=solver.log
# - Use 'log.info()', 'log.warning()', etc. in code

class QuantumSolver(ABC):
    """
    Abstract base class for quantum solvers providing unified quantum circuit execution.

    This class serves as the foundation for quantum Sudoku solving algorithms, offering
    a standardized interface for quantum circuit management, backend integration, and
    result analysis. It handles the complexity of quantum device interaction while
    providing a clean API for algorithm implementation.

    Core Functionality:
        - Backend Management: Register, configure, and switch between quantum backends
        - Circuit Compilation: Automatic compilation with optimization level support
        - Intelligent Caching: Multi-layer caching for circuits, compilations, and results
        - Resource Analysis: Detailed quantum resource usage analysis and comparison
        - Result Processing: Execution result collection, filtering, and visualization
        - Persistence: Disk-based caching for reproducible research and debugging

    Cache Architecture:
        The solver implements a caching system with multiple layers:
        
        1. In-Memory Caches:
           - Single main circuit (most recent)
           - Single compiled circuit (most recent) 
           - Execution results (all recent runs)
           - Transpilation metadata (resource analysis)
        
        2. Disk Caches:
           - Main circuits: .quantum_solver_cache/{solver_type}/{puzzle_hash}/main_circuit.json
           - Compiled circuits: .quantum_solver_cache/{backend}/{solver_type}/{puzzle_hash}/compiled_level{N}.json
           - Execution/transpilation data: .quantum_solver_cache/execution_cache.json, transpilation_cache.json
        
        Cache keys include puzzle hash and solver type for proper isolation between different
        problems and algorithms.

    Attributes:
        main_circuit (Circuit): Currently loaded main quantum circuit
        flattened (bool): Whether circuit registers have been flattened for backend compatibility
        backends (Dict[str, Any]): Registered backend instances mapped by name
        current_backend (str): Name of the currently active backend
        sudoku (object): Associated Sudoku puzzle instance (for puzzle-specific caching)

    Abstract Methods:
        build_circuit(): Must return a pytket Circuit implementing the solving algorithm
        resource_estimation(): Must return quantum resource requirements analysis

    Public Methods:
        Backend Management:
            init_ibm(token, instance): Configure IBM Quantum access and list devices
            init_quantinuum(token): Configure Quantinuum access (placeholder)
            add_backend(name, backend_instance): Register a new backend
            set_backend(name): Select active backend
        
        Circuit Operations:
            get_main_circuit(force_rebuild): Get cached main circuit with intelligent caching
            draw_circuit(): Visualize the quantum circuit
            flatten_registers(): Prepare circuit for generic backends
            find_transpiled_resources(levels, force): Analyze resources at optimization levels
        
        Execution:
            run(shots, force_run, opt_level): Execute on selected backend
            run_aer(shots, force_run): Execute on Aer simulator
        
        Results & Analysis:
            get_execution_results(backend, shots, opt_level): Retrieve cached results
            get_latest_counts(backend): Get most recent execution counts
            counts_plot(counts, shots, opt_level, backend): Plot measurement results
        
        Cache Management:
            clear_cache(cache_type): Clear specified cache types
            get_cache_stats(): Get cache usage statistics

    Raises:
        ValueError: For invalid parameters or missing required arguments
        KeyError: When accessing unregistered backends
        RuntimeError: For backend compatibility or circuit compilation issues

    Example:
        >>> # Subclass implementation
        >>> class MySolver(QuantumSolver):
        ...     def build_circuit(self):
        ...         # Return pytket Circuit for your algorithm
        ...         pass
        ...     def resource_estimation(self):
        ...         # Return resource analysis
        ...         pass
        >>> 
        >>> # Usage
        >>> solver = MySolver(sudoku=puzzle)
        >>> solver.init_ibm(token, instance)
        >>> solver.add_backend("device", "ibm_brisbane")
        >>> solver.set_backend("device")
        >>> result = solver.run(shots=1024)

    Note:
        This class is designed for research and benchmarking purposes, with emphasis
        on reproducibility, performance analysis, and ease of use across different
        quantum platforms.
    """
    def __init__(self, backends=None, sudoku=None):
        """
        Initialize a new QuantumSolver instance with optional pre-configured backends and Sudoku puzzle.

        Sets up the internal cache system, backend management, and puzzle-specific identification
        for proper result isolation. Loads any existing cache data from disk.

        Args:
            backends (dict, optional): Pre-configured backend instances mapped by name.
                Example: {"my_device": IBMQBackend("ibm_brisbane")}
                Defaults to empty dict.
            sudoku (object, optional): Sudoku puzzle instance with get_hash() method for
                puzzle-specific caching and result tracking. If None, uses generic
                identifier 'no_puzzle'. Defaults to None.

        Attributes Initialized:
            main_circuit (Circuit): Main quantum circuit (None until built)
            flattened (bool): Circuit register flattening status
            backends (dict): Available backend instances
            current_backend (str): Active backend name (None until set)
            sudoku (object): Associated puzzle instance for identification
            
        Cache System:
            Initializes multi-layer caching with:
            - In-memory caches for recent circuits and results
            - Disk-based persistence for reproducibility
            - Puzzle and solver-specific cache keys for isolation

        Note:
            Cache data is automatically loaded from disk if available, allowing
            resumption of work across sessions without recompilation or re-execution.
        """
        log.debug(f"Initializing QuantumSolver with {len(backends) if backends else 0} backends")
        
        # Circuit management - only keep one circuit in memory at a time
        self.main_circuit = None
        self._main_circuit_cache_key = None  # Track what's in memory
        self.flattened = False  # tracks if registers have been flattened

        # Backend management
        self.backends = backends.copy() if backends else {}
        self.current_backend = None
        
        # Sudoku integration for puzzle-specific caching
        self.sudoku = sudoku
        
        # Execution caches - use consistent cache keys: (backend, shots, opt_level, puzzle_hash, solver_type)
        self._execution_cache = {
            'counts': {},    # cache_key -> counts
            'handles': {},   # cache_key -> handle  
            'metadata': {}   # cache_key -> metadata
        }
        
        # Transpilation cache - separate from execution: (backend, opt_level, puzzle_hash, solver_type) -> metadata
        self._transpilation_cache = {}
        
        # Single compiled circuit cache - only keep most recent in memory
        self._compiled_circuit = None
        self._compiled_circuit_key = None

        # Load caches from disk at startup
        self._load_caches_from_disk()
        
        log.info("QuantumSolver initialized successfully")

    @abstractmethod
    def build_circuit(self):
        """
        Construct and return the quantum circuit implementing the solving algorithm.

        This method must be implemented by subclasses to build the quantum circuit
        that encodes their specific Sudoku solving approach (exact cover, graph coloring,
        backtracking, etc.).

        **Note**: This is an abstract method for implementation only. For accessing circuits
        with caching, use `get_main_circuit()` instead, which provides caching
        and disk persistence.

        The returned circuit should be a complete, executable quantum circuit that:
        - Encodes the Sudoku puzzle constraints
        - Implements the solving algorithm logic
        - Includes proper measurement operations
        - Is ready for compilation and execution

        Returns:
            Circuit: A pytket quantum circuit object representing the complete
                solving algorithm. The circuit should be gate-level and ready
                for backend compilation.

        Implementation Notes:
            - Use self.sudoku to access the puzzle instance if available
            - Consider qubit count limitations of target backends
            - Include measurement operations for result extraction
            - Document any algorithm-specific parameters or assumptions
            - This method is called internally by `get_main_circuit()` for caching

        Example:
            >>> def build_circuit(self):
            ...     circuit = Circuit(self.n_qubits)
            ...     # Add quantum gates for your algorithm
            ...     circuit.add_gate(...)
            ...     # Add measurements
            ...     circuit.measure_all()
            ...     return circuit
        """
        pass
    
    @abstractmethod
    def resource_estimation(self):
        """
        Analyze and return quantum resource requirements for the solving algorithm.

        This method must be implemented by subclasses to provide detailed analysis
        of the quantum resources needed for their specific algorithm implementation.
        This information is crucial for algorithm comparison and backend selection.

        The analysis should consider:
        - Qubit count requirements (logical qubits before compilation)
        - Gate count estimates (by gate type if relevant)
        - Circuit depth estimates
        - Memory requirements for classical processing
        - Scalability characteristics with puzzle size

        Returns:
            dict: Resource analysis with standardized keys:
                - 'qubits': Number of qubits required
                - 'gates': Total gate count estimate
                - 'depth': Circuit depth estimate (optional)

        Implementation Notes:
            - Use self.sudoku to analyze puzzle-specific requirements
            - Consider both best-case and worst-case scenarios
            - Document any approximations or assumptions made
            - Include scalability analysis for different puzzle sizes

        Example:
            >>> def resource_estimation(self):
            ...     n_cells = 81  # Standard 9x9 Sudoku
            ...     return {
            ...         'qubits': n_cells * 4,  # 4 qubits per cell
            ...         'gates': n_cells * 100,  # Rough estimate
            ...         'depth': 50,
            ...         'scalability': 'O(n²) for n×n puzzles'
            ...     }
        """
        pass
    
    def get_main_circuit(self, force_rebuild: bool = False):
        """
        Get the main quantum circuit with intelligent caching.

        This is the preferred method for accessing the quantum circuit, as it handles
        caching automatically to avoid redundant circuit construction. The circuit
        is cached both in memory and on disk for performance and reproducibility.

        Args:
            force_rebuild (bool, optional): If True, forces circuit reconstruction
                and ignores any cached versions. Useful when algorithm parameters
                or puzzle constraints have changed. Defaults to False.

        Returns:
            Circuit: The main quantum circuit implementing the solving algorithm.
                The circuit is ready for compilation, execution, or analysis.

        Note:
            - This method uses the caching system with puzzle and solver-specific keys
            - Cached circuits are automatically loaded from disk across sessions
            - Use force_rebuild=True when circuit logic has been modified
            - The returned circuit may need register flattening for some backends

        Example:
            >>> # Get cached circuit (builds if not cached)
            >>> circuit = solver.get_main_circuit()
            >>> print(f"Circuit has {circuit.n_qubits} qubits")
            
            >>> # Force rebuild if algorithm changed
            >>> circuit = solver.get_main_circuit(force_rebuild=True)
        """
        return self._get_or_build_main_circuit(force=force_rebuild)

    def draw_circuit(self):
        """
        Visualize the quantum circuit using pytket's circuit drawing functionality.

        The circuit will be displayed in a Jupyter notebook environment. Requires that `main_circuit` is constructed.
        """
        circuit = self._get_or_build_main_circuit()
        log.debug("Drawing quantum circuit")
        draw(circuit)

    def flatten_registers(self):
        """
        Flatten the circuit's registers for compatibility with generic backends.

        Returns:
            Circuit: The flattened circuit.
        """
        circuit = self._get_or_build_main_circuit()
        if not self.flattened:
            log.debug("Flattening circuit registers for backend compatibility")
            FlattenRegisters().apply(circuit)
            self.flattened = True
            log.debug("Circuit registers flattened successfully")
        return circuit

    def add_backend(self, name: str = "ibm_brisbane", backend_instance: str = "ibm_brisbane", **kwargs):
        """
        Register a quantum backend for circuit execution.

        Creates and registers a new backend instance that can be used for quantum
        circuit execution. Currently supports IBM Quantum backends through pytket-qiskit.

        Args:
            name (str, optional): Alias name for the backend. Used to reference this
                backend in other method calls. Defaults to "ibm_brisbane".
            backend_instance (str, optional): IBM Quantum backend device name as
                recognized by IBM's service (e.g., "ibm_brisbane", "ibm_kyoto").
                Must match an available device from your IBM account. 
                Defaults to "ibm_brisbane".
            **kwargs: Additional backend-specific configuration parameters.
                Currently unused but reserved for future extensions.

        Raises:
            Exception: If backend creation fails due to invalid device name,
                authentication issues, or device unavailability.

        Note:
            - Requires prior authentication via init_ibm()
            - Backend availability depends on your IBM Quantum account access
            - Use init_ibm() first to see available devices for your account
            - Multiple backends can be registered with different aliases

        Example:
            >>> solver.init_ibm(token, instance)  # Authenticate first
            >>> solver.add_backend("main_device", "ibm_brisbane")
            >>> solver.add_backend("backup_device", "ibm_kyoto")
            >>> solver.set_backend("main_device")  # Select for use
        """
        log.debug(f"Attempting to register backend '{name}' with instance '{backend_instance}'")
        try:
            self.backends[name] = IBMQBackend(backend_instance)
            log.info(f"Successfully registered backend '{name}' with instance '{backend_instance}'")
        except Exception as e:
            log.exception(f"Failed to register backend '{name}': {e}")
            raise

    def set_backend(self, name: str):
        """
        Select the active backend by name.

        Args:
            name (str): Name of the registered backend to select.

        Raises:
            KeyError: If the backend is not registered.
        """
        log.debug(f"Setting backend to '{name}'")
        if name not in self.backends:
            log.error(f"Backend '{name}' not found in registered backends: {list(self.backends.keys())}")
            raise KeyError(f"Backend '{name}' not registered")
        self.current_backend = name
        log.info(f"Active backend set to '{name}'")

    def init_ibm(self, token: str, instance: str):
        """
        Configure IBM Quantum credentials for pytket-qiskit and list available devices.

        Args:
            token (str): IBM Quantum API token.
            instance (str): IBM Cloud instance CRN or service name.

        Returns:
            list: List of available IBM backend device names.

        Reference:
            pytket-qiskit config: https://docs.quantinuum.com/tket/extensions/pytket-qiskit/
        """
        from pytket.extensions.qiskit import set_ibmq_config, IBMQBackend
        self.instance = instance
        if not instance:
            raise ValueError("'instance' is required for IBM Quantum hardware access and transpilation.")

        log.info("Authenticating with IBM Quantum service")
        # Set credentials for pytket-qiskit
        set_ibmq_config(ibmq_api_token=token, instance=instance)

        # List available devices
        devices = IBMQBackend.available_devices(instance=instance)
        log.info("IBM authentication successful")
        log.info(f"Found {len(devices)} IBM devices available to your account")
        log.debug(f"Available devices: {[dev.device_name for dev in devices]}")
        
        return [dev.device_name for dev in devices]

    def init_quantinuum(self, token: str):
        """
        Initialize connection to Quantinuum quantum computers (placeholder).

        Args:
            token (str): Authentication token for Quantinuum service.
        """
        log.info("Quantinuum initialization called (placeholder implementation)")
        pass

    def _get_backend(self, backend_name: str):
        """
        Retrieve a backend instance by name, ensuring it is registered.

        Args:
            backend_name (str): The name of the backend to retrieve.

        Returns:
            object: The backend instance.

        Raises:
            ValueError: If the backend is not registered.
        """
        if backend_name not in self.backends:
            log.error(f"Backend '{backend_name}' not found in registered backends: {list(self.backends.keys())}")
            raise ValueError(f"Backend '{backend_name}' not registered.")
        log.debug(f"Retrieved backend '{backend_name}'")
        return self.backends[backend_name]

    def _ensure_resource_metadata(self, backend_name: str, optimisation_level: int, compiled):
        """
        Compute and cache resource metadata for a compiled circuit, including qubit count, gate count, and depth.
        Now includes puzzle hash and solver type for proper identification.

        Args:
            backend_name (str): Identifier for the backend.
            optimisation_level (int): Optimization level used during compilation.
            compiled (Circuit): The compiled quantum circuit object.

        Returns:
            dict: Metadata dictionary for the compiled circuit.
        """
        # Check if metadata is already cached
        cached_metadata = self._get_cached_transpilation_metadata(backend_name, optimisation_level)
        if cached_metadata:
            return cached_metadata
        
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()
        
        # Compute metadata
        metadata = {
            "backend": backend_name,
            "optimisation_level": optimisation_level,
            "solver_type": solver_type,
            "puzzle_hash": puzzle_hash,
            "n_qubits": compiled.n_qubits,
            "n_gates": compiled.n_gates,
            "depth": compiled.depth()
        }
        
        # Cache the metadata
        self._cache_transpilation_metadata(backend_name, optimisation_level, metadata)
        
        return metadata

    def _get_cache_base_dir(self):
        """
        Returns the base directory for cache files used to persist compiled quantum circuits and related data.

        The location can be configured by setting the environment variable 'QUANTUM_SOLVER_CACHE_DIR'.
        If this variable is set, its value will be used as the cache directory.
        If not set, the default is a hidden directory named '.quantum_solver_cache' in the project root (the parent of 'src/sudoku_nisq').

        This allows you to control where persistent cache files are stored, which can be useful for sharing, debugging, or using different storage locations in development and production environments.
        """
        cache_dir = os.environ.get("QUANTUM_SOLVER_CACHE_DIR")
        if cache_dir:
            return cache_dir
        # Default: project root (parent of src/sudoku_nisq)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".quantum_solver_cache"))

    def _compiled_circuit_cache_path(self, backend_name: str, level: int, create_dirs: bool = False) -> str:
        """
        Returns a stable cache path including puzzle hash and solver type for proper circuit identification.

        Args:
            backend_name (str): Name of the backend.
            level (int): Optimization level.
            create_dirs (bool): Whether to create the directory structure. Defaults to False.

        Returns:
            str: Path to the cache file for the compiled circuit.
        """
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()
        
        # Validate that we have valid non-empty identifiers
        if not puzzle_hash or not puzzle_hash.strip():
            log.warning("Empty puzzle hash detected, using 'no_puzzle'")
            puzzle_hash = 'no_puzzle'
        if not solver_type or not solver_type.strip():
            log.warning("Empty solver type detected, using 'QuantumSolver'")
            solver_type = 'QuantumSolver'
        if not backend_name or not backend_name.strip():
            log.warning("Empty backend name detected, using 'unknown_backend'")
            backend_name = 'unknown_backend'
        
        # Sanitize path components (remove any problematic characters)
        puzzle_hash = "".join(c for c in puzzle_hash if c.isalnum() or c in "_-")
        solver_type = "".join(c for c in solver_type if c.isalnum() or c in "_-")
        backend_name = "".join(c for c in backend_name if c.isalnum() or c in "_-")
        
        # Use configurable cache base dir with puzzle and solver identification
        cache_dir = os.path.join(
            self._get_cache_base_dir(), 
            backend_name, 
            solver_type,
            puzzle_hash
        )
        
        # Only create directories when explicitly requested (e.g., when saving)
        if create_dirs:
            os.makedirs(cache_dir, exist_ok=True)

        # File name based on optimization level
        fname = f"compiled_level{level}.json"
        return os.path.join(cache_dir, fname)

    def _main_circuit_cache_path(self, create_dirs: bool = False) -> str:
        """
        Returns a stable cache path for the main circuit including puzzle hash and solver type.

        Args:
            create_dirs (bool): Whether to create the directory structure. Defaults to False.

        Returns:
            str: Path to the cache file for the main circuit.
        """
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()
        
        # Validate that we have valid non-empty identifiers
        if not puzzle_hash or not puzzle_hash.strip():
            log.warning("Empty puzzle hash detected, using 'no_puzzle'")
            puzzle_hash = 'no_puzzle'
        if not solver_type or not solver_type.strip():
            log.warning("Empty solver type detected, using 'QuantumSolver'")
            solver_type = 'QuantumSolver'
        
        # Sanitize path components (remove any problematic characters)
        puzzle_hash = "".join(c for c in puzzle_hash if c.isalnum() or c in "_-")
        solver_type = "".join(c for c in solver_type if c.isalnum() or c in "_-")
        
        # Use configurable cache base dir with puzzle and solver identification
        cache_dir = os.path.join(
            self._get_cache_base_dir(), 
            solver_type,
            puzzle_hash
        )
        
        # Only create directories when explicitly requested (e.g., when saving)
        if create_dirs:
            log.debug(f"Creating cache directories: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
        else:
            log.debug(f"Cache directory path (not creating): {cache_dir}")

        # File name for main circuit
        fname = "main_circuit.json"
        full_path = os.path.join(cache_dir, fname)
        log.debug(f"Main circuit cache path: {full_path}")
        return full_path

    def _get_or_compile(self, level: int, force: bool = False):
        """
        Retrieve a cached compiled circuit or compile a new one if needed.
        Uses a single in-memory cache slot and disk cache to avoid recompilation.
        Only the most recently used circuit is kept in memory.

        Args:
            level (int): The optimization level for compilation.
            force (bool): If True, force recompilation, ignoring any existing caches.

        Returns:
            Circuit: The compiled (or cached) circuit.
        """
        if self.current_backend is None:
            raise RuntimeError("No backend selected. Call set_backend() first.")

        backend_name = self.current_backend
        backend = self._get_backend(backend_name)
        
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()

        # Create cache key for this compilation request
        cache_key = (backend_name, solver_type, puzzle_hash, level)

        # Generate disk cache path
        cache_path = self._compiled_circuit_cache_path(backend_name, level)

        if not force:
            # Try in-memory cache first (single circuit)
            if self._compiled_circuit_key == cache_key and self._compiled_circuit is not None:
                log.debug(f"In-memory cache hit for {cache_key}")
                return self._compiled_circuit
            
            # Try disk cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        circ_dict = json.load(f)
                    compiled = Circuit.from_dict(circ_dict)
                    # Store in single in-memory slot
                    self._compiled_circuit = compiled
                    self._compiled_circuit_key = cache_key
                    log.info(f"Loaded compiled circuit from disk cache: {cache_path}")
                    return compiled
                except (json.JSONDecodeError, IOError) as e:
                    log.warning(f"Could not load from cache file {cache_path} due to {e}. Recompiling.")

        # Get main circuit (from cache or build fresh)
        main_circuit = self._get_or_build_main_circuit()
        if not self.flattened:
            self.flatten_registers()
        
        # Compile and cache (replaces any previous in-memory circuit)
        log.info(f"Compiling circuit for {solver_type}/{puzzle_hash} on {backend_name} at level {level}...")
        compiled = backend.get_compiled_circuit(main_circuit, optimisation_level=level)
        
        # Store in single in-memory slot (replaces previous)
        self._compiled_circuit = compiled
        self._compiled_circuit_key = cache_key
        
        # Save to disk
        try:
            # Create directories when saving
            save_path = self._compiled_circuit_cache_path(backend_name, level, create_dirs=True)
            with open(save_path, "w") as f:
                json.dump(compiled.to_dict(), f)
            log.info(f"Saved compiled circuit to disk cache: {save_path}")
        except IOError as e:
            log.error(f"Failed to save compiled circuit to {save_path}: {e}")
        return compiled

    def _get_or_build_main_circuit(self, force: bool = False):
        """
        Retrieve cached main circuit or build a new one if needed.
        Uses a single in-memory cache slot and disk cache to avoid rebuilding.
        Only the most recently used circuit is kept in memory.

        Args:
            force (bool): If True, force rebuilding, ignoring any existing caches.

        Returns:
            Circuit: The main circuit (cached or newly built).
        """
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()

        # Create cache key for this circuit request
        cache_key = (solver_type, puzzle_hash)
        log.debug(f"Main circuit cache key: {cache_key}")

        # Generate disk cache path
        cache_path = self._main_circuit_cache_path()
        log.debug(f"Checking for cached circuit at: {cache_path}")

        if not force:
            # Try in-memory cache first
            if self._main_circuit_cache_key == cache_key and self.main_circuit is not None:
                log.debug(f"In-memory cache hit for main circuit {cache_key}")
                return self.main_circuit
            else:
                log.debug(f"In-memory cache miss: stored key={self._main_circuit_cache_key}, circuit exists={self.main_circuit is not None}")
            
            # Try disk cache
            if os.path.exists(cache_path):
                log.debug(f"Found disk cache file: {cache_path}")
                try:
                    with open(cache_path, "r") as f:
                        circ_dict = json.load(f)
                    circuit = Circuit.from_dict(circ_dict)
                    # Store in single in-memory slot
                    self.main_circuit = circuit
                    self._main_circuit_cache_key = cache_key
                    self.flattened = False  # Reset flattened state from disk
                    log.info(f"Loaded main circuit from disk cache: {cache_path}")
                    return circuit
                except (json.JSONDecodeError, IOError) as e:
                    log.warning(f"Could not load main circuit from cache file {cache_path} due to {e}. Rebuilding.")
            else:
                log.debug(f"No disk cache found at: {cache_path}")

        # Build new circuit and cache (replaces any previous in-memory circuit)
        log.info(f"Building main circuit for {solver_type}/{puzzle_hash}...")
        circuit = self.build_circuit()  # Call the abstract method implemented by subclasses
        
        # Store in single in-memory slot (replaces previous)
        self.main_circuit = circuit
        self._main_circuit_cache_key = cache_key
        self.flattened = False  # Fresh circuit is not flattened
        
        # Save to disk
        try:
            # Create directories when saving
            save_path = self._main_circuit_cache_path(create_dirs=True)
            log.info(f"Saving main circuit to disk cache: {save_path}")
            
            # Verify the directory structure was created correctly
            cache_dir = os.path.dirname(save_path)
            if not os.path.exists(cache_dir):
                log.error(f"Cache directory was not created: {cache_dir}")
            else:
                log.debug(f"Cache directory exists: {cache_dir}")
            
            with open(save_path, "w") as f:
                json.dump(circuit.to_dict(), f)
            log.info(f"Successfully saved main circuit to disk cache: {save_path}")
        except IOError as e:
            log.error(f"Failed to save main circuit to {save_path}: {e}")
        except Exception as e:
            log.error(f"Unexpected error saving main circuit: {e}")
        
        return circuit

    def find_transpiled_resources(
        self,
        optimisation_levels = [0, 1, 2, 3],
        force_refresh = False
    ):
        """
        Compile and analyze quantum circuit resource usage across multiple optimization levels.

        This method provides comprehensive resource analysis by compiling the quantum circuit
        at different optimization levels and extracting key metrics. Results are cached for
        efficient repeated analysis and comparison studies.

        The analysis helps with:
        - Backend selection based on resource requirements
        - Optimization level comparison for performance tuning
        - Scalability analysis for research and benchmarking
        - Resource planning for quantum algorithm development

        Args:
            optimisation_levels (list, optional): Optimization levels to analyze.
                Valid values are integers 0-3, where:
                - 0: No optimization (baseline)
                - 1: Light optimization (default IBM setting)
                - 2: Medium optimization (balanced performance/compilation time)
                - 3: Heavy optimization (maximum gate reduction)
                Defaults to [0, 1, 2, 3] for complete analysis.
            force_refresh (bool, optional): If True, forces recompilation and ignores
                cached transpilation results. Useful when circuit or backend has changed.
                Defaults to False.

        Returns:
            list: List of resource metadata dictionaries, one per optimization level.
                Each dictionary contains:
                - 'backend': Backend name used for compilation
                - 'optimisation_level': Optimization level (0-3)
                - 'solver_type': Name of the solver class
                - 'puzzle_hash': Unique identifier for the puzzle instance
                - 'n_qubits': Number of qubits in compiled circuit
                - 'n_gates': Total gate count after compilation
                - 'depth': Circuit depth (critical path length)
                
                For compilation errors, entries contain:
                - 'backend': Backend name
                - 'optimisation_level': Failed optimization level
                - 'error': Error message description

        Raises:
            RuntimeError: If no backend is selected (call set_backend() first) or
                if the selected backend doesn't support transpilation.
            ValueError: If optimisation_levels contains invalid values (not in 0-3).

        Example:
            >>> # Basic usage - analyze all levels
            >>> resources = solver.find_transpiled_resources()
            >>> for r in resources:
            ...     print(f"Level {r['optimisation_level']}: {r['n_qubits']} qubits, {r['n_gates']} gates")
            
            >>> # Custom analysis - specific levels only
            >>> resources = solver.find_transpiled_resources([1, 2], force_refresh=True)
            
            >>> # Check for compilation errors
            >>> for r in resources:
            ...     if 'error' in r:
            ...         print(f"Compilation failed at level {r['optimisation_level']}: {r['error']}")

        Note:
            - Results are automatically cached for performance
            - Use force_refresh=True when circuit or backend configuration changes
            - Compilation may fail at higher optimization levels for complex circuits
            - Analysis is puzzle and solver-specific due to cache key design
        """
        if self.current_backend is None:
            raise RuntimeError("No backend selected. Call set_backend() first.")
        if not all(isinstance(l, int) and l in [0, 1, 2, 3] for l in optimisation_levels):
            raise ValueError("optimisation_levels must be a list of integers in {0, 1, 2, 3}.")

        backend_name = self.current_backend
        backend = self._get_backend(backend_name)

        # Ensure main circuit is available and flattened
        self._get_or_build_main_circuit()
        if not self.flattened:
            self.flatten_registers()

        if not hasattr(backend, "get_compiled_circuit"):
            raise RuntimeError(f"Backend '{type(backend).__name__}' does not support transpilation.")

        resources = []

        for level in optimisation_levels:
            if force_refresh:
                # Clear cached transpilation metadata
                cache_key = self._get_transpilation_cache_key(backend_name, level)
                self._transpilation_cache.pop(cache_key, None)
                log.debug(f"Forcing refresh for optimization level {level}")

            # Check if we have cached metadata
            cached_metadata = self._get_cached_transpilation_metadata(backend_name, level)
            if not force_refresh and cached_metadata:
                log.debug(f"Using cached resource data for level {level}")
                resources.append(cached_metadata)
                continue

            try:
                log.debug(f"Compiling circuit at optimization level {level}")
                circuit = self._get_or_compile(level, force=force_refresh)
                metadata = self._ensure_resource_metadata(backend_name, level, circuit)
                log.info(f"Level {level}: {metadata['n_qubits']} qubits, {metadata['n_gates']} gates, depth {metadata['depth']}")
                resources.append(metadata)
            except Exception as e:
                log.exception(f"Failed to compile at optimization level {level}: {e}")
                error_info = {
                    "backend": backend_name,
                    "optimisation_level": level,
                    "error": str(e)
                }
                # Do not cache error info, just log and append to results
                resources.append(error_info)

        return resources

    def run(
        self,
        shots = 1024,
        force_run = False,
        optimisation_level = 1
    ):
        """
        Execute the quantum circuit on the selected backend and return detailed results.

        This method handles the complete execution pipeline: circuit compilation at the
        specified optimization level, job submission to the quantum backend, result
        collection, and comprehensive metadata gathering. Results are automatically
        cached for performance and reproducibility.

        Args:
            shots (int, optional): Number of quantum measurements to perform.
                Higher values provide better statistical accuracy but increase
                execution time and cost. Typical values: 1024-8192. 
                Defaults to 1024.
            force_run (bool, optional): If True, bypasses cached results and forces
                fresh execution. Useful when testing consistency or after circuit
                modifications. Defaults to False.
            optimisation_level (int, optional): Circuit optimization level for compilation.
                Higher levels may reduce gate count but increase compilation time:
                - 0: No optimization (fastest compilation)
                - 1: Light optimization (recommended default)
                - 2: Medium optimization (balanced)
                - 3: Heavy optimization (maximum gate reduction)
                Defaults to 1.

        Returns:
            dict: Comprehensive execution results with keys:
                - 'counts': Dict mapping quantum states (bit strings) to measurement counts
                - 'handle': Backend-specific job handle for result tracking
                - 'metadata': Resource analysis dict containing:
                    - 'backend': Name of the backend used
                    - 'optimisation_level': Optimization level applied
                    - 'solver_type': Class name of the solver
                    - 'puzzle_hash': Unique puzzle identifier
                    - 'n_qubits': Number of qubits in compiled circuit
                    - 'n_gates': Total gate count after compilation
                    - 'depth': Circuit depth (critical path length)

        Raises:
            RuntimeError: If no backend is selected (call set_backend() first) or
                if the backend doesn't support required operations.
            ValueError: If shots <= 0, optimisation_level not in [0,1,2,3], or
                force_run is not boolean.

        Example:
            >>> # Basic execution with default parameters
            >>> result = solver.run()
            >>> counts = result['counts']
            >>> print(f"Most frequent state: {max(counts, key=counts.get)}")
            
            >>> # High-precision execution with custom optimization
            >>> result = solver.run(shots=8192, optimisation_level=2)
            >>> metadata = result['metadata']
            >>> print(f"Used {metadata['n_qubits']} qubits, {metadata['n_gates']} gates")
            
            >>> # Force fresh execution (ignore cache)
            >>> result = solver.run(shots=1024, force_run=True)

        Note:
            - Results are automatically cached based on backend, shots, optimization level,
              puzzle, and solver type for efficient repeated access
            - Use force_run=True when circuit or backend configuration has changed
            - Execution time varies significantly between simulators and real hardware
            - Real hardware execution may involve queue waiting times
        """
        # Ensure main circuit is available and flattened
        self._get_or_build_main_circuit()
        if not self.flattened:
            self.flatten_registers()
            
        if self.current_backend is None:
            raise RuntimeError("No backend selected. Call set_backend() first.")
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer.")
        if optimisation_level not in [0, 1, 2, 3]:
            raise ValueError("optimisation_level must be 0, 1, 2, or 3.")
        if not isinstance(force_run, bool):
            raise ValueError("force_run must be a boolean.")

        backend_name = self.current_backend
        backend = self._get_backend(backend_name)

        # Check for cached execution result
        cached_result = self._get_cached_execution_result(backend_name, shots, optimisation_level)
        if not force_run and cached_result:
            log.debug(f"Using cached results for {(backend_name, shots, optimisation_level)}")
            return cached_result

        if not hasattr(backend, "get_compiled_circuit") or \
           not hasattr(backend, "process_circuit") or \
           not hasattr(backend, "get_result"):
            raise RuntimeError(f"Backend '{type(backend).__name__}' does not support run().")

        log.info(f"Executing circuit on backend '{backend_name}' with {shots} shots (opt_level={optimisation_level})")
        compiled = self._get_or_compile(optimisation_level, force=force_run)
        handle = backend.process_circuit(compiled, n_shots=shots)
        counts = backend.get_result(handle).get_counts()
        log.debug(f"Retrieved {len(counts)} unique measurement outcomes")
        metadata = self._ensure_resource_metadata(backend_name, optimisation_level, compiled)

        # Cache the execution result
        self._cache_execution_result(backend_name, shots, optimisation_level, counts, handle, metadata)

        return {"counts": counts, "handle": handle, "metadata": metadata}

    def run_aer(self, shots = 1024, force_run = False):
        """
        Execute the quantum circuit using the Aer simulator backend, without transpilation or optimisation.

        Args:
            shots (int, optional): Number of execution shots. Defaults to 1024.
            force_run (bool, optional): If True, bypass cache and execute again. Defaults to False.

        Returns:
            dict: Dictionary with keys 'counts', 'handle', and 'metadata'.

        Raises:
            ValueError: If arguments are invalid.
        """
        # Ensure main circuit is available and flattened
        main_circuit = self._get_or_build_main_circuit()
        if not self.flattened:
            self.flatten_registers()
            
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer.")
        if not isinstance(force_run, bool):
            raise ValueError("force_run must be a boolean.")

        backend = AerBackend()
        
        # Check for cached Aer result
        cached_result = self._get_cached_execution_result("aer", shots, None)
        if not force_run and cached_result:
            log.debug(f"Using cached Aer results for {shots} shots")
            return cached_result

        log.info(f"Executing circuit on Aer simulator with {shots} shots")
        handle = backend.process_circuit(main_circuit, n_shots=shots)
        counts = backend.get_result(handle).get_counts()
        log.debug(f"Aer simulation completed, retrieved {len(counts)} unique outcomes")
        
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()
        
        metadata = {
            "backend": "aer",
            "optimisation_level": None,
            "solver_type": solver_type,
            "puzzle_hash": puzzle_hash,
            "n_qubits": main_circuit.n_qubits,
            "n_gates": main_circuit.n_gates,
            "depth": main_circuit.depth()
        }

        # Cache the execution result
        self._cache_execution_result("aer", shots, None, counts, handle, metadata)

        return {"counts": counts, "handle": handle, "metadata": metadata}

    def _get_latest_cache_key_for_backend(self):
        """
        Returns the most recent cache key for the current backend, if available.

        Returns:
            tuple: (backend_name, shots, optimisation_level, puzzle_hash, solver_type) or None if not found.
        """
        if not self._execution_cache['counts']:
            return None
        # Find the latest cache key for the current backend
        candidates = [k for k in self._execution_cache['counts'].keys() if k[0] == self.current_backend]
        if not candidates:
            return None
        # Return the most recently added
        return candidates[-1]

    def counts_plot(self, counts=None, shots=None, optimisation_level=None, backend_name=None):
        """
        Generate and display a bar chart visualization of quantum measurement results.

        This method creates a matplotlib bar chart showing the distribution of quantum
        states measured during circuit execution. It provides a visual analysis tool
        for understanding algorithm behavior, solution quality, and quantum effects.

        The visualization helps with:
        - Solution analysis: Identifying most probable states (potential solutions)
        - Algorithm debugging: Spotting unexpected state distributions
        - Noise analysis: Comparing ideal vs. real hardware results
        - Performance comparison: Visualizing results across different backends/settings

        Args:
            counts (dict, optional): Measurement counts mapping bit strings to frequencies.
                Format: {'000...0': count1, '000...1': count2, ...}
                If None, automatically retrieves cached results from recent executions.
                Defaults to None.
            shots (int, optional): Filter for specific shot count when using cached results.
                If None, uses most recent available results regardless of shot count.
                Defaults to None.
            optimisation_level (int, optional): Filter for specific optimization level 
                when using cached results. Valid values: 0-3 or None for Aer results.
                If None, uses most recent available results. Defaults to None.
            backend_name (str, optional): Filter for specific backend when using cached
                results. If None, uses current backend. Examples: "ibm_brisbane", "aer".
                Defaults to None.

        Raises:
            ValueError: If no counts are available (neither provided nor cached) and
                no quantum executions have been performed yet.

        Display:
            Creates a matplotlib figure with:
            - Bar chart of quantum states (x-axis) vs. measurement counts (y-axis)
            - Backend name in title for context
            - Rotated x-axis labels for readability with many states
            - Professional styling suitable for research presentations

        Example:
            >>> # Plot results from most recent execution
            >>> solver.run(shots=1024)
            >>> solver.counts_plot()
            
            >>> # Plot specific cached results
            >>> solver.counts_plot(shots=2048, optimisation_level=2, backend_name="ibm_brisbane")
            
            >>> # Plot custom counts data
            >>> custom_counts = {'0000': 512, '1111': 512}
            >>> solver.counts_plot(counts=custom_counts)
            
            >>> # Compare multiple results
            >>> aer_result = solver.run_aer(shots=1024)
            >>> solver.counts_plot(counts=aer_result['counts'])  # Aer results
            >>> 
            >>> ibm_result = solver.run(shots=1024)
            >>> solver.counts_plot(counts=ibm_result['counts'])  # Hardware results

        Note:
            - Requires matplotlib for display functionality
            - Works best in Jupyter notebook environments
            - For large state spaces, consider filtering to most probable states
            - State labels are automatically formatted for readability
        """
        log.debug(f"Creating counts plot: counts={'provided' if counts else 'from_cache'}, shots={shots}, opt_level={optimisation_level}")
        
        if counts is None:
            # Use the new helper method to get execution results
            result = self.get_execution_results(backend_name=backend_name, shots=shots, opt_level=optimisation_level)
            
            if result is None:
                log.error("No execution results available for plotting")
                raise ValueError("No counts available. Run run() or run_aer() first.")
            
            # Handle case where multiple results are returned
            if isinstance(result, list):
                # Use the most recent result
                result = result[-1]
                log.debug(f"Multiple results found, using most recent")
            
            counts = result["counts"]
            plot_backend = result["cache_key"][0]  # Extract backend name from cache key
            log.debug(f"Using cached counts from backend '{plot_backend}'")
        else:
            # Use provided counts
            plot_backend = backend_name or "Unknown"
            log.debug("Using provided counts for plotting")

        log.info(f"Generating plot for {len(counts)} measurement outcomes from {plot_backend}")
        
        # Prepare data for plotting
        keys = list(counts.keys())
        values = list(counts.values())
        keys_str = [str(key) for key in keys]  # Convert binary states to strings for display

        # Create and customize the bar plot
        plt.figure(figsize=(7, 5))
        plt.bar(keys_str, values, color='royalblue')
        
        # Format backend name for title
        title_backend = "Aer Simulator" if plot_backend == "aer" else plot_backend
        plt.title(f"Measurement Counts ({title_backend})")
        plt.xlabel('States')
        plt.ylabel('Counts')
        plt.xticks(rotation=90)  # Rotate labels for better readability
        plt.tight_layout()
        plt.show()
        log.debug("Counts plot displayed successfully")

    def _get_execution_cache_key(self, backend_name: str, shots: int, opt_level: int):
        """Generate consistent cache key for execution results including puzzle and solver identification."""
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()
        cache_key = (backend_name, shots, opt_level, puzzle_hash, solver_type)
        log.debug(f"Generated execution cache key: {cache_key}")
        return cache_key
    
    def _get_transpilation_cache_key(self, backend_name: str, opt_level: int):
        """Generate consistent cache key for transpilation results including puzzle and solver identification."""
        puzzle_hash = self._get_puzzle_hash()
        solver_type = self._get_solver_type()
        return (backend_name, opt_level, puzzle_hash, solver_type)
    
    def _cache_execution_result(self, backend_name: str, shots: int, opt_level: int, counts, handle, metadata):
        """Cache execution results consistently and save to disk."""
        cache_key = self._get_execution_cache_key(backend_name, shots, opt_level)
        self._execution_cache['counts'][cache_key] = counts
        self._execution_cache['handles'][cache_key] = handle
        self._execution_cache['metadata'][cache_key] = metadata
        
        # Save to disk after caching
        self._save_caches_to_disk()
    
    def _get_cached_execution_result(self, backend_name: str, shots: int, opt_level: int):
        """Retrieve cached execution results if available."""
        cache_key = self._get_execution_cache_key(backend_name, shots, opt_level)
        if cache_key in self._execution_cache['counts']:
            return {
                "counts": self._execution_cache['counts'][cache_key],
                "handle": self._execution_cache['handles'][cache_key],
                "metadata": self._execution_cache['metadata'][cache_key]
            }
        return None
    
    def _cache_transpilation_metadata(self, backend_name: str, opt_level: int, metadata):
        """Cache transpilation metadata consistently and save to disk."""
        cache_key = self._get_transpilation_cache_key(backend_name, opt_level)
        self._transpilation_cache[cache_key] = metadata
        
        # Save to disk after caching
        self._save_caches_to_disk()
    
    def _get_cached_transpilation_metadata(self, backend_name: str, opt_level: int):
        """Retrieve cached transpilation metadata if available."""
        cache_key = self._get_transpilation_cache_key(backend_name, opt_level)
        return self._transpilation_cache.get(cache_key)

    def _get_solver_type(self):
        """
        Get a string identifier for the current solver type.
        
        Returns:
            str: Class name of the solver (e.g., 'ExactCoverQuantumSolver')
        """
        solver_type = self.__class__.__name__
        log.debug(f"Solver type: {solver_type}")
        return solver_type

    def _get_puzzle_hash(self):
        """
        Get the puzzle hash if working with a Sudoku instance.
        
        Returns:
            str: Puzzle hash from sudoku.get_hash() or 'no_puzzle' if not using Sudoku
        """
        if self.sudoku and hasattr(self.sudoku, 'get_hash'):
            puzzle_hash = self.sudoku.get_hash()
            log.debug(f"Puzzle hash: {puzzle_hash}")
            return puzzle_hash
        else:
            log.debug("No puzzle available, using 'no_puzzle' as hash")
            return 'no_puzzle'
    
    def get_execution_results(self, backend_name: str = None, shots: int = None, opt_level: int = None):
        """
        Retrieve execution results from cache with flexible filtering.
        
        Args:
            backend_name (str, optional): Filter by backend name. If None, uses current backend.
            shots (int, optional): Filter by number of shots. If None, returns all.
            opt_level (int, optional): Filter by optimization level. If None, returns all.
            
        Returns:
            dict or list: Single result dict if exact match found, list of results if multiple matches, None if no matches.
        """
        if not self._execution_cache['counts']:
            return None
            
        # If no filters specified, return the most recent result
        if backend_name is None and shots is None and opt_level is None:
            cache_key = list(self._execution_cache['counts'].keys())[-1]
            return {
                "cache_key": cache_key,
                "counts": self._execution_cache['counts'][cache_key],
                "handle": self._execution_cache['handles'][cache_key],
                "metadata": self._execution_cache['metadata'][cache_key]
            }
        
        # Use current backend if not specified
        if backend_name is None:
            backend_name = self.current_backend
            
        # Find matching cache entries
        matches = []
        for key in self._execution_cache['counts'].keys():
            key_backend, key_shots, key_opt_level, key_puzzle_hash, key_solver_type = key
            
            # Check if this key matches our criteria
            if (backend_name is None or key_backend == backend_name) and \
               (shots is None or key_shots == shots) and \
               (opt_level is None or key_opt_level == opt_level):
                matches.append({
                    "cache_key": key,
                    "counts": self._execution_cache['counts'][key],
                    "handle": self._execution_cache['handles'][key],
                    "metadata": self._execution_cache['metadata'][key]
                })
        
        if not matches:
            return None
        elif len(matches) == 1:
            return matches[0]
        else:
            return matches
        
        # Use current backend if not specified
        if backend_name is None:
            backend_name = self.current_backend
            
        # Find matching cache entries
        matches = []
        for key in self._execution_cache['counts'].keys():
            key_backend, key_shots, key_opt_level, key_puzzle_hash, key_solver_type = key
            
            # Check if this key matches our criteria
            if (backend_name is None or key_backend == backend_name) and \
               (shots is None or key_shots == shots) and \
               (opt_level is None or key_opt_level == opt_level):
                matches.append({
                    "cache_key": key,
                    "counts": self._execution_cache['counts'][key],
                    "handle": self._execution_cache['handles'][key],
                    "metadata": self._execution_cache['metadata'][key]
                })
        
        if not matches:
            return None
        elif len(matches) == 1:
            return matches[0]
        else:
            return matches

    def get_latest_counts(self, backend_name: str = None):
        """
        Get the most recent execution counts for a backend.
        
        Args:
            backend_name (str, optional): Backend name. If None, uses current backend.
            
        Returns:
            dict: Counts dictionary or None if no results found.
        """
        result = self.get_execution_results(backend_name=backend_name)
        return result["counts"] if result else None

    def clear_cache(self, cache_type: str = "all"):
        """
        Clear execution, transpilation, compiled circuit, and/or main circuit caches.
        Also removes cache files from disk.
        
        Args:
            cache_type (str): Type of cache to clear: "execution", "transpilation", "compiled", "main", or "all"
        """
        if cache_type in ("execution", "all"):
            self._execution_cache = {'counts': {}, 'handles': {}, 'metadata': {}}
            log.info("Execution cache cleared")
            # Remove execution cache file from disk
            exec_path = self._get_execution_cache_path()
            if os.path.exists(exec_path):
                try:
                    os.remove(exec_path)
                    log.info(f"Deleted execution cache file: {exec_path}")
                except Exception as e:
                    log.warning(f"Could not delete execution cache file: {e}")
            
        if cache_type in ("transpilation", "all"):
            self._transpilation_cache = {}
            log.info("Transpilation cache cleared")
            # Remove transpilation cache file from disk
            trans_path = self._get_transpilation_cache_path()
            if os.path.exists(trans_path):
                try:
                    os.remove(trans_path)
                    log.info(f"Deleted transpilation cache file: {trans_path}")
                except Exception as e:
                    log.warning(f"Could not delete transpilation cache file: {e}")
            
        if cache_type in ("compiled", "all"):
            self._compiled_circuit = None
            self._compiled_circuit_key = None
            log.info("Compiled circuit cache cleared")
            # Remove compiled circuit cache files from disk
            # Note: Compiled circuits are stored per backend/optimization level, so we need to remove all of them
            try:
                cache_base = self._get_cache_base_dir()
                puzzle_hash = self._get_puzzle_hash()
                solver_type = self._get_solver_type()
                
                # Remove compiled circuit files for all backends and optimization levels
                pattern = os.path.join(cache_base, "*", solver_type, puzzle_hash, "compiled_level*.json")
                compiled_files = glob.glob(pattern)
                for file_path in compiled_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        log.info(f"Deleted compiled circuit file: {file_path}")
                
                # Also remove empty directories if they exist
                circuit_dir = os.path.join(cache_base, "*", solver_type, puzzle_hash)
                for dir_path in glob.glob(circuit_dir):
                    if os.path.exists(dir_path) and not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        log.debug(f"Removed empty directory: {dir_path}")
                        
            except Exception as e:
                log.warning(f"Could not delete compiled circuit cache files: {e}")
            
        if cache_type in ("main", "all"):
            self.main_circuit = None
            self._main_circuit_cache_key = None
            self.flattened = False
            log.info("Main circuit cache cleared")
            # Remove main circuit cache file from disk
            main_path = self._main_circuit_cache_path()
            if os.path.exists(main_path):
                try:
                    os.remove(main_path)
                    log.info(f"Deleted main circuit file: {main_path}")
                    
                    # Also remove the directory if it's empty
                    main_dir = os.path.dirname(main_path)
                    if os.path.exists(main_dir) and not os.listdir(main_dir):
                        os.rmdir(main_dir)
                        log.debug(f"Removed empty directory: {main_dir}")
                        
                except Exception as e:
                    log.warning(f"Could not delete main circuit file: {e}")

    def get_cache_stats(self):
        """
        Get statistics about the current cache usage.
        
        Returns:
            dict: Cache statistics including counts and memory usage estimates.
        """
        stats = {
            "execution_results": len(self._execution_cache['counts']),
            "transpilation_metadata": len(self._transpilation_cache),
            "compiled_circuits": 1 if self._compiled_circuit is not None else 0,
            "main_circuits": 1 if self.main_circuit is not None else 0
        }
        
        return stats

    def _get_execution_cache_path(self, create_dirs: bool = False):
        """Get path for execution results cache file."""
        cache_dir = self._get_cache_base_dir()
        if create_dirs:
            os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "execution_cache.json")

    def _get_transpilation_cache_path(self, create_dirs: bool = False):
        """Get path for transpilation metadata cache file."""
        cache_dir = self._get_cache_base_dir()
        if create_dirs:
            os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "transpilation_cache.json")

    def _load_caches_from_disk(self):
        """Load execution and transpilation caches from disk."""
        log.debug("Loading caches from disk")
        
        # Load execution cache
        exec_path = self._get_execution_cache_path()
        if os.path.exists(exec_path):
            try:
                with open(exec_path, 'r') as f:
                    data = json.load(f)
                # Convert JSON string keys back to tuples using json.loads
                self._execution_cache = {
                    'counts': {tuple(json.loads(k)): v for k, v in data.get('counts', {}).items()},
                    'handles': {},  # Handles can't be serialized, skip them
                    'metadata': {tuple(json.loads(k)): v for k, v in data.get('metadata', {}).items()}
                }
                log.info(f"Loaded {len(self._execution_cache['counts'])} execution results from disk")
            except Exception as e:
                log.warning(f"Could not load execution cache: {e}")
                # Initialize empty cache on failure
                self._execution_cache = {'counts': {}, 'handles': {}, 'metadata': {}}
        else:
            log.debug("No execution cache file found, starting with empty cache")
        
        # Load transpilation cache
        trans_path = self._get_transpilation_cache_path()
        if os.path.exists(trans_path):
            try:
                with open(trans_path, 'r') as f:
                    data = json.load(f)
                # Convert JSON string keys back to tuples using json.loads
                self._transpilation_cache = {tuple(json.loads(k)): v for k, v in data.items()}
                log.info(f"Loaded {len(self._transpilation_cache)} transpilation metadata from disk")
            except Exception as e:
                log.warning(f"Could not load transpilation cache: {e}")
                # Initialize empty cache on failure
                self._transpilation_cache = {}
        else:
            log.debug("No transpilation cache file found, starting with empty cache")

    def _save_caches_to_disk(self):
        """Save execution and transpilation caches to disk."""
        # Save execution cache
        exec_path = self._get_execution_cache_path(create_dirs=True)
        try:
            # Convert tuple keys to JSON strings for serialization
            data = {
                'counts': {},
                'metadata': {}
            }
            
            # Safely convert execution cache keys and ensure values are JSON serializable
            for k, v in self._execution_cache['counts'].items():
                try:
                    json_key = json.dumps(k)
                    # Ensure the counts dict itself is JSON serializable
                    if isinstance(v, dict):
                        # Convert any non-string keys in counts to strings
                        serializable_counts = {str(count_key): count_val for count_key, count_val in v.items()}
                    else:
                        serializable_counts = v
                    data['counts'][json_key] = serializable_counts
                except Exception as key_error:
                    log.warning(f"Could not serialize execution cache key {k}: {key_error}")
            
            for k, v in self._execution_cache['metadata'].items():
                try:
                    json_key = json.dumps(k)
                    # Metadata should already be JSON serializable, but double-check
                    json.dumps(v)  # Test serialization
                    data['metadata'][json_key] = v
                except Exception as key_error:
                    log.warning(f"Could not serialize metadata cache key {k}: {key_error}")
            
            with open(exec_path, 'w') as f:
                json.dump(data, f, indent=2)
            log.debug(f"Saved execution cache to {exec_path}")
        except Exception as e:
            log.error(f"Failed to save execution cache: {e}")
        
        # Save transpilation cache
        trans_path = self._get_transpilation_cache_path(create_dirs=True)
        try:
            # Convert tuple keys to JSON strings
            data = {}
            for k, v in self._transpilation_cache.items():
                try:
                    json_key = json.dumps(k)
                    # Test that value is JSON serializable
                    json.dumps(v)
                    data[json_key] = v
                except Exception as key_error:
                    log.warning(f"Could not serialize transpilation cache key {k}: {key_error}")
            
            with open(trans_path, 'w') as f:
                json.dump(data, f, indent=2)
            log.debug(f"Saved transpilation cache to {trans_path}")
        except Exception as e:
            log.error(f"Failed to save transpilation cache: {e}")
