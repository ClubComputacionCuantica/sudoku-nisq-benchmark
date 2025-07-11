from abc import ABC, abstractmethod
from pytket.passes import FlattenRegisters
from pytket.circuit.display import render_circuit_jupyter as draw
from pytket.extensions.qiskit import IBMQBackend, AerBackend
import os
import json
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

# Logging usage:
# - Set log level via QUANTUM_SOLVER_LOG_LEVEL (e.g., DEBUG, INFO, WARNING)
# - To log to a file, set QUANTUM_SOLVER_LOG_FILE to a writable file path
# - Example: export QUANTUM_SOLVER_LOG_LEVEL=DEBUG
#            export QUANTUM_SOLVER_LOG_FILE=solver.log
# - Use 'log.info()', 'log.warning()', etc. in code

# The ExactCoverQuantumSolver, GraphColoringQuantumSolver, and BacktrackingQuantumSolver
# classes inherit from QuantumSolver and implement their specific logic.
# The QuantumSolver class provides a unified interface for different quantum backends,
# allowing for easy switching and execution of quantum circuits.
# The code is structured to allow for easy extension and integration with various quantum backends,
# while maintaining a clean and modular design.

class QuantumSolver(ABC):
    """
    Abstract base class for quantum solvers, providing a unified interface for quantum circuit execution across different backend providers.

    This class handles:
        - Backend management (IBMQ, etc.)
        - Circuit compilation and optimization
        - Resource analysis and caching
        - Result collection and visualization
        - Aer simulator run

    Attributes:
        main_circuit (Circuit): The main quantum circuit to be executed.
        flattened (bool): Tracks if registers have been flattened for backend compatibility.
        backends (Dict[str, Any]): Dictionary mapping backend names to their instances.
        current_backend (str): Name of the currently selected backend.
        backend_counts (Dict[Tuple, Dict]): Cache for execution results (counts).
        backend_handles (Dict[Tuple, Any]): Cache for backend execution handles.
        backend_metadata (Dict[Tuple, Dict]): Cache for execution metadata.

    Methods:
        get_circuit(): Construct and return a pytket Circuit for the problem (abstract).
        resource_estimation(): Analyze and return quantum resources needed (abstract).
        draw_circuit(): Visualize the quantum circuit in a Jupyter notebook.
        flatten_registers(): Flatten circuit registers for backend compatibility.
        add_backend(name, backend_instance): Register a backend by instance name.
        set_backend(name): Select the active backend by name.
        init_ibm(token): Authenticate with IBM and list available backends.
        init_quantinuum(token): Placeholder for Quantinuum backend initialization.
        find_transpiled_resources(optimisation_levels, force_refresh): Compile and analyze circuit resources at multiple optimization levels.
        run(shots, force_run, optimisation_level): Execute the compiled quantum circuit and return results and metadata.
        run_aer(shots, force_run): Execute the circuit using the Aer simulator backend.
        counts_plot(counts, shots, optimisation_level): Plot a bar chart of measurement counts.
    """
    def __init__(self, backends=None):
        """
        Initialize a new QuantumSolver instance.

        Args:
            backends (dict, optional): Dictionary of pre-configured backend instances. Defaults to None.
        """
        self.main_circuit = None
        self.flattened = False  # tracks if registers have been flattened

        # Backend management
        self.backends = backends.copy() if backends else {}
        self.current_backend = None
        
        # Execution caches - use consistent cache keys: (backend, shots, opt_level)
        self._execution_cache = {
            'counts': {},    # (backend, shots, opt_level) -> counts
            'handles': {},   # (backend, shots, opt_level) -> handle  
            'metadata': {}   # (backend, shots, opt_level) -> metadata
        }
        
        # Transpilation cache - separate from execution: (backend, opt_level) -> metadata
        self._transpilation_cache = {}

    @abstractmethod
    def get_circuit(self):
        """
        Construct and return a pytket Circuit for the problem.

        This method should be implemented by subclasses to build the quantum circuit that solves their specific problem instance.

        Returns:
            Circuit: A pytket quantum circuit object representing the problem.
        """
        pass
    
    @abstractmethod
    def resource_estimation(self):
        """
        Analyze and return quantum resources needed for the problem instance.

        This method should be implemented by subclasses to determine required qubits, gates, and other quantum resources.
        """
        pass
    
    def draw_circuit(self):
        """
        Visualize the quantum circuit using pytket's circuit drawing functionality.

        The circuit will be displayed in a Jupyter notebook environment. Requires that `main_circuit` is constructed.
        """
        draw(self.main_circuit)

    def flatten_registers(self):
        """
        Flatten the circuit's registers for compatibility with generic backends.

        Raises:
            ValueError: If `main_circuit` is not yet constructed.

        Returns:
            Circuit: The flattened circuit.
        """
        if self.main_circuit is None:
            log.error("Attempted to flatten registers but main_circuit is None")
            raise ValueError(f"main_circuit not yet constructed. Call get_circuit() first.")
        log.debug("Flattening circuit registers for backend compatibility")
        FlattenRegisters().apply(self.main_circuit)
        self.flattened = True
        log.debug("Circuit registers flattened successfully")
        return self.main_circuit

    def add_backend(self, name: str = "ibm_brisbane", backend_instance: str = "ibm_brisbane", **kwargs):
        """
        Register a backend by instance name (debug version).

        Args:
            name (str): Alias for the backend.
            backend_instance (str): Backend instance name.
            **kwargs: Additional backend-specific arguments.

        For debugging: prints arguments and exceptions.
        """
        log.debug(f"Attempting to register backend '{name}' with instance '{backend_instance}'")
        print(f"add_backend called with name={name}, backend_instance={backend_instance}, kwargs={kwargs}")
        try:
            self.backends[name] = IBMQBackend(backend_instance)
            log.info(f"Successfully registered backend '{name}' with instance '{backend_instance}'")
            print(f"Backend '{name}' registered successfully.")
        except Exception as e:
            log.error(f"Failed to register backend '{name}': {e}")
            print(f"Error registering backend '{name}': {e}")
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

        References:
            pytket-qiskit config: https://docs.quantinuum.com/tket/extensions/pytket-qiskit/
        """
        from pytket.extensions.qiskit import set_ibmq_config, IBMQBackend
        self.instance = instance
        if not instance:
            raise ValueError("'instance' is required for IBM Quantum hardware access and transpilation.")

        # Set credentials for pytket-qiskit
        set_ibmq_config(ibmq_api_token=token, instance=instance)

        # List available devices
        devices = IBMQBackend.available_devices(instance=instance)
        print("IBM authentication successful (pytket-qiskit).")
        print("IBM devices available to your account:")
        print(devices)
        #for dev in devices:
        #    print(f"  - '{dev.device_name}' ({dev.n_qubits} qubits)")
        print("Call add_backend(name, backend_instance) to register one.")
        return [dev.device_name for dev in devices]

    def init_quantinuum(self, token: str):
        """
        Initialize connection to Quantinuum quantum computers (placeholder).

        Args:
            token (str): Authentication token for Quantinuum service.
        """
        pass

    def _ensure_cache_dict(self, attr: str, backend_name: str):
        """
        Ensure that a given attribute (usually a cache dictionary) exists and contains an entry for the specified backend.
        Initializes them if needed.

        Args:
            attr (str): The name of the attribute to ensure (e.g., 'compiled_circuits').
            backend_name (str): The name of the backend to initialize the sub-dictionary for.

        Returns:
            dict: The sub-dictionary corresponding to the backend.
        """
        if not hasattr(self, attr):
            setattr(self, attr, {})
        getattr(self, attr).setdefault(backend_name, {})
        return getattr(self, attr)[backend_name]

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
            raise ValueError(f"Backend '{backend_name}' not registered.")
        return self.backends[backend_name]

    def _ensure_resource_metadata(self, backend_name: str, optimisation_level: int, compiled):
        """
        Compute and cache resource metadata for a compiled circuit, including qubit count, gate count, and depth.

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
        
        # Compute metadata
        metadata = {
            "backend": backend_name,
            "optimisation_level": optimisation_level,
            "n_qubits": compiled.n_qubits,
            "n_gates": compiled.n_gates,
            "depth": compiled.depth()
        }
        
        # Cache the metadata
        self._cache_transpilation_metadata(backend_name, optimisation_level, metadata)
        
        # Also maintain backward compatibility with old cache structure
        resources = self._ensure_cache_dict("transpiled_resources", backend_name)
        resources[optimisation_level] = metadata
        
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

    def _compiled_circuit_cache_path(self, backend_name: str, level: int) -> str:
        """
        Returns a stable cache path for a (backend, level)-tuple.

        Args:
            backend_name (str): Name of the backend.
            level (int): Optimization level.

        Returns:
            str: Path to the cache file for the compiled circuit.
        """
        # Use configurable cache base dir
        cache_dir = os.path.join(self._get_cache_base_dir(), backend_name)
        os.makedirs(cache_dir, exist_ok=True)

        # File name based on optimization level and backend
        fname = f"compiled_level{level}.json"
        return os.path.join(cache_dir, fname)

    def _get_or_compile(self, level: int, force: bool = False):
        """
        Retrieve a cached compiled circuit or compile a new one if needed.
        Uses a two-level cache (in-memory and disk) to avoid recompilation.

        This method first checks an in-memory cache. If not found, it checks for a
        pre-compiled circuit on disk. If neither cache contains the circuit, it
        compiles the circuit, stores it in both caches, and then returns it.

        The disk cache is robust against file corruption; if a cache file is
        invalid, it will be ignored and the circuit will be recompiled.

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

        compiled_dict = self._ensure_cache_dict("compiled_circuits", backend_name)

        # Generate a cache path based on backend name and optimization level
        cache_path = self._compiled_circuit_cache_path(backend_name, level)

        if not force:
            # Try in-memory cache first
            if level in compiled_dict:
                log.debug(f"In-memory cache hit for {backend_name} level {level}.")
                return compiled_dict[level]
            # Try disk cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        circ_dict = json.load(f)
                    compiled = Circuit.from_dict(circ_dict)
                    compiled_dict[level] = compiled
                    log.info(f"Loaded compiled circuit from disk cache: {cache_path}")
                    return compiled
                except (json.JSONDecodeError, IOError) as e:
                    log.warning(f"Could not load from cache file {cache_path} due to {e}. Recompiling.")

        # Compile and cache
        log.info(f"Compiling circuit for {backend_name} at level {level}...")
        compiled = backend.get_compiled_circuit(self.main_circuit, optimisation_level=level)
        compiled_dict[level] = compiled
        # Save to disk
        try:
            with open(cache_path, "w") as f:
                json.dump(compiled.to_dict(), f)
            log.info(f"Saved compiled circuit to disk cache: {cache_path}")
        except IOError as e:
            log.error(f"Failed to save compiled circuit to {cache_path}: {e}")
        return compiled

    def find_transpiled_resources(
        self,
        optimisation_levels = [0, 1, 2, 3],
        force_refresh = False
    ):
        """
        Compile and analyze the circuit at multiple optimization levels to extract resource usage.

        Args:
            optimisation_levels (list, optional): Optimization levels to analyze (0-3). Defaults to [0, 1, 2, 3].
            force_refresh (bool, optional): If True, recompile even if metadata is cached. Defaults to False.

        Returns:
            list: Metadata dictionaries or error information per level.

        Raises:
            RuntimeError: If no backend is selected.
            ValueError: If invalid optimization levels are provided or circuit is not constructed.
        """
        if self.current_backend is None:
            raise RuntimeError("No backend selected. Call set_backend() first.")
        if not all(isinstance(l, int) and l in [0, 1, 2, 3] for l in optimisation_levels):
            raise ValueError("optimisation_levels must be a list of integers in {0, 1, 2, 3}.")

        backend_name = self.current_backend
        backend = self._get_backend(backend_name)

        if self.main_circuit is None:
            raise ValueError("main_circuit not yet constructed. Call get_circuit() first.")

        if not self.flattened:
            self.flatten_registers()

        if not hasattr(backend, "get_compiled_circuit"):
            raise RuntimeError(f"Backend '{type(backend).__name__}' does not support transpilation.")

        resources = []
        cache_dict = self._ensure_cache_dict("transpiled_resources", backend_name)

        for level in optimisation_levels:
            if force_refresh:
                cache_dict.pop(level, None)
                log.debug(f"Forcing refresh for optimization level {level}")

            if not force_refresh and level in cache_dict:
                log.debug(f"Using cached resource data for level {level}")
                resources.append(cache_dict[level])
                continue

            try:
                log.debug(f"Compiling circuit at optimization level {level}")
                circuit = self._get_or_compile(level, force=force_refresh)
                metadata = self._ensure_resource_metadata(backend_name, level, circuit)
                log.info(f"Level {level}: {metadata['n_qubits']} qubits, {metadata['n_gates']} gates, depth {metadata['depth']}")
                resources.append(metadata)
            except Exception as e:
                log.error(f"Failed to compile at optimization level {level}: {e}")
                error_info = {
                    "backend": backend_name,
                    "optimisation_level": level,
                    "error": str(e)
                }
                cache_dict[level] = error_info
                resources.append(error_info)

        return resources

    def run(
        self,
        shots = 1024,
        force_run = False,
        optimisation_level = 1
    ):
        """
        Execute the compiled quantum circuit and return execution results and metadata.

        Args:
            shots (int, optional): Number of execution shots. Defaults to 1024.
            force_run (bool, optional): If True, bypass cache and execute again. Defaults to False.
            optimisation_level (int, optional): Optimization level used in compilation. Defaults to 1.

        Returns:
            dict: Dictionary with keys 'counts', 'handle', and 'metadata'.

        Raises:
            ValueError: If arguments are invalid or circuit is not constructed.
            RuntimeError: If no backend is selected or backend does not support execution.
        """
        if self.main_circuit is None:
            raise ValueError("main_circuit not yet constructed. Call get_circuit() first.")
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
            ValueError: If arguments are invalid or circuit is not constructed.
        """
        if self.main_circuit is None:
            raise ValueError("main_circuit not yet constructed. Call get_circuit() first.")
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
        handle = backend.process_circuit(self.main_circuit, n_shots=shots)
        counts = backend.get_result(handle).get_counts()
        log.debug(f"Aer simulation completed, retrieved {len(counts)} unique outcomes")
        metadata = {
            "backend": "aer",
            "optimisation_level": None,
            "n_qubits": self.main_circuit.n_qubits,
            "n_gates": self.main_circuit.n_gates,
            "depth": self.main_circuit.depth()
        }

        # Cache the execution result
        self._cache_execution_result("aer", shots, None, counts, handle, metadata)

        return {"counts": counts, "handle": handle, "metadata": metadata}

    def _get_latest_cache_key_for_backend(self):
        """
        Returns the most recent cache key for the current backend, if available.

        Returns:
            tuple: (backend_name, shots, optimisation_level) or None if not found.
        """
        if not self._execution_cache['counts']:
            return None
        # Find the latest cache key for the current backend
        candidates = [k for k in self._execution_cache['counts'].keys() if k[0] == self.current_backend]
        if not candidates:
            return None
        # Return the most recently added
        return candidates[-1]

    @property
    def backend_counts(self):
        """Backward compatibility property for accessing execution counts."""
        return self._execution_cache['counts']
    
    @property
    def backend_handles(self):
        """Backward compatibility property for accessing execution handles."""
        return self._execution_cache['handles']
    
    @property
    def backend_metadata(self):
        """Backward compatibility property for accessing execution metadata."""
        return self._execution_cache['metadata']

    def counts_plot(self, counts=None, shots=None, optimisation_level=None):
        """
        Plot a bar chart of the measurement counts for different quantum states.

        This method visualizes the probability distribution of measured quantum states using matplotlib. It can either use provided counts or retrieve cached counts from whichever backend was run last (current backend or Aer).

        Args:
            counts (dict, optional): Dictionary mapping quantum states to their counts. If None, uses cached counts from the most recently run backend.
            shots (int, optional): Number of shots to select cached result. If None, uses most recent.
            optimisation_level (int, optional): Optimisation level to select cached result. If None, uses most recent.

        Raises:
            ValueError: If no counts are available and none are provided.
        """
        log.debug(f"Creating counts plot: counts={'provided' if counts else 'from_cache'}, shots={shots}, opt_level={optimisation_level}")
        if counts is None:
            if not self.backend_counts:
                log.error("No counts available for plotting and none provided")
                raise ValueError("No counts available. Run run() or run_aer() first.")
            
            # If specific parameters are provided, try to find exact match
            if shots is not None and optimisation_level is not None:
                if self.current_backend:
                    cache_key = (self.current_backend, shots, optimisation_level)
                    if cache_key in self.backend_counts:
                        counts = self.backend_counts[cache_key]
                        backend_name = self.current_backend
                    else:
                        raise ValueError(f"No counts found for backend '{self.current_backend}' with {shots} shots and optimization level {optimisation_level}")
                else:
                    raise ValueError("No backend selected and specific parameters provided")
            elif shots is not None:
                # Look for Aer with specific shots
                aer_key = ("aer", shots, None)
                if aer_key in self.backend_counts:
                    counts = self.backend_counts[aer_key]
                    backend_name = "aer"
                else:
                    raise ValueError(f"No Aer counts found with {shots} shots")
            else:
                # Find the most recently added cache entry (last run)
                if not self.backend_counts:
                    raise ValueError("No counts available. Run run() or run_aer() first.")
                
                # Get the most recent cache key (Python dicts maintain insertion order since 3.7)
                cache_key = list(self.backend_counts.keys())[-1]
                counts = self.backend_counts[cache_key]
                backend_name = cache_key[0]
                log.debug(f"Using cached counts from backend '{backend_name}'")
        else:
            # Use provided counts, but we don't know which backend
            backend_name = "Unknown"
            log.debug("Using provided counts for plotting")

        log.info(f"Generating plot for {len(counts)} measurement outcomes from {backend_name}")
        # Use raw counts directly
        keys = list(counts.keys())
        values = list(counts.values())
        keys_str = [str(key) for key in keys]  # Convert binary states to strings for display

        # Create and customize the bar plot
        plt.figure(figsize=(7, 5))
        plt.bar(keys_str, values, color='royalblue')
        
        # Use the actual backend name from the cache key for the title
        if 'backend_name' in locals():
            title_backend = "Aer Simulator" if backend_name == "aer" else backend_name
        else:
            title_backend = "Custom Data"
        
        plt.title(f"Measurement Counts ({title_backend})")
        plt.xlabel('States')
        plt.ylabel('Counts')
        plt.xticks(rotation=90)  # Rotate labels for better readability
        plt.tight_layout()
        plt.show()
        log.debug("Counts plot displayed successfully")

    def _get_execution_cache_key(self, backend_name: str, shots: int, opt_level: int):
        """Generate consistent cache key for execution results."""
        return (backend_name, shots, opt_level)
    
    def _get_transpilation_cache_key(self, backend_name: str, opt_level: int):
        """Generate consistent cache key for transpilation results."""
        return (backend_name, opt_level)
    
    def _cache_execution_result(self, backend_name: str, shots: int, opt_level: int, counts, handle, metadata):
        """Cache execution results consistently."""
        cache_key = self._get_execution_cache_key(backend_name, shots, opt_level)
        self._execution_cache['counts'][cache_key] = counts
        self._execution_cache['handles'][cache_key] = handle
        self._execution_cache['metadata'][cache_key] = metadata
    
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
        """Cache transpilation metadata consistently."""
        cache_key = self._get_transpilation_cache_key(backend_name, opt_level)
        self._transpilation_cache[cache_key] = metadata
    
    def _get_cached_transpilation_metadata(self, backend_name: str, opt_level: int):
        """Retrieve cached transpilation metadata if available."""
        cache_key = self._get_transpilation_cache_key(backend_name, opt_level)
        return self._transpilation_cache.get(cache_key)
