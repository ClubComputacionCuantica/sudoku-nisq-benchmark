from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pytket import Circuit, Qubit, OpType
from pytket.passes import FlattenRegisters
from pytket.circuit.display import render_circuit_jupyter as draw
from pytket.extensions.qiskit import IBMQBackend, AerBackend

# The ExactCoverQuantumSolver, GraphColoringQuantumSolver, and BacktrackingQuantumSolver
# classes inherit from QuantumSolver and implement their specific logic.
# The QuantumSolver class provides a unified interface for different quantum backends,
# allowing for easy switching and execution of quantum circuits.
# The code is structured to allow for easy extension and integration with various quantum backends,
# while maintaining a clean and modular design.

class QuantumSolver(ABC):
    """
    Abstract base class for quantum solvers. Provides a unified interface for quantum circuit
    execution across different backend providers.

    This class handles:
    - Backend management (Aer, IBMQ, etc.)
    - Circuit compilation and optimization
    - Resource analysis and caching
    - Result collection and visualization

    Attributes:
        main_circuit (Circuit): The main quantum circuit to be executed
        flattened (bool): Tracks if registers have been flattened
        backends (Dict): Dictionary mapping backend names to their instances
        current_backend (str): Name of the currently selected backend
        backend_counts (Dict): Cache for execution results
        backend_handles (Dict): Cache for backend execution handles
    """
    def __init__(self, backends=None):
        """
        Initialize a new QuantumSolver instance.

        Args:
            backends (Dict, optional): Dictionary of pre-configured backend instances.
                                     Defaults to None.
        """
        self.main_circuit = None
        self.flattened = False  # tracks if registers have been flattened

        # backends is a dict: name -> backend instance
        self.backends = backends.copy() if backends else {}
        self.current_backend = None
        
        # Caches for run() results
        self.backend_counts = {}  # backend_name -> counts
        self.backend_handles = {}  # backend_name -> handle

    @abstractmethod
    def find_resources(self):
        """
        Determine quantum resources needed for the problem.
        
        This method should be implemented by subclasses to analyze the problem
        and determine required qubits, gates, and other quantum resources.
        """
        pass

    @abstractmethod
    def get_circuit(self):
        """
        Construct and return a pytket Circuit for the problem.
        
        This method should be implemented by subclasses to build the quantum
        circuit that solves their specific problem.

        Returns:
            Circuit: A pytket quantum circuit object.
        """
        pass
    
    def draw_circuit(self):
        """
        Visualize the quantum circuit using pytket's circuit drawing functionality.
        The circuit will be displayed in a Jupyter notebook environment.
        """
        draw(self.main_circuit)

    def flatten_registers(self):
        """This method flattens the circuit's registers for compatibility with generic backends."""
        if self.main_circuit is None:
            raise ValueError(f"main_circuit not yet constructed. Call get_circuit() first.")
        FlattenRegisters().apply(self.main_circuit)
        self.flattened = True
        return self.main_circuit

    def add_backend(self, name: str = "ibm", backend_instance: str = "ibm_brisbane"):
        """Register a backend by instance name."""
        if backend_instance.startswith("ibm"):
            self.backends[name] = IBMQBackend(backend_instance)
            return

        if name == "aer" or backend_instance == "AerBackend":
            raise ValueError("Call init_aer() to set Aer backend, not add_backend().")

        raise ValueError(f"Unsupported backend instance: {backend_instance}")


    def set_backend(self, name: str):
        """Select the active backend by name."""
        if name not in self.backends:
            raise KeyError(f"Backend '{name}' not registered")
        self.current_backend = name

    def init_ibm(self, token: str):
        """Authenticate with IBM and list available backends."""
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, overwrite=True)
        devices = IBMQBackend.available_devices()
        
        print("IBM authentication successful.")
        print(f"IBM devices available to your account:\n{devices}")
        print("Call add_backend(name, backend_instance) to register one.")

        return devices

    def init_aer(self, aer_name: str = "aer"):
        """Register and set Aer simulator as backend."""
        self.add_backend(aer_name, AerBackend())
        self.set_backend(aer_name)

    def init_quantinuum(self, token: str):
        """
        Initialize connection to Quantinuum quantum computers.
        
        Note: This is a placeholder method for future implementation.

        Args:
            token (str): Authentication token for Quantinuum service.
        """
        pass

    def _ensure_cache_dict(self, attr: str, backend_name: str) -> Dict:
        """
        Ensure that a given attribute (usually a cache dictionary) exists and contains an entry
        for the specified backend. Initializes them if needed.

        Args:
            attr (str): The name of the attribute to ensure (e.g., 'compiled_circuits').
            backend_name (str): The name of the backend to initialize the sub-dictionary for.

        Returns:
            Dict: The sub-dictionary corresponding to the backend.
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
            Any: The backend instance.

        Raises:
            ValueError: If the backend is not registered.
        """
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not registered.")
        return self.backends[backend_name]

    def _ensure_resource_metadata(self, backend_name: str, optimisation_level: int, compiled) -> Dict[str, Any]:
        """
        
        Compute and cache resource metadata for a compiled circuit, including qubit count,
        gate count, and depth.

        Args:
            backend_name (str): Identifier for the backend.
            optimisation_level (int): Optimization level used during compilation.
            compiled: The compiled quantum circuit object.

        Returns:
            Dict[str, Any]: Metadata dictionary for the compiled circuit.
        """
        resources = self._ensure_cache_dict("transpiled_resources", backend_name)
        metadata = {
            "backend": backend_name,
            "optimisation_level": optimisation_level,
            "n_qubits": compiled.n_qubits,
            "n_gates": compiled.n_gates,
            "depth": compiled.depth()
        }
        resources[optimisation_level] = metadata
        return metadata

    def _get_or_compile(self, backend, backend_name: str, level: int, force: bool = False):
        """
        Retrieve a cached compiled circuit or compile a new one if needed.

        Args:
            backend: Backend object supporting get_compiled_circuit().
            backend_name (str): Backend identifier.
            level (int): Optimization level (0-3).
            force (bool): If True, force recompilation.

        Returns:
            The compiled quantum circuit.
        """
        self._get_backend(backend_name)
        compiled_dict = self._ensure_cache_dict("compiled_circuits", backend_name)

        if not force and level in compiled_dict:
            return compiled_dict[level]

        compiled = backend.get_compiled_circuit(self.main_circuit, optimisation_level=level)
        compiled_dict[level] = compiled
        return compiled

    def find_transpiled_resources(self,
                                  optimisation_levels: List[int] = [0, 1, 2, 3],
                                  force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Compile and analyze the circuit at multiple optimization levels to extract resource usage.

        Args:
            optimisation_levels (List[int]): Optimization levels to analyze (0-3).
            force_refresh (bool): If True, recompile even if metadata is cached.

        Returns:
            List[Dict[str, Any]]: Metadata dictionaries or error information per level.
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

            if not force_refresh and level in cache_dict:
                resources.append(cache_dict[level])
                continue

            try:
                circuit = self._get_or_compile(backend, backend_name, level, force=force_refresh)
                metadata = self._ensure_resource_metadata(backend_name, level, circuit)
                resources.append(metadata)
            except Exception as e:
                error_info = {
                    "backend": backend_name,
                    "optimisation_level": level,
                    "error": str(e)
                }
                cache_dict[level] = error_info
                resources.append(error_info)

        return resources

    def run(self,
            shots: int = 1024,
            force_run: bool = False,
            optimisation_level: int = 1) -> Dict[str, Any]:
        """
        Execute the compiled quantum circuit and return execution results and metadata.

        Args:
            shots (int): Number of execution shots.
            force_run (bool): If True, bypass cache and execute again.
            optimisation_level (int): Optimization level used in compilation.

        Returns:
            Dict[str, Any]: Dictionary with keys 'counts', 'handle', and 'metadata'.
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
        cache_key = (backend_name, shots, optimisation_level)

        self.backend_counts = getattr(self, "backend_counts", {})
        self.backend_handles = getattr(self, "backend_handles", {})
        self.backend_metadata = getattr(self, "backend_metadata", {})

        if not force_run and cache_key in self.backend_counts:
            return {
                "counts": self.backend_counts[cache_key],
                "handle": self.backend_handles[cache_key],
                "metadata": self.backend_metadata[cache_key]
            }

        if not hasattr(backend, "get_compiled_circuit") or \
           not hasattr(backend, "process_circuit") or \
           not hasattr(backend, "get_result"):
            raise RuntimeError(f"Backend '{type(backend).__name__}' does not support run().")

        compiled = self._get_or_compile(backend, backend_name, optimisation_level, force=force_run)
        handle = backend.process_circuit(compiled, n_shots=shots)
        counts = backend.get_result(handle).get_counts()
        metadata = self._ensure_resource_metadata(backend_name, optimisation_level, compiled)

        self.backend_handles[cache_key] = handle
        self.backend_counts[cache_key] = counts
        self.backend_metadata[cache_key] = metadata

        return {"counts": counts, "handle": handle, "metadata": metadata}

    def counts_plot(self, counts=None):
        """
        Plots a bar chart of the measurement probabilities for different quantum states.

        This method visualizes the probability distribution of measured quantum states
        using matplotlib. It can either use provided counts or retrieve cached counts
        from the current backend.

        Args:
            counts (dict, optional): Dictionary mapping quantum states to their counts.
                                   If None, uses cached counts from current backend.

        Raises:
            RuntimeError: If no backend is selected.
            ValueError: If no counts are available and none are provided.
        """
        if counts is None:
            if self.current_backend is None:
                raise RuntimeError("No backend selected. Call set_backend() first.")
            counts = self.backend_counts.get(self.current_backend)
            if counts is None:
                raise ValueError("No counts available. Run run() first.")

        import matplotlib.pyplot as plt
        from pytket.utils import probs_from_counts

        # Convert raw counts to probability distribution
        data = probs_from_counts(counts)
        keys = list(data.keys())
        values = list(data.values())
        keys_str = [str(key) for key in keys]  # Convert binary states to strings for display

        # Create and customize the bar plot
        plt.figure(figsize=(7, 5))
        plt.bar(keys_str, values, color='royalblue')
        plt.title(f"Measurement Probabilities ({self.current_backend})")
        plt.xlabel('States')
        plt.ylabel('Probability')
        plt.xticks(rotation=90)  # Rotate labels for better readability
        plt.tight_layout()
        plt.show()


