"""
Qiskit Aer Provider for local quantum simulation.

This module provides comprehensive access to Qiskit Aer's simulation capabilities,
including multiple simulation methods, noise models, GPU acceleration, and custom
backend configurations.
"""

from typing import Optional, Dict, Any, List
import logging

from .base import QuantumProvider

logger = logging.getLogger(__name__)


class AerProvider(QuantumProvider):
    """
    Provider for Qiskit Aer local quantum simulation.
    
    Supports:
    - Multiple simulation methods (statevector, density_matrix, stabilizer, MPS, etc.)
    - Noise models (custom or from real devices)
    - GPU acceleration (when available)
    - Precision control (single/double)
    - Custom backend options
    
    Examples
    --------
    Basic ideal simulation:
    
    >>> from sudoku_nisq.backends import BackendManager
    >>> manager = BackendManager.inst()
    >>> alias = manager.init_aer(method="statevector")
    >>> backend = manager.get_backend(alias)
    
    Simulation with noise model:
    
    >>> from qiskit_aer.noise import NoiseModel, depolarizing_error
    >>> noise_model = NoiseModel()
    >>> noise_model.add_all_qubit_quantum_error(
    ...     depolarizing_error(0.01, 2), ['cx']
    ... )
    >>> alias = manager.init_aer(
    ...     method="density_matrix",
    ...     noise_model=noise_model,
    ...     alias="noisy_sim"
    ... )
    
    GPU-accelerated simulation:
    
    >>> alias = manager.init_aer(
    ...     method="statevector",
    ...     device="GPU",
    ...     precision="single"
    ... )
    
    Device emulation from real backend:
    
    >>> from qiskit_ibm_runtime import QiskitRuntimeService
    >>> from qiskit_aer.noise import NoiseModel
    >>> service = QiskitRuntimeService()
    >>> real_backend = service.backend("ibm_brisbane")
    >>> noise_model = NoiseModel.from_backend(real_backend)
    >>> alias = manager.init_aer(
    ...     method="density_matrix",
    ...     noise_model=noise_model,
    ...     coupling_map=real_backend.coupling_map,
    ...     basis_gates=real_backend.configuration().basis_gates
    ... )
    """
    
    def __init__(self):
        super().__init__()
        self._configured = True  # No authentication needed for local simulator
        
    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "aer"
    
    @property
    def sdk_type(self) -> str:
        """SDK type for this provider."""
        return "qiskit"
    
    @property
    def is_configured(self) -> bool:
        """Aer is always available (no authentication required)."""
        return self._configured
    
    def authenticate(self, **kwargs) -> List[str]:
        """
        No authentication needed for local Aer simulator.
        
        Returns
        -------
        list
            Available simulation methods.
        """
        self._configured = True
        return self.list_available_devices()
    
    def list_available_devices(self, **kwargs) -> List[str]:
        """
        List available Aer simulation methods.
        
        Returns
        -------
        list
            Available simulation methods that can be used as device names.
        
        Examples
        --------
        >>> provider = AerProvider()
        >>> methods = provider.list_available_devices()
        >>> print(methods)
        ['automatic', 'statevector', 'density_matrix', 'stabilizer', ...]
        """
        try:
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            methods = list(sim.available_methods())  # Convert tuple to list
            logger.info(f"Available Aer simulation methods: {methods}")
            return methods
        except ImportError:
            logger.error("qiskit-aer is not installed")
            return []
        except Exception as e:
            logger.error(f"Error querying Aer methods: {e}")
            return ["automatic", "statevector", "density_matrix", "stabilizer"]
    
    def add_device(
        self,
        device: str,
        alias: Optional[str] = None,
        method: Optional[str] = None,
        noise_model: Optional[Any] = None,
        coupling_map: Optional[Any] = None,
        basis_gates: Optional[List[str]] = None,
        device_type: str = "CPU",
        precision: str = "double",
        max_parallel_threads: Optional[int] = None,
        max_parallel_experiments: Optional[int] = None,
        blocking_enable: bool = True,
        blocking_qubits: int = 5,
        **backend_options
    ) -> str:
        """
        Add an AerSimulator backend with specified configuration.
        
        Parameters
        ----------
        device : str
            Simulation method (e.g., "automatic", "statevector", "density_matrix").
            Also used as default alias if none provided.
        alias : str, optional
            Custom alias for this backend configuration.
        method : str, optional
            Override simulation method (defaults to device parameter).
        noise_model : NoiseModel, optional
            Qiskit Aer noise model for noisy simulation.
        coupling_map : list or CouplingMap, optional
            Device coupling map for layout constraints.
        basis_gates : list, optional
            Basis gates for device emulation.
        device_type : str, default "CPU"
            Compute device: "CPU" or "GPU" (requires qiskit-aer-gpu).
        precision : str, default "double"
            Floating point precision: "single" or "double".
        max_parallel_threads : int, optional
            Maximum threads for OpenMP parallelization.
        max_parallel_experiments : int, optional
            Maximum parallel circuit executions.
        blocking_enable : bool, default True
            Enable automatic qubit blocking for large circuits.
        blocking_qubits : int, default 5
            Qubits per block for blocked simulation.
        ``backend_options``
            Additional AerSimulator backend options.
        
        Returns
        -------
        str
            The alias assigned to this backend.
        
        Raises
        ------
        ImportError
            If qiskit-aer is not installed.
        ValueError
            If specified method is not available.
        
        Examples
        --------
        Ideal statevector simulation:
        
        >>> provider = AerProvider()
        >>> alias = provider.add_device("statevector")
        
        Noisy density matrix simulation:
        
        >>> from qiskit_aer.noise import depolarizing_error, NoiseModel
        >>> noise = NoiseModel()
        >>> noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['u1', 'u2', 'u3'])
        >>> alias = provider.add_device(
        ...     "density_matrix",
        ...     noise_model=noise,
        ...     alias="noisy_dm"
        ... )
        
        GPU-accelerated MPS simulation:
        
        >>> alias = provider.add_device(
        ...     "matrix_product_state",
        ...     device_type="GPU",
        ...     precision="single",
        ...     alias="mps_gpu"
        ... )
        """
        try:
            from qiskit_aer import AerSimulator
        except ImportError as e:
            raise ImportError(
                "qiskit-aer is required for Aer provider. "
                "Install with: pip install qiskit-aer"
            ) from e
        
        # Use device as method if method not explicitly provided
        sim_method = method or device
        
        # Validate method availability
        available_methods = self.list_available_devices()
        if sim_method not in available_methods and available_methods:
            logger.warning(
                f"Method '{sim_method}' may not be available. "
                f"Available methods: {available_methods}"
            )
        
        # Build backend options
        options = {
            "method": sim_method,
            "device": device_type,
            "precision": precision,
            "blocking_enable": blocking_enable,
            "blocking_qubits": blocking_qubits,
        }
        
        # Add optional parameters
        if noise_model is not None:
            options["noise_model"] = noise_model
        if coupling_map is not None:
            options["coupling_map"] = coupling_map
        if basis_gates is not None:
            options["basis_gates"] = basis_gates
        if max_parallel_threads is not None:
            options["max_parallel_threads"] = max_parallel_threads
        if max_parallel_experiments is not None:
            options["max_parallel_experiments"] = max_parallel_experiments
        
        # Merge additional backend options
        options.update(backend_options)
        
        # Create AerSimulator with options
        try:
            backend = AerSimulator(**options)
            logger.info(
                f"Created AerSimulator with method='{sim_method}', "
                f"device='{device_type}', precision='{precision}'"
            )
        except Exception as e:
            logger.error(f"Failed to create AerSimulator: {e}")
            # Fallback to basic configuration
            backend = AerSimulator()
            logger.warning("Created AerSimulator with default configuration")
        
        # Register backend
        backend_alias = alias or f"aer_{device}"
        self._backends[backend_alias] = backend
        
        logger.info(f"Registered Aer backend with alias '{backend_alias}'")
        return backend_alias
    
    def init_device(self, device: str, alias: Optional[str] = None, **kwargs) -> str:
        """
        Initialize Aer device (no authentication needed, directly add device).
        
        This is the unified interface method called by BackendManager for consistency
        with other providers. Since Aer is a local simulator, no authentication is
        required - this simply calls add_device with the provided parameters.
        
        Parameters
        ----------
        device : str
            Simulation method to use.
        alias : str, optional
            Custom alias for the backend.
        ``kwargs``
            Additional arguments passed to add_device (method, noise_model, etc.).
        
        Returns
        -------
        str
            The alias assigned to this backend.
        
        Examples
        --------
        >>> provider = AerProvider()
        >>> alias = provider.init_device("statevector", alias="my_sim")
        """
        return self.add_device(device=device, alias=alias, **kwargs)
    
    def get_device(self, alias: str):
        """
        Get AerSimulator backend by alias.
        
        Parameters
        ----------
        alias : str
            Backend alias.
        
        Returns
        -------
        AerSimulator
            The configured backend instance.
        
        Raises
        ------
        KeyError
            If alias not found in registered backends.
        """
        if alias not in self._backends:
            raise KeyError(
                f"Backend '{alias}' not found. "
                f"Available: {list(self._backends.keys())}"
            )
        return self._backends[alias]
    
    def query_available_devices(self) -> Dict[str, Any]:
        """
        Query detailed information about Aer capabilities.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'methods': List of available simulation methods
            - 'devices': List of available compute devices (CPU, GPU)
            - 'has_gpu': Whether GPU support is available
            - 'version': qiskit-aer version
        
        Examples
        --------
        >>> provider = AerProvider()
        >>> info = provider.query_available_devices()
        >>> if info['has_gpu']:
        ...     print("GPU acceleration available!")
        >>> print(f"Aer version: {info['version']}")
        """
        try:
            from qiskit_aer import AerSimulator
            import qiskit_aer
            
            sim = AerSimulator()
            methods = list(sim.available_methods())  # Convert tuple to list
            devices = list(sim.available_devices())  # Convert tuple to list
            
            return {
                "methods": methods,
                "devices": devices,
                "has_gpu": "GPU" in devices,
                "version": qiskit_aer.__version__,
            }
        except ImportError:
            return {
                "methods": [],
                "devices": ["CPU"],
                "has_gpu": False,
                "version": "not installed",
            }
        except Exception as e:
            logger.error(f"Error querying Aer capabilities: {e}")
            return {
                "methods": ["automatic"],
                "devices": ["CPU"],
                "has_gpu": False,
                "version": "unknown",
            }
