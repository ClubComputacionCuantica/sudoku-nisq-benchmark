"""Unified manager for quantum computing backends across multiple providers."""

from typing import Any, Dict, List, Optional
from pytket.extensions.quantinuum.backends.credential_storage import CredentialStorage
from .providers import QuantumProvider, IBMProvider, QuantinuumProvider


class BackendManager:
    """Unified manager for quantum computing backends across multiple providers.

    This class provides a single interface for managing quantum backends from different
    providers (IBM, Quantinuum, etc.). It uses a provider pattern to keep provider-specific
    code separate while maintaining a unified API.

    :ivar dict[str, :class:`sudoku_nisq.providers.base.QuantumProvider`] _providers: Registry of quantum providers.
    :ivar dict[str, str] _backend_to_provider: Mapping of backend aliases to provider names.

    Example:
        .. code-block:: python

            # Create manager instance
            manager = BackendManager()

            # Initialize IBM backend
            manager.authenticate_ibm(api_token="your_token", instance="your_instance")
            manager.add_backend("ibm", "ibm_brisbane", alias="ibm_device")

            # Initialize Quantinuum backend
            manager.authenticate_quantinuum()
            manager.add_backend("quantinuum", "H1-1", alias="quantinuum_device")

            # Use backends uniformly
            ibm_backend = manager.get("ibm_device")
            quantinuum_backend = manager.get("quantinuum_device")
    """
    
    def __init__(self):
        self._providers: Dict[str, QuantumProvider] = {}
        self._backend_to_provider: Dict[str, str] = {}
        
        # Register built-in providers
        self.register_provider(IBMProvider())
        self.register_provider(QuantinuumProvider())
    
    def register_provider(self, provider: QuantumProvider) -> None:
        """Register a new quantum provider.

        Args:
            provider (:class:`~sudoku_nisq.providers.base.QuantumProvider`): The provider instance to register.
        """
        self._providers[provider.provider_name] = provider
    
    def get_provider(self, provider_name: str) -> QuantumProvider:
        """Get a specific provider by name.

        Args:
            provider_name (str): Name of the provider to retrieve.

        Returns:
            :class:`~sudoku_nisq.providers.base.QuantumProvider`: The requested provider instance.

        Raises:
            ValueError: If the provider is not found.
        """
        if provider_name not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(f"Provider '{provider_name}' not found. Available: {available}")
        return self._providers[provider_name]
    
    def list_providers(self) -> List[str]:
        """List all registered provider names.
        
        Returns:
            List[str]: List of provider names.
        """
        return list(self._providers.keys())
    
    # Unified backend access methods
    def get(self, alias: str) -> Any:
        """Get a backend by alias from any provider.
        
        Args:
            alias (str): The alias of the backend to retrieve.
            
        Returns:
            Any: The backend instance ready for use.
            
        Raises:
            ValueError: If the backend alias is not found.
        """
        if alias not in self._backend_to_provider:
            available = list(self._backend_to_provider.keys())
            if not available:
                raise ValueError(
                    f"Backend '{alias}' not found. No backends registered yet. "
                    f"Use authenticate and add_backend methods to register backends first."
                )
            else:
                raise ValueError(f"Backend '{alias}' not found. Available backends: {available}")
        
        provider_name = self._backend_to_provider[alias]
        return self._providers[provider_name].get_backend(alias)
    
    def add_backend(self, provider_name: str, device: str, alias: Optional[str] = None, **kwargs) -> Any:
        """Add a backend through a specific provider.
        
        Args:
            provider_name (str): Name of the provider (e.g., 'ibm', 'quantinuum').
            device (str): Device name (e.g., 'ibm_brisbane', 'H1-1').
            alias (Optional[str]): Custom alias for the backend.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            Any: The created backend instance.
            
        Raises:
            ValueError: If provider not found or alias already exists.
        """
        provider = self.get_provider(provider_name)
        alias = alias or device
        
        if alias in self._backend_to_provider:
            raise ValueError(f"Backend alias '{alias}' already exists")
        
        backend = provider.add_device(device, alias, **kwargs)
        self._backend_to_provider[alias] = provider_name
        return backend
    
    def remove(self, alias: str) -> None:
        """Remove a backend from any provider.
        
        Args:
            alias (str): The alias of the backend to remove.
            
        Raises:
            ValueError: If the backend alias is not found.
        """
        if alias not in self._backend_to_provider:
            raise ValueError(f"Backend '{alias}' not found")
        
        provider_name = self._backend_to_provider[alias]
        self._providers[provider_name].remove_backend(alias)
        del self._backend_to_provider[alias]
    
    def all(self) -> List[str]:
        """List all backend aliases across all providers.
        
        Returns:
            List[str]: List of all registered backend aliases.
        """
        return list(self._backend_to_provider.keys())
    
    def all_backends(self) -> Dict[str, Any]:
        """Get all backend instances across all providers.
        
        Returns:
            Dict[str, Any]: Dictionary mapping alias to backend instance.
        """
        backends = {}
        for alias in self._backend_to_provider:
            try:
                backends[alias] = self.get(alias)
            except Exception as e:
                backends[alias] = f"Error: {e}"
        return backends
    
    def aliases(self) -> List[str]:
        """List all backend aliases (same as all()).
        
        Returns:
            List[str]: List of all registered backend aliases.
        """
        return self.all()
    
    def clear(self) -> None:
        """Clear all backends from all providers."""
        for provider in self._providers.values():
            provider.clear_backends()
        self._backend_to_provider.clear()
    
    def is_registered(self, alias: str) -> bool:
        """Check if a backend alias is registered.
        
        Args:
            alias (str): The alias to check.
            
        Returns:
            bool: True if the alias is registered, False otherwise.
        """
        return alias in self._backend_to_provider
    
    def info(self) -> Dict[str, Dict[str, Any]]:
        """Get info about all backends across all providers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping alias to backend info.
        """
        info = {}
        for provider in self._providers.values():
            info.update(provider.backend_info())
        return info
    
    def count(self) -> int:
        """Count total backends across all providers.
        
        Returns:
            int: Total number of registered backends.
        """
        return len(self._backend_to_provider)
    
    # Provider-specific convenience methods
    def authenticate_ibm(self, **kwargs) -> List[str]:
        """Authenticate with IBM provider.
        
        Args:
            **kwargs: Arguments passed to IBM provider's authenticate method.
            
        Returns:
            List[str]: List of available IBM devices.
        """
        return self.get_provider("ibm").authenticate(**kwargs)
    
    def authenticate_quantinuum(self, **kwargs) -> List[str]:
        """Authenticate with Quantinuum provider.
        
        Args:
            **kwargs: Arguments passed to Quantinuum provider's authenticate method.
            
        Returns:
            List[str]: List of available Quantinuum devices.
        """
        return self.get_provider("quantinuum").authenticate(**kwargs)
    
    def list_available_ibm_devices(self, **kwargs) -> List[str]:
        """List available IBM devices.
        
        Returns:
            List[str]: List of available IBM device names.
        """
        return self.get_provider("ibm").list_available_devices(**kwargs)
    
    def list_available_quantinuum_devices(self, **kwargs) -> List[str]:
        """List available Quantinuum devices.
        
        Returns:
            List[str]: List of available Quantinuum device names.
        """
        return self.get_provider("quantinuum").list_available_devices(**kwargs)
    
    def add_ibm_device(self, device: str, alias: Optional[str] = None, **kwargs) -> Any:
        """Add an IBM device backend.
        
        Args:
            device (str): IBM device name (e.g., 'ibm_brisbane').
            alias (Optional[str]): Custom alias for the device.
            **kwargs: Additional arguments passed to IBM provider.
            
        Returns:
            Any: The created IBM backend instance.
        """
        alias = alias or device
        backend = self.add_backend("ibm", device, alias, **kwargs)
        return backend
    
    def add_quantinuum_device(self, device: str, alias: Optional[str] = None, **kwargs) -> Any:
        """Add a Quantinuum device backend.
        
        Args:
            device (str): Quantinuum device name (e.g., 'H1-1').
            alias (Optional[str]): Custom alias for the device.
            **kwargs: Additional arguments passed to Quantinuum provider.
            
        Returns:
            Any: The created Quantinuum backend instance.
        """
        alias = alias or device
        backend = self.add_backend("quantinuum", device, alias, **kwargs)
        return backend
    
    def init_ibm(self, device: str, alias: Optional[str] = None, **kwargs) -> str:
        """Initialize IBM backend in one step (authenticate + add device).
        
        Args:
            device (str): IBM device name.
            alias (Optional[str]): Custom alias for the device.
            **kwargs: Arguments for authentication (api_token, instance) and device setup.
            
        Returns:
            str: The alias used for the backend.
        """
        provider = self.get_provider("ibm")
        alias = provider.init_device(device=device, alias=alias, **kwargs)
        self._backend_to_provider[alias] = "ibm"
        return alias
    
    def init_quantinuum(self, device: str, alias: Optional[str] = None, **kwargs) -> str:
        """Initialize Quantinuum backend in one step (authenticate + add device).
        
        Args:
            device (str): Quantinuum device name.
            alias (Optional[str]): Custom alias for the device.
            **kwargs: Arguments for authentication (token_store, provider) and device setup.
            
        Returns:
            str: The alias used for the backend.
        """
        provider = self.get_provider("quantinuum")
        alias = provider.init_device(device=device, alias=alias, **kwargs)
        self._backend_to_provider[alias] = "quantinuum"
        return alias
    
    def list_available_devices(self, provider_name: str, **kwargs) -> List[str]:
        """List available devices for a specific provider.
        
        Args:
            provider_name (str): Name of the provider.
            **kwargs: Additional arguments passed to the provider.
            
        Returns:
            List[str]: List of available device names.
        """
        return self.get_provider(provider_name).list_available_devices(**kwargs)
    
    def validate_alias(self, alias: str) -> None:
        """Validate that an alias is available for use.
        
        Args:
            alias (str): The alias to validate.
            
        Raises:
            ValueError: If the alias is already in use.
        """
        if self.is_registered(alias):
            raise ValueError(f"Alias '{alias}' is already registered")
    
    def get_backend_sdk(self, alias: str) -> str:
        """Get the SDK type for a given backend alias.
        
        Args:
            alias (str): The backend alias to check.
            
        Returns:
            str: SDK name ("pytket", "qiskit", "braket")
            
        Raises:
            ValueError: If the backend alias is not found.
        """
        if alias not in self._backend_to_provider:
            raise ValueError(f"Backend '{alias}' not found")
        
        backend = self.get(alias)
        
        # Check for pytket backend characteristics
        if hasattr(backend, 'get_compiled_circuit') and hasattr(backend, 'process_circuit'):
            return "pytket"
        
        # Check for qiskit backend characteristics  
        elif hasattr(backend, 'transpile') or 'qiskit' in str(type(backend)).lower():
            return "qiskit"
            
        # Check for braket backend characteristics
        elif hasattr(backend, 'run') and 'braket' in str(type(backend)).lower():
            return "braket"
            
        else:
            # Default to pytket for unknown backends
            return "pytket"

    # -----------------------------
    # Singleton accessor (no API shadowing)
    # -----------------------------
    _singleton: Optional["BackendManager"] = None

    @classmethod
    def inst(cls) -> "BackendManager":
        """Get or create the process-wide BackendManager instance.

        Use this in call sites: BackendManager.inst().get(alias)
        """
        if cls._singleton is None:
            cls._singleton = BackendManager()
        return cls._singleton
