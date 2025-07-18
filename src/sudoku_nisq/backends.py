"""
BackendManager: Global registry for quantum hardware backends.

Provides a centralized way to authenticate once and access backends by alias.
Key design principles:
- One global registry for all hardware backends  
- Fail-fast validation with helpful error messages
- Simplified init_*() methods for one-step setup
- Clear separation: Aer simulator handled separately, not in this registry

Usage:
    # One-time setup
    BackendManager.init_ibm(api_token="...", instance="...", device="ibm_brisbane")  
    BackendManager.init_quantinuum(device="H1-1", alias="h1", provider="microsoft")
    
    # Access backends
    backend = BackendManager.get("ibm_brisbane")
    all_aliases = BackendManager.all()
"""

from typing import Any, ClassVar, Dict, Optional, List
from pytket.extensions.qiskit import IBMQBackend, set_ibmq_config
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import MemoryCredentialStorage, CredentialStorage
from qiskit_ibm_runtime import QiskitRuntimeService

class BackendManager:
    """
    Global registry for quantum hardware backends with fail-fast validation.
    
    Design principles:
    - Single global registry: authenticate once, use everywhere
    - Fail-fast validation: helpful error messages on missing backends  
    - One-step initialization: init_*() methods handle auth + device setup
    - Hardware only: Aer simulator handled separately (not in registry)
    
    Key methods:
    - init_ibm() / init_quantinuum(): One-step backend setup
    - get(): Retrieve backend by alias (fails fast if not found)
    - all(): List all registered aliases  
    - validate_alias(): Explicit validation with helpful errors
    """
    _backends: ClassVar[Dict[str, Any]] = {}
    _ibm_configured: ClassVar[bool] = False
    _quantinuum_configured: ClassVar[bool] = False
    
    @classmethod
    def authenticate_ibm(
        cls,
        api_token: str,
        instance: Optional[str] = None,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Call this once at startup to configure IBM credentials and list available devices.
        
        Args:
            api_token: Your IBM API token
            instance: Your instance CRN (long string beginning with "crn:")
            
        Returns:
            List[str]: List of available device names for your account
        """
        if cls._ibm_configured and not overwrite:
            # If already configured, just return available devices
            return cls.list_available_ibm_devices()
            
        set_ibmq_config(ibmq_api_token=api_token, instance=instance)
        cls._ibm_configured = True
        # List and return available devices after successful authentication
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=api_token, instance=instance, overwrite=True)
            print("IBM authentication successful")
            return cls.list_available_ibm_devices()
        except Exception as e:
            print(f"IBM authentication successful but failed to list devices: {e}")
            return []

    @classmethod
    def list_available_ibm_devices(cls) -> List[str]:
        """
        List available IBM devices without re-authentication.
        
        Returns:
            List[str]: List of available device names
        """
        if not cls._ibm_configured:
            raise RuntimeError("Call authenticate_ibm() first")
        try:
            devices = QiskitRuntimeService().backends()
            device_names = [dev.backend_name for dev in devices if dev.backend_name is not None]
            print(f"Found {len(devices)} IBM devices available to your account")
            print(f"Available devices: {device_names}")
            return device_names
        except Exception as e:
            print(f"Warning: Failed to list devices using QiskitRuntimeService: {e}")
            try:
                print("Attempting fallback method to list devices...")
                # Fallback method using IBMQBackend
                devices = IBMQBackend.available_devices(device="ibm_brisbane")
                device_names = [dev.device_name for dev in devices if dev.device_name is not None]
                print(f"Found {len(devices)} IBM devices available to your account")
                print(f"Available devices: {device_names}")
                return device_names
            except Exception as e:
                print(f"Fallback also failed: {e}")
                return []

    @classmethod
    def add_ibm_device(
        cls,
        device: str,
        alias: Optional[str] = None,
    ) -> IBMQBackend:
        """
        After you've called authenticate_ibm(), use this to
        register as many IBMQ devices as you like.
        
        Args:
            device: IBM device name (e.g., "ibm_kyiv", "ibm_brisbane")
            alias: Optional alias for the device (defaults to device name)
            
        Returns:
            IBMQBackend: The initialized backend instance
        """
        if not cls._ibm_configured:
            raise RuntimeError("Call authenticate_ibm() before adding devices")
        name = alias or device
        backend = IBMQBackend(device)
        cls._backends[name] = backend
        return backend
    
    @classmethod
    def authenticate_quantinuum(
        cls,
        token_store: Optional[CredentialStorage] = None,
        provider: Optional[str] = None,
    ) -> List[str]:
        """
        Call once at startup to log in to Quantinuum and list available devices.

        Args:
            token_store: where to save auth tokens (defaults to in-memory).
            provider: e.g. 'microsoft' for federated login.

        Returns:
            List[str]: names of all available Quantinuum backends.
        """
        if cls._quantinuum_configured:
            return cls.list_available_quantinuum_devices(token_store, provider)

        api_handler = QuantinuumAPI(
            token_store=token_store or MemoryCredentialStorage(),
            provider=provider,
        )
        # This will prompt for credentials if needed
        api_handler.login()
        cls._quantinuum_configured = True

        try:
            infos = QuantinuumBackend.available_devices(api_handler=api_handler)
            names = [info.device_name for info in infos if info.device_name is not None]
            print("Quantinuum authentication successful")
            print(f"Found {len(names)} devices: {names}")
            return names
        except Exception as e:
            print(f"Warning: Authenticated but failed to list devices: {e}")
            return []

    @classmethod
    def add_quantinuum_device(
        cls,
        device: str,
        alias: Optional[str] = None,
        token_store: Optional[CredentialStorage] = None,
        provider: Optional[str] = None,
    ) -> QuantinuumBackend:
        """
        After you've called authenticate_quantinuum(), use this to
        register a Quantinuum backend for later use.

        Args:
            device: e.g. "H1-1", "H2-2E"
            alias: name under which to store it (defaults to `device`)
            token_store: same store you used for authenticate_quantinuum()
            provider: same provider as above

        Returns:
            QuantinuumBackend: the ready-to-use backend instance
        """
        if not cls._quantinuum_configured:
            raise RuntimeError("Call authenticate_quantinuum() before adding devices")

        api_handler = QuantinuumAPI(
            token_store=token_store or MemoryCredentialStorage(),
            provider=provider,
        )
        name = alias or device
        backend = QuantinuumBackend(device_name=device, api_handler=api_handler)
        cls._backends[name] = backend
        return backend

    @classmethod
    def list_available_quantinuum_devices(
        cls,
        token_store: Optional[CredentialStorage] = None,
        provider: Optional[str] = None,
    ) -> List[str]:
        """
        List devices without re‑authenticating.
        """
        api_handler = QuantinuumAPI(
            token_store=token_store or MemoryCredentialStorage(),
            provider=provider,
        )
        infos = QuantinuumBackend.available_devices(api_handler=api_handler)
        return [info.device_name for info in infos if info.device_name is not None]
    
    @classmethod
    def get(cls, alias: str) -> Any:
        """
        Retrieve a previously-registered backend by alias.
        
        Args:
            alias: Backend alias to retrieve
            
        Returns:
            Backend instance
            
        Raises:
            ValueError: If backend not found (fail fast with helpful message)
        """
        if alias not in cls._backends:
            available = list(cls._backends.keys())
            if not available:
                raise ValueError(f"Backend '{alias}' not found. No backends registered yet. "
                               f"Call init_ibm() or init_quantinuum() first.")
            else:
                raise ValueError(f"Backend '{alias}' not found. Available backends: {available}")
        
        return cls._backends[alias]
    
    @classmethod
    def all(cls) -> List[str]:
        """
        Return all registered backend aliases.
        
        Returns:
            List[str]: List of all registered backend aliases
        """
        return list(cls._backends.keys())
    
    @classmethod
    def all_backends(cls) -> Dict[str, Any]:
        """
        Return a shallow copy of alias → backend mapping.
        
        Returns:
            Dict[str, Any]: Copy of the backend registry
        """
        return dict(cls._backends)
    
    @classmethod
    def aliases(cls) -> List[str]:
        """List all registered aliases."""
        return list(cls._backends.keys())

    @classmethod
    def remove(cls, alias: str) -> None:
        """
        Unregister a backend by alias.
        
        Args:
            alias: Backend alias to remove
            
        Raises:
            ValueError: If alias not found
        """
        if alias not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(f"Cannot remove '{alias}' - not found. Available: {available}")
        
        del cls._backends[alias]

    @classmethod
    def clear(cls) -> None:
        """Clear the entire registry."""
        cls._backends.clear()

    @classmethod
    def is_registered(cls, alias: str) -> bool:
        """Check if an alias is present."""
        return alias in cls._backends

    @classmethod
    def info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return summary info for each registered backend:
        { alias: { "type": ..., "device": ... }, … }
        """
        info: Dict[str, Dict[str, Any]] = {}
        for alias, be in cls._backends.items():
            try:
                info[alias] = {
                    "type":   type(be).__name__,
                    "device": getattr(be, "device_name", getattr(be, "name", None)),
                }
            except Exception as e:
                info[alias] = {"error": str(e)}
        return info

    @classmethod
    def init_ibm(cls, api_token: str, instance: str, device: str, alias: Optional[str] = None) -> str:
        """
        One-step IBM backend initialization: authenticate + add device.
        
        Args:
            api_token: Your IBM API token
            instance: Your instance CRN
            device: IBM device name (e.g., "ibm_brisbane", "ibm_kyiv")
            alias: Optional alias for the device (defaults to device name)
            
        Returns:
            str: The alias used for the registered backend
            
        Raises:
            ValueError: If alias already exists
            RuntimeError: If authentication or device addition fails
        """
        alias = alias or device
        
        # Check for existing alias
        if alias in cls._backends:
            raise ValueError(f"Backend alias '{alias}' already exists. Available: {list(cls._backends.keys())}")
        
        try:
            # Authenticate if needed
            if not cls._ibm_configured:
                cls.authenticate_ibm(api_token, instance)
            
            # Add the specific device
            cls.add_ibm_device(device, alias)
            return alias
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize IBM backend '{device}' as '{alias}': {e}") from e

    @classmethod
    def init_quantinuum(
        cls, 
        device: str, 
        alias: Optional[str] = None,
        token_store: Optional[CredentialStorage] = None,
        provider: Optional[str] = None,
    ) -> str:
        """
        One-step Quantinuum backend initialization: authenticate + add device.
        
        Args:
            device: Quantinuum device name (e.g., "H1-1", "H2-2E")
            alias: Optional alias for the device (defaults to device name)
            token_store: Where to save auth tokens (defaults to in-memory)
            provider: e.g. 'microsoft' for federated login
            
        Returns:
            str: The alias used for the registered backend
            
        Raises:
            ValueError: If alias already exists
            RuntimeError: If authentication or device addition fails
        """
        alias = alias or device
        
        # Check for existing alias
        if alias in cls._backends:
            raise ValueError(f"Backend alias '{alias}' already exists. Available: {list(cls._backends.keys())}")
        
        try:
            # Authenticate if needed
            if not cls._quantinuum_configured:
                cls.authenticate_quantinuum(token_store, provider)
            
            # Add the specific device
            cls.add_quantinuum_device(device, alias, token_store, provider)
            return alias
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Quantinuum backend '{device}' as '{alias}': {e}") from e

    @classmethod
    def validate_alias(cls, alias: str) -> None:
        """
        Validate that an alias exists in the registry.
        
        Args:
            alias: Backend alias to validate
            
        Raises:
            ValueError: If alias not found (with helpful message)
        """
        # This is just a wrapper around get() for explicit validation
        cls.get(alias)  # Will raise ValueError if not found
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of registered backends.
        
        Returns:
            int: Number of registered backends
        """
        return len(cls._backends)