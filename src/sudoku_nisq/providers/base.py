"""Abstract base class for quantum computing providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class QuantumProvider(ABC):
    """Abstract base class for quantum computing providers."""
    
    def __init__(self) -> None:
        self._configured = False
        self._backends: Dict[str, Any] = {}
    
    @property
    def is_configured(self) -> bool:
        """Check if the provider is configured/authenticated."""
        return self._configured
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'ibm', 'quantinuum')."""
        pass
    
    @property
    @abstractmethod
    def sdk_type(self) -> str:
        """SDK used by this provider ('qiskit', 'pytket', 'braket')."""
        pass
    
    @abstractmethod
    def authenticate(self, **kwargs: Any) -> List[str]:
        """Authenticate with the provider and return available devices.

        Subclasses should accept provider-specific parameters via ``**kwargs``.
        Required parameters must be validated internally and raise a
        descriptive error if missing.
        """
        pass
    
    @abstractmethod
    def list_available_devices(self, **kwargs) -> List[str]:
        """List available devices for this provider."""
        pass
    
    @abstractmethod
    def add_device(self, device: str, alias: Optional[str] = None, **kwargs) -> Any:
        """Add a device backend to the provider's registry."""
        pass

    @abstractmethod
    def init_device(self, device: str, alias: Optional[str] = None, **kwargs: Any) -> str:
        """Initialize device (authenticate if needed + add device) and return alias.

        Unified interface so BackendManager can call provider.init_device(...)
        without needing provider-type specific casts.
        """
        pass
    
    def get_backend(self, alias: str) -> Any:
        """Get a backend by alias from this provider."""
        if alias not in self._backends:
            raise ValueError(f"Backend '{alias}' not found in {self.provider_name} provider")
        return self._backends[alias]
    
    def remove_backend(self, alias: str) -> None:
        """Remove a backend from this provider."""
        if alias not in self._backends:
            raise ValueError(f"Backend '{alias}' not found in {self.provider_name} provider")
        del self._backends[alias]
    
    def list_backends(self) -> List[str]:
        """List all backend aliases for this provider."""
        return list(self._backends.keys())
    
    def clear_backends(self) -> None:
        """Clear all backends for this provider."""
        self._backends.clear()
    
    def backend_info(self) -> Dict[str, Dict[str, Any]]:
        """Get info about all backends for this provider."""
        info = {}
        for alias, backend in self._backends.items():
            try:
                info[alias] = {
                    "provider": self.provider_name,
                    "type": type(backend).__name__,
                    "device": getattr(backend, "device_name", getattr(backend, "name", None)),
                }
            except Exception as e:
                info[alias] = {"provider": self.provider_name, "error": str(e)}
        return info