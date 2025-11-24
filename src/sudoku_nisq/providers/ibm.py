"""IBM Quantum provider implementation using native Qiskit runtime backends."""

from typing import List, Optional, Any
from qiskit_ibm_runtime import QiskitRuntimeService
from .base import QuantumProvider


class IBMProvider(QuantumProvider):
    """IBM Quantum provider implementation using native Qiskit runtime backends.
    
    This provider returns native QiskitRuntimeService backends that support
    the full Qiskit 2.2+ transpilation pipeline with 6-stage compilation:
    init → layout → routing → translation → optimization → scheduling.
    
    Native backends provide access to Target-driven compilation and advanced
    features like dynamic circuits, pulse-level control, and error mitigation.
    """
    
    def __init__(self):
        """Initialize IBM provider."""
        super().__init__()
        self._service: Optional[QiskitRuntimeService] = None
    
    @property
    def provider_name(self) -> str:
        return "ibm"
    
    @property
    def sdk_type(self) -> str:
        """IBM Quantum uses native Qiskit SDK."""
        return "qiskit"
    
    def authenticate(self, **kwargs: Any) -> List[str]:
        """Authenticate with IBM Quantum and return available devices.

        Accepts provider-specific parameters via ``**kwargs`` to keep the
        signature compatible with the abstract base class.

        Expected keyword arguments (in ``kwargs``):
            api_token (str): IBM Quantum Platform API token (required).
            instance (str | None): IBM Quantum instance CRN (optional).
            overwrite (bool): Force re-authentication if already configured.

        Raises:
            ValueError: If required parameters are missing.
        """
        api_token = kwargs.get("api_token")
        if not api_token:
            raise ValueError("'api_token' is required for IBM authentication")
        instance = kwargs.get("instance")
        overwrite = bool(kwargs.get("overwrite", False))

        if self._configured and not overwrite:
            print("IBM provider already configured. Use overwrite=True to reconfigure.")
            return self.list_available_devices()

        try:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                token=api_token,
                instance=instance,
                overwrite=True
            )
            # Initialize service for immediate use
            self._service = QiskitRuntimeService()
            self._configured = True
            print("IBM authentication successful (native Qiskit runtime)")
            return self.list_available_devices()
        except Exception as e:
            raise RuntimeError(f"IBM authentication failed: {e}") from e
    
    def list_available_devices(self, **kwargs) -> List[str]:
        """List available IBM Quantum devices without re-authentication.
        
        Returns:
            List[str]: List of available IBM Quantum device names.
            
        Raises:
            RuntimeError: If authenticate() has not been called first.
        """
        if not self._configured:
            raise RuntimeError("Call authenticate() before listing devices")
        
        if self._service is None:
            self._service = QiskitRuntimeService()
            
        try:
            backends = self._service.backends()
            device_names = [backend.name for backend in backends]
            print(f"Found {len(backends)} IBM devices available to your account")
            print(f"Available devices: {device_names}")
            return device_names
        except Exception as e:
            raise RuntimeError(f"Failed to list IBM devices: {e}") from e
    
    def add_device(
        self, 
        device: str, 
        alias: Optional[str] = None,
        **kwargs
    ):
        """Register an IBM Quantum device for use in the backend registry.
        
        Args:
            device (str): IBM Quantum device name (e.g., "ibm_brisbane").
            alias (Optional[str]): Custom alias for the device.
                
        Returns:
            The native Qiskit runtime backend instance.
            
        Raises:
            RuntimeError: If authenticate() has not been called first.
        """
        if not self._configured:
            raise RuntimeError("Call authenticate() before adding devices")
        
        if self._service is None:
            self._service = QiskitRuntimeService()
            
        name = alias or device
        backend = self._service.backend(device)
        self._backends[name] = backend
        print(f"Added IBM device '{device}' as '{name}' (native Qiskit backend)")
        return backend
    
    def init_device(self, device: str, alias: Optional[str] = None, **kwargs: Any) -> str:
        """Initialize IBM device (authenticate if needed) and return alias.

        Expects api_token / instance in kwargs. Keeps signature uniform with
        QuantumProvider base class for mypy compatibility.
        """
        if not self._configured:
            self.authenticate(**kwargs)
        alias = alias or device
        self.add_device(device, alias)
        return alias