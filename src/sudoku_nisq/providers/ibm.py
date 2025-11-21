"""IBM Quantum provider implementation."""

from typing import List, Optional, Any
from pytket.extensions.qiskit import IBMQBackend, set_ibmq_config
from qiskit_ibm_runtime import QiskitRuntimeService
from .base import QuantumProvider


class IBMProvider(QuantumProvider):
    """IBM Quantum provider implementation."""
    
    @property
    def provider_name(self) -> str:
        return "ibm"
    
    @property
    def sdk_type(self) -> str:
        """IBM Quantum uses Qiskit SDK."""
        return "qiskit"
    
    def authenticate(self, **kwargs: Any) -> List[str]:
        """Authenticate with IBM Quantum and return available devices.

        Accepts provider-specific parameters via **kwargs to keep signature
        compatible with abstract base class.

        Expected kwargs:
            api_token (str): IBM Quantum Platform API token (required)
            instance (str | None): IBM Quantum instance CRN (optional)
            overwrite (bool): Force re-authentication if already configured

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

        set_ibmq_config(ibmq_api_token=api_token, instance=instance)
        self._configured = True

        try:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform",
                token=api_token,
                instance=instance,
                overwrite=True
            )
            print("IBM authentication successful")
            return self.list_available_devices()
        except Exception as e:
            print(f"IBM authentication successful but failed to list devices: {e}")
            return []
    
    def list_available_devices(self, **kwargs) -> List[str]:
        """List available IBM Quantum devices without re-authentication.
        
        Returns:
            List[str]: List of available IBM Quantum device names.
            
        Raises:
            RuntimeError: If authenticate() has not been called first.
        """
        if not self._configured:
            raise RuntimeError("Call authenticate() before listing devices")
            
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
                devices = IBMQBackend.available_devices(device="ibm_brisbane")
                device_names = [dev.device_name for dev in devices if dev.device_name is not None]
                print(f"Found {len(devices)} IBM devices available to your account")
                print(f"Available devices: {device_names}")
                return device_names
            except Exception as fallback_e:
                print(f"Fallback also failed: {fallback_e}")
                return []
    
    def add_device(
        self, 
        device: str, 
        alias: Optional[str] = None,
        **kwargs
    ) -> IBMQBackend:
        """Register an IBM Quantum device for use in the backend registry.
        
        Args:
            device (str): IBM Quantum device name (e.g., "ibm_brisbane").
            alias (Optional[str]): Custom alias for the device.
                
        Returns:
            IBMQBackend: The initialized and ready-to-use backend instance.
            
        Raises:
            RuntimeError: If authenticate() has not been called first.
        """
        if not self._configured:
            raise RuntimeError("Call authenticate() before adding devices")
            
        name = alias or device
        backend = IBMQBackend(device)
        self._backends[name] = backend
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