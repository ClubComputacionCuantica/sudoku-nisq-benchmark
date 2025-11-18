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
    
    def authenticate(
        self,
        api_token: str,
        instance: Optional[str] = None,
        overwrite: bool = False,
        **kwargs
    ) -> List[str]:
        """Configure IBM Quantum credentials and discover available devices.
        
        Args:
            api_token (str): Your IBM Quantum Platform API token.
            instance (Optional[str]): Your IBM Quantum instance CRN.
            overwrite (bool): If True, forces re-authentication even if already configured.
                
        Returns:
            List[str]: List of available quantum device names for your account.
            
        Raises:
            RuntimeError: If authentication fails or credentials are invalid.
        """
        if self._configured and not overwrite:
            print("IBM provider already configured. Use overwrite=True to reconfigure.")
            return self.list_available_devices()
            
        set_ibmq_config(ibmq_api_token=api_token, instance=instance)
        self._configured = True
        
        # List and return available devices after successful authentication
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
    
    def init_device(
        self,
        api_token: str,
        instance: str,
        device: str,
        alias: Optional[str] = None,
        **kwargs
    ) -> str:
        """Initialize IBM device in one step (authenticate + add device).
        
        Args:
            api_token (str): Your IBM Quantum Platform API token.
            instance (str): Your IBM Quantum instance CRN.
            device (str): IBM Quantum device name.
            alias (Optional[str]): Custom alias for the device.
            
        Returns:
            str: The alias used for the backend.
        """
        if not self._configured:
            self.authenticate(api_token=api_token, instance=instance)
        
        alias = alias or device
        self.add_device(device, alias)
        return alias