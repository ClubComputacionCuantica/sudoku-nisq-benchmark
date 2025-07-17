import pytest
from unittest.mock import MagicMock, patch
from sudoku_nisq.backends import BackendManager

@pytest.fixture(autouse=True)
def clear_backend_manager():
    """Fixture to ensure the BackendManager is cleared before and after each test."""
    BackendManager.clear()
    BackendManager._ibm_configured = False
    BackendManager._quantinuum_configured = False
    yield
    BackendManager.clear()
    BackendManager._ibm_configured = False
    BackendManager._quantinuum_configured = False

def test_initial_state():
    """Test the initial state of the BackendManager."""
    assert BackendManager.all() == []
    assert BackendManager.count() == 0
    assert not BackendManager._ibm_configured
    assert not BackendManager._quantinuum_configured

@patch('sudoku_nisq.backends.set_ibmq_config')
@patch('sudoku_nisq.backends.BackendManager.list_available_ibm_devices')
def test_authenticate_ibm_success(mock_list_devices, mock_set_config):
    """Test successful IBM authentication."""
    mock_list_devices.return_value = ["ibm_test_device"]

    devices = BackendManager.authenticate_ibm("fake_token", "fake_instance")

    mock_set_config.assert_called_once_with(ibmq_api_token="fake_token", instance="fake_instance")
    mock_list_devices.assert_called_once()
    assert BackendManager._ibm_configured
    assert devices == ["ibm_test_device"]

@patch('sudoku_nisq.backends.set_ibmq_config', side_effect=Exception("Auth Error"))
def test_authenticate_ibm_failure(mock_set_config):
    """Test failed IBM authentication."""
    with pytest.raises(Exception, match="Auth Error"):
        BackendManager.authenticate_ibm("fake_token", "fake_instance")
    assert not BackendManager._ibm_configured

@patch('sudoku_nisq.backends.BackendManager.list_available_ibm_devices')
def test_authenticate_ibm_already_configured(mock_list_devices):
    """Test IBM authentication when already configured (should skip auth and list devices)."""
    # Simulate already configured state
    BackendManager._ibm_configured = True
    mock_list_devices.return_value = ["ibm_device1", "ibm_device2"]
    
    devices = BackendManager.authenticate_ibm("fake_token", "fake_instance")
    
    # Should call list_available_ibm_devices without any parameters
    mock_list_devices.assert_called_once_with()
    assert devices == ["ibm_device1", "ibm_device2"]
    
@patch('sudoku_nisq.backends.set_ibmq_config')
@patch('sudoku_nisq.backends.BackendManager.list_available_ibm_devices')
def test_authenticate_ibm_overwrite(mock_list_devices, mock_set_config):
    """Test IBM authentication with overwrite=True."""
    # Simulate already configured state
    BackendManager._ibm_configured = True
    mock_list_devices.return_value = ["ibm_test_device"]
    
    devices = BackendManager.authenticate_ibm("fake_token", "fake_instance", overwrite=True)
    
    # Should call set_ibmq_config even when already configured due to overwrite=True
    mock_set_config.assert_called_once_with(ibmq_api_token="fake_token", instance="fake_instance")
    mock_list_devices.assert_called_once()
    assert devices == ["ibm_test_device"]

@patch('sudoku_nisq.backends.IBMQBackend')
def test_add_ibm_device(mock_ibmq_backend):
    """Test adding an IBM device after authentication."""
    # Simulate authentication
    BackendManager._ibm_configured = True
    
    mock_backend_instance = MagicMock()
    mock_ibmq_backend.return_value = mock_backend_instance

    backend = BackendManager.add_ibm_device("test_device", alias="my_ibm")

    assert backend == mock_backend_instance
    mock_ibmq_backend.assert_called_once_with("test_device")
    assert BackendManager.is_registered("my_ibm")
    assert BackendManager.get("my_ibm") == mock_backend_instance

def test_add_ibm_device_without_auth():
    """Test that adding an IBM device without prior authentication fails."""
    with pytest.raises(RuntimeError, match=r"Call authenticate_ibm\(\) before adding devices"):
        BackendManager.add_ibm_device("test_device")

@patch('sudoku_nisq.backends.QiskitRuntimeService')
def test_list_available_ibm_devices_success(mock_qiskit_runtime_service):
    """Test successful listing of IBM devices."""
    # Simulate authentication
    BackendManager._ibm_configured = True
    
    # Mock QiskitRuntimeService().backends()
    mock_runtime_instance = MagicMock()
    mock_device1 = MagicMock()
    mock_device1.backend_name = "ibm_brisbane"
    mock_device2 = MagicMock()
    mock_device2.backend_name = None  # Test filtering of None values
    mock_device3 = MagicMock()
    mock_device3.backend_name = "ibm_kyiv"
    mock_runtime_instance.backends.return_value = [mock_device1, mock_device2, mock_device3]
    mock_qiskit_runtime_service.return_value = mock_runtime_instance
    
    devices = BackendManager.list_available_ibm_devices()
    
    mock_runtime_instance.backends.assert_called_once()
    assert devices == ["ibm_brisbane", "ibm_kyiv"]  # Should filter out None values

def test_list_available_ibm_devices_not_configured():
    """Test that listing devices without authentication fails."""
    with pytest.raises(RuntimeError, match="Call authenticate_ibm\\(\\) first"):
        BackendManager.list_available_ibm_devices()

@patch('sudoku_nisq.backends.IBMQBackend.available_devices')
@patch('sudoku_nisq.backends.QiskitRuntimeService', side_effect=Exception("QiskitRuntimeService error"))
def test_list_available_ibm_devices_fallback(mock_qiskit_runtime_service, mock_available_devices):
    """Test fallback mechanism when QiskitRuntimeService fails."""
    # Simulate authentication
    BackendManager._ibm_configured = True

    # Mock fallback method to return device info objects
    mock_device = MagicMock()
    mock_device.device_name = "ibm_fallback_device"
    mock_available_devices.return_value = [mock_device]

    devices = BackendManager.list_available_ibm_devices()

    # Ensure QiskitRuntimeService is called first and fails
    mock_qiskit_runtime_service.assert_called_once()
    # Ensure the fallback method is called
    mock_available_devices.assert_called_once_with(device="ibm_brisbane")
    assert devices == ["ibm_fallback_device"]

@patch('sudoku_nisq.backends.IBMQBackend.available_devices', side_effect=Exception("Fallback error"))
@patch('sudoku_nisq.backends.QiskitRuntimeService', side_effect=Exception("QiskitRuntimeService error"))
def test_list_available_ibm_devices_both_fail(mock_qiskit_runtime_service, mock_available_devices):
    """Test when both QiskitRuntimeService and fallback fail."""
    # Simulate authentication
    BackendManager._ibm_configured = True
    
    devices = BackendManager.list_available_ibm_devices()
    
    # Should attempt both methods and return empty list
    mock_qiskit_runtime_service.assert_called_once()
    mock_available_devices.assert_called_once_with(device="ibm_brisbane")
    assert devices == []

@patch('sudoku_nisq.backends.QuantinuumAPI')
@patch('sudoku_nisq.backends.QuantinuumBackend.available_devices')
def test_authenticate_quantinuum_success(mock_available_devices, mock_api):
    """Test successful Quantinuum authentication."""
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    mock_device_info = MagicMock()
    mock_device_info.device_name = "H1-1"
    mock_available_devices.return_value = [mock_device_info]

    devices = BackendManager.authenticate_quantinuum()

    mock_api_instance.login.assert_called_once()
    assert BackendManager._quantinuum_configured
    assert devices == ["H1-1"]

@patch('sudoku_nisq.backends.QuantinuumBackend')
@patch('sudoku_nisq.backends.QuantinuumAPI')
def test_add_quantinuum_device(mock_api, mock_quantinuum_backend):
    """Test adding a Quantinuum device after authentication."""
    BackendManager._quantinuum_configured = True
    
    mock_backend_instance = MagicMock()
    mock_quantinuum_backend.return_value = mock_backend_instance

    backend = BackendManager.add_quantinuum_device("H1-1", alias="my_h1")

    assert backend == mock_backend_instance
    mock_quantinuum_backend.assert_called_once()
    assert BackendManager.is_registered("my_h1")

def test_get_backend_not_found():
    """Test that getting a non-existent backend raises an error."""
    with pytest.raises(ValueError, match="not found"):
        BackendManager.get("non_existent")

    # Add one backend and test the error message again
    BackendManager._backends["some_backend"] = MagicMock()
    with pytest.raises(ValueError, match=r"Available backends: \['some_backend'\]"):
        BackendManager.get("non_existent")

def test_remove_backend():
    """Test removing a backend."""
    mock_be = MagicMock()
    BackendManager._backends["to_remove"] = mock_be
    
    assert BackendManager.is_registered("to_remove")
    BackendManager.remove("to_remove")
    assert not BackendManager.is_registered("to_remove")

def test_info_method():
    """Test the info method for summarizing backends."""
    mock_be1 = MagicMock()
    mock_be1.device_name = "device1"
    mock_be2 = MagicMock()
    # Simulate a backend without device_name but with name
    del mock_be2.device_name
    mock_be2.name = "device2"
    
    BackendManager._backends['be1'] = mock_be1
    BackendManager._backends['be2'] = mock_be2

    info = BackendManager.info()
    
    assert "be1" in info
    assert "be2" in info
    assert info['be1']['device'] == "device1"
    assert info['be2']['device'] == "device2"
    assert info['be1']['type'] == "MagicMock"

@patch('sudoku_nisq.backends.BackendManager.authenticate_ibm')
@patch('sudoku_nisq.backends.BackendManager.add_ibm_device')
def test_init_ibm_one_step(mock_add_device, mock_authenticate):
    """Test the one-step init_ibm method."""
    alias = BackendManager.init_ibm("token", "instance", "device", alias="my_device")
    
    mock_authenticate.assert_called_once_with("token", "instance")
    mock_add_device.assert_called_once_with("device", "my_device")
    assert alias == "my_device"

def test_init_ibm_alias_exists():
    """Test that init_ibm fails if the alias already exists."""
    BackendManager._backends["existing_alias"] = MagicMock()
    with pytest.raises(ValueError, match="already exists"):
        BackendManager.init_ibm("token", "instance", "device", alias="existing_alias")

@patch('sudoku_nisq.backends.BackendManager.add_ibm_device')
def test_init_ibm_already_authenticated(mock_add_device):
    """Test init_ibm when IBM is already authenticated."""
    # Simulate already configured state
    BackendManager._ibm_configured = True
    
    alias = BackendManager.init_ibm("token", "instance", "device", alias="my_device")
    
    # Should skip authentication and just add device
    mock_add_device.assert_called_once_with("device", "my_device")
    assert alias == "my_device"

@patch('sudoku_nisq.backends.BackendManager.authenticate_ibm', side_effect=Exception("Auth failed"))
def test_init_ibm_authentication_failure(mock_authenticate):
    """Test init_ibm when authentication fails."""
    with pytest.raises(RuntimeError, match="Failed to initialize IBM backend 'device' as 'device': Auth failed"):
        BackendManager.init_ibm("token", "instance", "device")
    
    mock_authenticate.assert_called_once_with("token", "instance")

@patch('sudoku_nisq.backends.BackendManager.authenticate_ibm')
@patch('sudoku_nisq.backends.BackendManager.add_ibm_device', side_effect=Exception("Device add failed"))
def test_init_ibm_device_add_failure(mock_add_device, mock_authenticate):
    """Test init_ibm when device addition fails."""
    with pytest.raises(RuntimeError, match="Failed to initialize IBM backend 'device' as 'device': Device add failed"):
        BackendManager.init_ibm("token", "instance", "device")
    
    mock_authenticate.assert_called_once_with("token", "instance")
    mock_add_device.assert_called_once_with("device", "device")
