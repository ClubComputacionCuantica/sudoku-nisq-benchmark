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
@patch('sudoku_nisq.backends.IBMQBackend.available_devices')
def test_authenticate_ibm_success(mock_available_devices, mock_set_config):
    """Test successful IBM authentication."""
    mock_device = MagicMock()
    mock_device.device_name = "ibm_test_device"
    mock_available_devices.return_value = [mock_device]

    devices = BackendManager.authenticate_ibm("fake_token", "fake_instance")

    mock_set_config.assert_called_once_with(ibmq_api_token="fake_token", instance="fake_instance")
    assert BackendManager._ibm_configured
    assert devices == ["ibm_test_device"]

@patch('sudoku_nisq.backends.set_ibmq_config', side_effect=Exception("Auth Error"))
def test_authenticate_ibm_failure(mock_set_config):
    """Test failed IBM authentication."""
    with pytest.raises(Exception, match="Auth Error"):
        BackendManager.authenticate_ibm("fake_token", "fake_instance")
    assert not BackendManager._ibm_configured

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
    with pytest.raises(ValueError, match="Available backends: \['some_backend'\]"):
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
