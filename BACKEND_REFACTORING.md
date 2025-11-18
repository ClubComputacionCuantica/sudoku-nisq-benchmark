# BackendManager Refactoring - Provider Pattern

This document explains the refactored BackendManager that uses a provider pattern for clean separation of quantum computing provider implementations.

## Overview

The new architecture separates provider-specific code into individual provider classes while maintaining a unified interface through the `BackendManager`. This makes it easy to add new quantum computing providers without modifying existing code.

## Architecture

```
BackendManager (Unified Interface)
├── IBMProvider (IBM Quantum)
├── QuantinuumProvider (Quantinuum)
└── [Your Custom Provider]
```

### Core Components

1. **`QuantumProvider` (Abstract Base Class)**: Defines the interface all providers must implement
2. **`IBMProvider`**: Implementation for IBM Quantum devices
3. **`QuantinuumProvider`**: Implementation for Quantinuum devices  
4. **`BackendManager`**: Unified manager that orchestrates all providers

## Usage Examples

### Basic Usage (Instance-based)

```python
from sudoku_nisq.backends import BackendManager

# Create manager instance
manager = BackendManager()

# Authenticate with providers
manager.authenticate_ibm(api_token="token", instance="instance")
manager.authenticate_quantinuum()

# Add devices
manager.add_backend("ibm", "ibm_brisbane", alias="my_ibm")
manager.add_backend("quantinuum", "H1-1", alias="my_quantinuum")

# Use backends uniformly
ibm_backend = manager.get("my_ibm")
quantinuum_backend = manager.get("my_quantinuum")
```

### Convenience Methods

```python
# One-step initialization
manager.init_ibm("ibm_brisbane", api_token="token", instance="instance")
manager.init_quantinuum("H1-1")

# List available devices
ibm_devices = manager.list_available_ibm_devices()
quantinuum_devices = manager.list_available_quantinuum_devices()
```

## Adding New Providers

### Step 1: Create Provider Class

```python
from sudoku_nisq.providers.base import QuantumProvider

class MyProvider(QuantumProvider):
    @property
    def provider_name(self) -> str:
        return "myprovider"
    
    def authenticate(self, **kwargs) -> List[str]:
        # Implement authentication logic
        self._configured = True
        return ["device1", "device2"]
    
    def list_available_devices(self, **kwargs) -> List[str]:
        # Implement device listing
        return ["device1", "device2"]
    
    def add_device(self, device: str, alias: Optional[str] = None, **kwargs):
        # Implement device initialization
        backend = MyBackend(device)
        name = alias or device
        self._backends[name] = backend
        return backend
```

### Step 2: Register Provider

```python
# Register with manager
manager = BackendManager()
manager.register_provider(MyProvider())

# Use like any other provider
manager.add_backend("myprovider", "device1", alias="my_device")
backend = manager.get("my_device")
```

## Migration from Old BackendManager

### Old Way (Class Methods)
```python
# Old static/class-based approach
BackendManager.authenticate_ibm(token="token")
BackendManager.add_ibm_device("ibm_brisbane")
backend = BackendManager.get("ibm_brisbane")
```

### New Way (Instance-based)
```python
# New instance-based approach
manager = BackendManager()
manager.authenticate_ibm(api_token="token")
manager.add_ibm_device("ibm_brisbane")
backend = manager.get("ibm_brisbane")
```

## Benefits

1. **Clean Separation**: Provider-specific code is isolated
2. **Easy Extension**: Add new providers without changing existing code
3. **Unified Interface**: Single API for all quantum providers
4. **Better Testing**: Each provider can be tested independently
5. **Maintainability**: Changes to one provider don't affect others

## File Structure

```
src/sudoku_nisq/
├── backends.py                 # BackendManager
├── providers/
│   ├── __init__.py
│   ├── base.py                # QuantumProvider abstract class
│   ├── ibm.py                 # IBM implementation
│   ├── quantinuum.py          # Quantinuum implementation
│   └── aws_example.py         # Example new provider
└── example_usage.py           # Usage examples
```

## Provider Requirements

Each provider must implement:

- `provider_name` property: Unique identifier
- `authenticate(**kwargs)`: Set up credentials and return available devices
- `list_available_devices(**kwargs)`: List available devices
- `add_device(device, alias, **kwargs)`: Add a device backend

Optional methods you can override:
- `get_backend(alias)`: Custom backend retrieval logic
- `backend_info()`: Custom info formatting
- Provider-specific convenience methods

## Advanced Usage

### Direct Provider Access

```python
# Access provider directly for advanced features
ibm_provider = manager.get_provider("ibm")
devices = ibm_provider.list_available_devices()

# Provider-specific methods
ibm_provider.some_ibm_specific_method()
```

### Multiple Manager Instances

```python
# Different configurations for different projects
production_manager = BackendManager()
development_manager = BackendManager()

# Each maintains separate backend registries
```

This new architecture provides a solid foundation for scaling to many quantum computing providers while keeping the code organized and maintainable.