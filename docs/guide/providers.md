# Providers

This framework uses a provider pattern to abstract quantum backend interactions. Each provider encapsulates authentication, device discovery, registration, and (where applicable) job submission.

## Optional Dependencies

The framework includes several quantum providers, but only the **Aer provider** (local simulator) is available by default. To enable additional providers, install their dependencies:

```bash
# IBM Quantum (qiskit-ibm-runtime)
pip install qiskit-ibm-runtime

# Quantinuum via Nexus (qnexus)
pip install qnexus

# AWS Braket (amazon-braket-sdk, boto3)
pip install amazon-braket-sdk boto3
```

Providers are automatically registered when their dependencies are available. You can check which providers are active:

```python
from sudoku_nisq.backends import BackendManager

manager = BackendManager()
print(manager.list_providers())  # ['aer', 'ibm', 'quantinuum', 'aws'] (if all installed)
```

## Using Providers

### Basic Setup

```python
from sudoku_nisq.backends import BackendManager

manager = BackendManager()

# Authenticate IBM (Qiskit runtime)
manager.authenticate_ibm(api_token="your_token", instance="your_instance")

# Add an IBM device
manager.add_backend("ibm", "ibm_brisbane", alias="my_ibm_device")

# Retrieve by alias
backend = manager.get("my_ibm_device")
```

### Convenience Methods

One-step initialization is available for common providers:

```python
# IBM Quantum
manager.init_ibm(
    device="ibm_brisbane",
    alias="ibm_dev",
    api_token="your_token",
    instance="your_instance"
)

# Quantinuum (Nexus)
manager.init_quantinuum(
    device="H1-1",
    alias="qtm_dev"
)

# Aer (local simulator)
manager.init_aer(device="statevector", alias="sim")

# List devices
ibm_devices = manager.list_available_ibm_devices()
qtm_devices = manager.list_available_quantinuum_devices()
aer_methods = manager.list_available_aer_devices()
```

### SDK Type Detection

Each provider declares a native SDK, which is used to select circuit builders:

```python
manager.get_backend_sdk("ibm_dev")  # "qiskit"
manager.get_backend_sdk("qtm_dev")  # "pytket"
manager.get_backend_sdk("sim")      # "qiskit"
```

## Provider Interface (Summary)

All providers implement `QuantumProvider`:

- `provider_name`: unique id (e.g., `ibm`, `quantinuum`, `aer`).
- `sdk_type`: SDK string (`qiskit`, `pytket`, `braket`).
- `authenticate(**kwargs) -> List[str]`: configure credentials; returns device identifiers.
- `list_available_devices(**kwargs) -> List[str]`: enumerate devices/methods.
- `add_device(device, alias=None, **kwargs)`: register a backend under an alias.
- `init_device(device, alias=None, **kwargs) -> str`: authenticate if needed, then add.

## Provider Notes (High Level)

### Aer (AerProvider)

- SDK: Qiskit (local simulator).
- Auth: none; always available.
- Devices: simulation methods (e.g., `statevector`, `density_matrix`).
- Options: noise models, GPU, precision, and advanced backend settings.

### AWS Braket (AWSProvider)

- SDK: Amazon Braket.
- Status: Template implementation (requires `amazon-braket-sdk` and `boto3`).
- Installation: `pip install amazon-braket-sdk boto3`
- Note: This provider is optional and only available when AWS dependencies are installed.

### IBM Quantum (IBMProvider)

- SDK: Qiskit (native runtime backends).
- Auth: API token and optional instance (CRN).
- Devices: `service.backends()`; add via `service.backend(name)`.

### Quantinuum (QuantinuumProvider)

- SDK: PyTKET (qnexus optional).
- Auth: `qnexus` login (browser or interactive) and set active project.
- Devices: discovered via Nexus; add stores a Nexus config and optionally a PyTKET backend.
- Jobs: compile → execute model via Nexus (advanced usage; not required for simple listing/registration).

## Architecture

Providers implement a common interface and are orchestrated by `BackendManager`:

```
QuantumProvider (Abstract Base Class)
├── IBMProvider (IBM Quantum) – Qiskit
├── QuantinuumProvider (Quantinuum Nexus) – PyTKET
├── AerProvider (Qiskit Aer simulator) – Qiskit
└── AWSProvider (Amazon Braket) – Braket
```

`BackendManager` maintains a unified registry of backend aliases across providers.