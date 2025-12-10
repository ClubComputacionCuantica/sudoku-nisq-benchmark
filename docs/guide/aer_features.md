# Qiskit Aer Integration

This guide covers the comprehensive Qiskit Aer integration in sudoku-nisq-benchmark, providing access to Aer's full simulation capabilities including multiple simulation methods, noise models, GPU acceleration, and device emulation.

## Overview

Qiskit Aer is a high-performance quantum circuit simulator that goes far beyond basic statevector simulation. It provides:

- **Multiple simulation methods**: statevector, density_matrix, stabilizer, MPS, unitary, superop
- **Noise modeling**: Custom noise models and device emulation
- **GPU acceleration**: When qiskit-aer-gpu is installed
- **Precision control**: Single or double precision floating point
- **Performance tuning**: Threading, blocking, and parallelization options

The sudoku-nisq-benchmark module now provides full access to these capabilities through a clean, integrated API.

## Installation

Basic installation (Aer included by default):

```bash
pip install sudoku-nisq-benchmark
```

The package depends on `qiskit-aer`, so CPU Aer simulation works out of the box.

For GPU support (Linux x86_64 only):

```bash
# CUDA 12
pip install qiskit-aer-gpu

# CUDA 11
pip install qiskit-aer-gpu-cu11
```

Note: GPU builds are optional, mutually exclusive, and require a working NVIDIA CUDA installation. On Windows and macOS, use CPU Aer.

## Quick Start

### Basic Ideal Simulation

```python
from sudoku_nisq import QSudoku
from sudoku_nisq.solvers import ExactCoverQuantumSolver

# Create puzzle
puzzle = QSudoku.generate(size=2, num_missing_cells=4)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")

# Run on Aer with default settings (automatic method selection)
result = puzzle.run_aer(shots=1024)
print(f"Measured {len(result.get_counts())} unique outcomes")
```

### Decoding and Display

Convert counts into assignments and an optional filled board:

```python
formatted = puzzle.format_result(result)
print(f"Success rate: {formatted['success_rate']:.1%}")
top = formatted['solutions'][0]
print("Top assignments:", top['assignments'][:5])
if top['board'] is not None:
    print("Filled board preview:", top['board'][:2])
```

For generic exact cover (no Sudoku board), you can inspect subset selections:

```python
counts = result.get_counts()
bitstring, _ = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0]
bs = bitstring if isinstance(bitstring, str) else ''.join(str(b) for b in bitstring)
decoded = puzzle._solver.decode_bitstring(bs)
print("Selected indices:", decoded['selected_indices'])
print("Selected subsets (sample):", list(decoded['selected_subsets'].items())[:3])
```

### Exact Cover Quick Demo (no Sudoku)

```python
from sudoku_nisq.q_exact_cover import QExactCover

# Tiny example problem and circuit
qec = QExactCover.create_small_example()
circuit = qec.build_circuit(sdk="qiskit")
print(f"Circuit qubits: {circuit.num_qubits}, depth: {circuit.depth()}")

# Run on Aer
result = qec.run_aer(shots=512, opt_level=1)
print(f"Outcomes: {len(result['counts'])}")
```

### Specifying Simulation Method

```python
# Statevector simulation (ideal, memory-efficient for small circuits)
result = puzzle.run_aer(
    shots=2048,
    method="statevector",
    optimization_level=2
)

# Density matrix simulation (supports noisy gates)
result = puzzle.run_aer(
    shots=4096,
    method="density_matrix"
)

# Matrix Product State (MPS) for large circuits
result = puzzle.run_aer(
    shots=1024,
    method="matrix_product_state"
)
```

## Simulation Methods

Aer provides several simulation methods optimized for different use cases:

### Automatic Selection

```python
result = puzzle.run_aer(shots=1024, method="automatic")
```

Aer automatically selects the best method based on circuit structure and noise model.

### Statevector Simulation

Ideal for small to medium circuits without noise. Most memory-efficient for pure states.

```python
result = puzzle.run_aer(
    shots=1024,
    method="statevector",
    precision="double"  # or "single" for half memory
)
```

**Best for**: Circuits up to ~25 qubits, ideal simulation, fast execution.

```{note}
Statevector simulation memory grows as 2^n. Check your puzzle's qubit requirement with `resource_estimation()` before running.
```

### Density Matrix Simulation

Required for accurate noisy simulation. Supports mixed states and decoherence.

```python
result = puzzle.run_aer(
    shots=4096,
    method="density_matrix",
    noise_model=my_noise_model
)
```

**Best for**: Noisy simulation, circuits up to ~15 qubits (scales as 2^(2n)).

```{note}
Density matrix requires significantly more memory than statevector. Use for noisy simulations only.
```

### Stabilizer Simulation

Fast simulation for Clifford circuits (gates: H, S, CNOT, CZ, SWAP).

```python
result = puzzle.run_aer(
    shots=1024,
    method="stabilizer"
)
```

**Best for**: Clifford-only circuits, very fast, scales to 1000+ qubits.

### Matrix Product State (MPS)

Tensor network simulation for circuits with low entanglement.

```python
result = puzzle.run_aer(
    shots=2048,
    method="matrix_product_state",
    blocking_enable=True,
    blocking_qubits=5
)
```

**Best for**: Large circuits with structured entanglement, 50+ qubits possible.

### Extended Stabilizer

Approximate simulation for Clifford + T circuits.

```python
result = puzzle.run_aer(
    shots=1024,
    method="extended_stabilizer"
)
```

**Best for**: Clifford + T circuits, good accuracy-speed tradeoff.

## Noise Modeling

### Custom Noise Models

Create custom noise models with specific error rates:

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Build noise model
noise = NoiseModel()

# Add single-qubit gate errors (1% depolarizing)
noise.add_all_qubit_quantum_error(
    depolarizing_error(0.01, 1),
    ['u1', 'u2', 'u3', 'h', 's', 't']
)

# Add two-qubit gate errors (2% depolarizing)
noise.add_all_qubit_quantum_error(
    depolarizing_error(0.02, 2),
    ['cx', 'cz', 'swap']
)

# Add readout errors (5% bit flip)
from qiskit_aer.noise import ReadoutError
readout_error = ReadoutError([[0.95, 0.05], [0.05, 0.95]])
noise.add_all_qubit_readout_error(readout_error)

# Run with noise
result = puzzle.run_aer(
    shots=8192,
    method="density_matrix",
    noise_model=noise
)
```

### Device Emulation

Emulate real quantum hardware by generating noise from device characteristics:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel

# Get real device
service = QiskitRuntimeService()
real_backend = service.backend("ibm_brisbane")

# Generate noise model from device
noise = NoiseModel.from_backend(real_backend)

# Run simulation with realistic noise
result = puzzle.run_aer(
    shots=4096,
    method="density_matrix",
    noise_model=noise,
    coupling_map=real_backend.coupling_map,
    basis_gates=real_backend.configuration().basis_gates,
    optimization_level=2
)
```

### Convenience Method for Device Noise

Use the built-in convenience method:

```python
# Automatically generate noise from device name
result = puzzle.run_aer_with_noise(
    shots=8192,
    device_name="ibm_brisbane",
    optimization_level=2
)

# Or with custom noise model
result = puzzle.run_aer_with_noise(
    shots=4096,
    noise_model=my_custom_noise
)
```

## Backend Manager Integration

### Registering Aer Simulators

```python
from sudoku_nisq.backends import BackendManager

manager = BackendManager.inst()

# Basic statevector simulator
alias = manager.init_aer(
    device="statevector",
    alias="aer_sv"
)

# Noisy density matrix simulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

noise = NoiseModel()
noise.add_all_qubit_quantum_error(
    depolarizing_error(0.01, 2), ['cx']
)

alias = manager.init_aer(
    device="density_matrix",
    noise_model=noise,
    alias="noisy_sim"
)

# Use registered backend
backend = manager.get(alias)
result = puzzle.run(alias, opt_level=1, shots=2048)
```

### Direct QSudoku Integration

```python
# Initialize and attach in one step
alias = puzzle.init_aer(
    method="density_matrix",
    device="GPU",  # Use GPU if available
    precision="single",
    alias="gpu_sim"
)

# Run using the alias
result = puzzle.run(alias, opt_level=1, shots=4096)
```

## GPU Acceleration

### Requirements

- Linux x86_64
- NVIDIA GPU with CUDA support
- qiskit-aer-gpu package

### Installation

```bash
# CUDA 12
pip install qiskit-aer-gpu

# CUDA 11
pip install qiskit-aer-gpu-cu11
```

### Usage

```python
# Check GPU availability
from sudoku_nisq.providers import AerProvider

provider = AerProvider()
info = provider.query_available_devices()
print(f"GPU available: {info['has_gpu']}")
print(f"Devices: {info['devices']}")

# Run on GPU with single precision
result = puzzle.run_aer(
    shots=2048,
    method="statevector",
    device="GPU",
    precision="single"  # Single precision uses half the GPU memory
)
```

GPU acceleration provides significant speedup for:
- Statevector simulation (2-10x faster)
- Density matrix simulation (2-10x faster)
- Unitary simulation (2-10x faster)

## Precision Control

### Double Precision (Default)

```python
result = puzzle.run_aer(
    shots=1024,
    method="statevector",
    precision="double"
)
```

- Full 64-bit floating point
- Maximum accuracy
- Standard memory usage

### Single Precision

```python
result = puzzle.run_aer(
    shots=1024,
    method="statevector",
    precision="single"
)
```

- 32-bit floating point
- Half the memory usage
- 2x faster in many cases
- Good accuracy for most applications

## Performance Tuning

### Threading Control

```python
result = puzzle.run_aer(
    shots=1024,
    method="statevector",
    max_parallel_threads=4  # Limit OpenMP threads
)
```

### Blocking for Large Circuits

```python
result = puzzle.run_aer(
    shots=1024,
    method="statevector",
    blocking_enable=True,
    blocking_qubits=5  # Process 5 qubits at a time
)
```

Blocking reduces memory usage for large circuits by splitting computation into smaller blocks.

### Parallel Experiments

```python
# Run multiple circuits in parallel (useful for batched jobs)
result = puzzle.run_aer(
    shots=1024,
    max_parallel_experiments=4
)
```

### Reproducible Results

```python
# Set seed for reproducibility
result1 = puzzle.run_aer(
    shots=1024,
    method="statevector",
    seed_simulator=42
)

result2 = puzzle.run_aer(
    shots=1024,
    method="statevector",
    seed_simulator=42
)

# Both results will be identical
assert result1.get_counts() == result2.get_counts()
```

## Advanced Examples

### Comparing Methods

```python
import time

methods = ["statevector", "density_matrix", "matrix_product_state"]
results = {}

for method in methods:
    start = time.time()
    result = puzzle.run_aer(
        shots=1024,
        method=method,
        optimization_level=1
    )
    elapsed = time.time() - start
    
    results[method] = {
        "time": elapsed,
        "counts": len(result.get_counts()),
        "result": result
    }
    
    print(f"{method}: {elapsed:.2f}s, {results[method]['counts']} outcomes")
```

### Noise Strength Study

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

error_rates = [0.001, 0.005, 0.01, 0.02]
results = []

for rate in error_rates:
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(rate, 2), ['cx']
    )
    
    result = puzzle.run_aer(
        shots=4096,
        method="density_matrix",
        noise_model=noise
    )
    
    # Analyze solution quality
    counts = result.get_counts()
    results.append({
        "error_rate": rate,
        "unique_outcomes": len(counts),
        "counts": counts
    })
```

### GPU vs CPU Comparison

```python
# Check if GPU is available
provider = AerProvider()
info = provider.query_available_devices()

if info['has_gpu']:
    # Run on CPU
    start = time.time()
    cpu_result = puzzle.run_aer(
        shots=2048,
        method="statevector",
        device="CPU",
        precision="double"
    )
    cpu_time = time.time() - start
    
    # Run on GPU
    start = time.time()
    gpu_result = puzzle.run_aer(
        shots=2048,
        method="statevector",
        device="GPU",
        precision="single"
    )
    gpu_time = time.time() - start
    
    print(f"CPU: {cpu_time:.2f}s")
    print(f"GPU: {gpu_time:.2f}s")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

### Complete Workflow with Error Mitigation

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Create noisy simulation
noise = NoiseModel()
noise.add_all_qubit_quantum_error(
    depolarizing_error(0.01, 2), ['cx']
)

# Run with Zero Noise Extrapolation
result = puzzle.run(
    "aer_noisy",  # Pre-registered noisy backend
    opt_level=2,
    shots=8192,
    use_zne=True  # Enable error mitigation
)

print(f"Raw success probability: {result.success_prob:.4f}")
print(f"Mitigated success probability: {result.mitigated_success_prob:.4f}")
```

## API Reference

### QuantumSolver.run_aer()

```python
result = solver.run_aer(
    shots=1024,
    method="automatic",
    noise_model=None,
    coupling_map=None,
    basis_gates=None,
    device="CPU",
    precision="double",
    optimization_level=1,
    seed_simulator=None,
    max_parallel_threads=None,
    max_parallel_experiments=None,
    blocking_enable=True,
    blocking_qubits=5,
    **backend_options
)
```

### QSudoku.init_aer()

```python
alias = puzzle.init_aer(
    method="automatic",
    noise_model=None,
    coupling_map=None,
    basis_gates=None,
    device="CPU",
    precision="double",
    alias=None,
    **backend_options
)
```

### QSudoku.run_aer_with_noise()

```python
result = puzzle.run_aer_with_noise(
    shots=1024,
    noise_model=None,
    device_name=None,
    method="density_matrix",
    optimization_level=1,
    **aer_options
)
```

### BackendManager.init_aer()

```python
alias = manager.init_aer(
    device="automatic",
    alias=None,
    method=None,
    noise_model=None,
    coupling_map=None,
    basis_gates=None,
    device_type="CPU",
    precision="double",
    **backend_options
)
```

## Best Practices

1. **Choose the right method**:
   - Small circuits (<20 qubits), ideal: `statevector`
   - Noisy simulation: `density_matrix`
   - Large circuits with structure: `matrix_product_state`
   - Clifford circuits: `stabilizer`

2. **Use appropriate shot counts**:
   - Quick tests: 256-512 shots
   - Standard: 1024-2048 shots
   - High precision: 4096-8192 shots
   - Statistical analysis: 10000+ shots

3. **GPU acceleration**:
   - Use `precision="single"` to double GPU memory capacity
   - Best for circuits with 15+ qubits
   - Statevector and density_matrix methods benefit most

4. **Noise modeling**:
   - Use `density_matrix` method for accurate noisy simulation
   - Statevector with noise is approximate but faster
   - Generate device noise models for realistic testing

5. **Optimization levels**:
   - Level 0: No optimization (baseline)
   - Level 1: Light optimization (good default)
   - Level 2: Medium optimization (recommended for real devices)
   - Level 3: Heavy optimization (may be slow to transpile)

## Troubleshooting

### Out of Memory Errors

```python
# Use single precision
result = puzzle.run_aer(method="statevector", precision="single")

# Or use MPS method
result = puzzle.run_aer(method="matrix_product_state")

# Or enable blocking
result = puzzle.run_aer(
    method="statevector",
    blocking_enable=True,
    blocking_qubits=3
)
```

### GPU Not Detected

```bash
# Check installation
python -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"

# Should show: ['CPU', 'GPU'] if GPU support is installed
```

### Slow Simulation

```python
# Reduce optimization level
result = puzzle.run_aer(optimization_level=0)

# Or use faster method
result = puzzle.run_aer(method="statevector")  # Instead of density_matrix

# Or reduce shots
result = puzzle.run_aer(shots=512)  # Instead of 4096
```

## See Also

- [Qiskit Aer Documentation](https://qiskit.github.io/qiskit-aer/)
- [Error Mitigation Guide](error_mitigation.md)
- {ref}`backend-configuration`
- [Examples](examples.md)
