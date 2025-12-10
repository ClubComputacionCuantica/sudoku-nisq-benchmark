# Additional Features

This guide covers advanced capabilities for circuit analysis, performance profiling, and resource management.

## Gate Counting

The framework automatically tracks quantum gate usage during circuit construction, providing detailed metrics for algorithm analysis and resource estimation.

### Automatic Gate Tracking

Gate counting is **enabled by default** for all circuit builders and requires no additional configuration:

```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

# Create and solve puzzle
puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

# Build circuit (gate counting happens automatically)
circuit = puzzle.build_circuit(sdk="qiskit")

# Access gate counts
gate_counts = puzzle._solver.get_gate_counts()
print(gate_counts)
# Output: {'H': 45, 'X': 12, 'CX': 8, 'CCX': 4, 'C3X': 2, 'Measure': 9}
```

### Tracked Gate Types

The framework tracks all fundamental quantum gates used in circuit construction:

| Gate Type | Description | Example Count |
|-----------|-------------|---------------|
| `H` | Hadamard gates | Single-qubit superposition |
| `X` | Pauli-X (NOT) gates | Bit flips |
| `CX` | Controlled-X (CNOT) | 1 control qubit |
| `CCX` | Toffoli gate | 2 control qubits |
| `C3X`, `C4X`, ... | Multi-controlled X | 3+, 4+, ... control qubits |
| `CZ` | Controlled-Z | 1 control qubit |
| `CCZ` | Controlled-controlled-Z | 2 control qubits |
| `C3Z`, `C4Z`, ... | Multi-controlled Z | 3+, 4+, ... control qubits |
| `Measure` | Measurement operations | Final readout |

Multi-controlled gates are counted by their control count for precise resource analysis. For example, `C8X` indicates a controlled-X gate with 8 control qubits.

### Gate Counts in Metadata

Gate counts are automatically stored in the circuit metadata and persisted with cached circuits:

```python
# Get resource summary including gate counts
resources = puzzle.report_resources()

solver_data = resources['solvers']['ExactCoverQuantumSolver']
encoding_data = solver_data['encodings']['simple']

print(f"Qubits: {encoding_data['main_circuit_resources']['n_qubits']}")
print(f"Depth: {encoding_data['main_circuit_resources']['depth']}")
print(f"Gate counts: {encoding_data['gate_counts']}")
```

### SDK Consistency

Gate counting is consistent across all SDK implementations (PyTKET, Qiskit, Braket). The framework tracks gates as they are constructed, ensuring accurate counts regardless of the SDK used.

#### PyTKET Decomposition Options

PyTKET supports an optional parameter for controlled-Z gate representation:

```python
# Default: decompose CnZ gates into H + MCX + H for consistency with Qiskit
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=True)

# Alternative: count CnZ as native gates
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=False)
```

With `decompose_cnz=True` (default), PyTKET gate counts match Qiskit exactly. With `decompose_cnz=False`, PyTKET counts native `CnZ` gates.

### Use Cases

**Algorithm Analysis**:
```python
# Compare gate usage across encodings
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
circuit_simple = puzzle.build_circuit()
gates_simple = puzzle._solver.get_gate_counts()

puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")
circuit_pattern = puzzle.build_circuit()
gates_pattern = puzzle._solver.get_gate_counts()

print(f"Simple encoding uses {sum(gates_simple.values())} total gates")
print(f"Pattern encoding uses {sum(gates_pattern.values())} total gates")
```

**Resource Planning**:
```python
# Estimate resource requirements before execution
gate_counts = puzzle._solver.get_gate_counts()

# Count multi-controlled gates (expensive operations)
mcx_gates = sum(count for gate, count in gate_counts.items() 
                if gate.startswith('C') and gate.endswith('X') and gate != 'CX')

print(f"Circuit requires {mcx_gates} multi-controlled X gates")
print(f"Max control count: {max(len(g) - 1 for g in gate_counts if g.startswith('C'))}")
```

## Memory Tracking (Development Feature)

For debugging and profiling large problem instances, the framework includes optional memory tracking that monitors RAM usage during circuit construction.

### Enabling Memory Tracking

Memory tracking is **disabled by default** to avoid performance overhead in production use:

```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

# Enable memory tracking for development/profiling
puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple", track_memory=True)

# Build circuit with memory profiling
circuit = puzzle.build_circuit()

# Access memory statistics
memory_usage = puzzle._solver.get_memory_usage()
if memory_usage:
    print(f"Initial memory: {memory_usage['initial_mb']:.2f} MB")
    print(f"Peak memory: {memory_usage['peak_mb']:.2f} MB")
    print(f"Final memory: {memory_usage['current_mb']:.2f} MB")
    print(f"Memory delta: {memory_usage['delta_mb']:.2f} MB")
```

### Memory Metrics

When enabled, the tracker captures:

- **Initial memory**: RAM usage before circuit construction
- **Peak memory**: Maximum RAM usage during construction
- **Current memory**: RAM usage after construction completes
- **Delta**: Net change in memory usage
- **Snapshots**: Timestamped memory readings at key construction points

### When to Use Memory Tracking

✅ **Use memory tracking for**:
- Debugging memory issues with large circuits
- Profiling circuit construction bottlenecks
- Optimizing algorithm implementations
- Research and development

❌ **Avoid memory tracking for**:
- Production runs (adds overhead)
- Benchmarking (can skew performance metrics)
- Small problems (overhead exceeds benefit)

### Requirements

Memory tracking uses the `psutil` library, which is already included in the project dependencies. No additional installation is required.

## Gate Counting Implementation Details

The framework implements gate counting at the circuit construction level for maximum accuracy:

### Construction-Time Counting

Gates are counted as they are added to the circuit, not through post-hoc analysis:

```python
# Inside circuit builder (simplified)
counter = GateCounter()

# Each gate addition increments the counter
circuit.add_gate(H, [q0])
counter.increment('H')

circuit.add_gate(MCX, control_qubits + [target])
counter.increment(f'C{len(control_qubits)}X')

# Return circuit with counts
return circuit, counter.to_dict()
```

This approach ensures:
- **Accuracy**: Every gate is counted exactly once
- **Consistency**: No ambiguity from circuit optimization or equivalence
- **Performance**: Minimal overhead during construction

### Integration Points

Gate counts flow through the solver architecture:

1. **Circuit Builder** → Returns `(circuit, gate_counts)` tuple
2. **Solver** → Stores counts in `self._gate_counts` attribute
3. **Base Class** → Provides `get_gate_counts()` accessor method
4. **Metadata** → Persists counts with cached circuits

### Example: Custom Gate Analysis

```python
# Build circuit
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
circuit = puzzle.build_circuit()

# Analyze specific gate types
gate_counts = puzzle._solver.get_gate_counts()

# Single-qubit gates
single_qubit = gate_counts.get('H', 0) + gate_counts.get('X', 0)

# Two-qubit gates
two_qubit = gate_counts.get('CX', 0) + gate_counts.get('CZ', 0)

# Multi-qubit gates (3+ qubits involved)
multi_qubit = sum(count for gate, count in gate_counts.items()
                  if gate.startswith('C') and len(gate) > 2)

print(f"Single-qubit: {single_qubit}, Two-qubit: {two_qubit}, Multi-qubit: {multi_qubit}")
```

## Performance Considerations

### Gate Counting Overhead

Gate counting adds **negligible overhead** (<1% in typical cases) since it's a simple counter increment during construction. It's safe to leave enabled for all use cases.

### Memory Tracking Overhead

Memory tracking uses `psutil.Process().memory_info()` which has an overhead, for these reasons, it's opt-in via `track_memory=True`.