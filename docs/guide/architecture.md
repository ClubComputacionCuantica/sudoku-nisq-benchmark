# Architecture

This project separates quantum Sudoku solving into four clean layers: encodings, circuits, providers, and solvers. This design keeps responsibilities clear and makes it easy to add new algorithms, SDKs, or hardware backends.

## Layers

### Encodings
**What to solve**: Transform Sudoku puzzles into quantum-ready problem formulations.

- **Input**: `SudokuPuzzle` (grid size, prefilled cells).
- **Output**: Problem-specific representation (e.g., exact-cover: `universe`, `simple_subsets`, `pattern_subsets`).
- **Key point**: No SDK or provider dependencies—pure problem transformation.
- **Implementation**: `sudoku_nisq.encodings.exact_cover_encoding.ExactCoverEncoding`

**Example**:
```python
from sudoku_nisq.encodings import ExactCoverEncoding

encoding = ExactCoverEncoding(puzzle)
print(f"Universe: {len(encoding.universe)} constraints")
print(f"Subsets: {len(encoding.simple_subsets)} options")
```

### Circuits
**How to solve**: Build SDK-native quantum circuits from problem encodings.

- **Input**: Solver context (encoding data + algorithm params like `num_solutions`, `decompose_cnz`).
- **Output**: SDK-native circuit object (`pytket.Circuit`, `qiskit.QuantumCircuit`).
- **Key point**: Pure circuit construction—no authentication, device selection, or execution logic.
- **Implementation pattern**:
  - `sudoku_nisq.circuits.exact_cover.qiskit_impl.build_exact_cover_circuit(solver)`
  - `sudoku_nisq.circuits.exact_cover.pytket_impl.build_exact_cover_circuit(solver)`
- **Side effect**: Returns gate counts for resource analysis.

### Providers
**Where to run**: Manage authentication, device access, and circuit execution across quantum platforms.

- **Input**: Circuit + execution params (shots, optimization level).
- **Output**: Normalized result dict with `counts` and `metadata`.
- **Responsibilities**:
  - Authentication and credential management
  - Device discovery and registration
  - Backend lifecycle (add, get, remove, list)
  - SDK type declaration for circuit compatibility
- **Base interface**: `sudoku_nisq.providers.base.QuantumProvider` (abstract class)
- **Implementations**: 
  - `sudoku_nisq.providers.ibm.IBMProvider` (Qiskit SDK)
  - `sudoku_nisq.providers.quantinuum_pending.QuantinuumProvider` (PyTKET SDK)
  - AWS Braket support: in progress

**Key design**: Providers declare their SDK type via the `sdk_type` property, enabling correct circuit builder selection.

**Backend Manager**: `sudoku_nisq.backends.BackendManager` provides a unified registry for all providers:
```python
from sudoku_nisq.backends import BackendManager

manager = BackendManager.inst()  # Singleton accessor
manager.init_ibm(device="ibm_brisbane", alias="ibm_dev", api_token="...", instance="...")
manager.init_quantinuum(device="H1-1", alias="qtm_dev")

# SDK-aware backend lookup
sdk_type = manager.get_backend_sdk("ibm_dev")  # Returns "qiskit"
backend = manager.get("ibm_dev")
```

### Solvers
**Orchestrate end-to-end**: Coordinate encodings, circuits, and providers to solve puzzles.

- **Base class**: `sudoku_nisq.quantum_solver.QuantumSolver` (abstract)
  - Provides infrastructure: circuit caching, metadata tracking, backend integration
  - Requires subclasses to implement `_build_sdk_circuit(sdk_type)` and `resource_estimation()`
- **Example implementation**: `sudoku_nisq.solvers.exact_cover_solver.ExactCoverQuantumSolver`
  - Uses Grover's algorithm via exact-cover formulation
  - Supports "simple" and "pattern" encodings
  - Delegates to SDK-specific circuit builders based on backend's SDK type

**Flow**:
1. Puzzle → Encoding (problem transformation)
2. Encoding → Circuit (SDK-specific builder selected by backend's SDK type)
3. Circuit → Backend execution (via `BackendManager`)
4. Counts → Solution mapping + metadata

**Integration with infrastructure**:
- **MetadataManager**: Tracks solver performance, caching, resource usage
- **ExperimentRunner**: Orchestrates large-scale benchmarking campaigns with crash-safe progress tracking

## Dependencies

```
encodings  →  circuits  →  providers
      ↓           ↓           ↓
      └───────→ solvers ←─────┘
                  ↓
         (infrastructure: caching, metadata, experiments)
```

- **Encodings**: Self-contained (no circuit/provider/solver deps)
- **Circuits**: Depend on encoding outputs and SDK libraries only
- **Providers**: Depend on SDK libraries (Qiskit, PyTKET, Braket)
- **Solvers**: Orchestrate all layers + use infrastructure services
- **Infrastructure**: `MetadataManager`, `ExperimentRunner` support solvers

## Public API

Core entry points from `sudoku_nisq/__init__.py`:
- `QSudoku`: High-level solver interface
- `ExperimentRunner`: Benchmarking orchestration
- `ExactCoverEncoding`: Encoding implementation
- `ExactCoverQuantumSolver`: Concrete solver
- `BackendManager`: Unified backend registry

## Extension Points

1. **New encoding**: Implement class with `.universe` and subset attributes; update solver to use it
2. **New circuit builder**: Add `sudoku_nisq.circuits.<problem>.<sdk>_impl.build_*_circuit(solver)` function
3. **New provider**: Subclass `QuantumProvider`; implement abstract methods; register in `BackendManager.__init__()`
4. **New solver**: Subclass `QuantumSolver`; implement `_build_sdk_circuit(sdk_type)` and `resource_estimation()`

## Current Status

**Implemented**:
- ✅ Exact-cover encoding (simple + pattern strategies)
- ✅ Qiskit and PyTKET circuit builders
- ✅ IBM Quantum provider with full authentication
- ✅ Quantinuum provider (pending final testing)
- ✅ `BackendManager` with unified multi-provider interface
- ✅ Gate counting and resource estimation
- ✅ Crash-safe experiment runner with progress tracking

**In Progress**:
- 🔄 AWS Braket native circuit builder (currently uses PyTKET fallback)
- 🔄 Quantinuum provider validation and testing

**Planned**:
- 📋 Graph coloring encoding and circuit implementation
- 📋 Result normalization protocol documentation
- 📋 Error taxonomy and exception hierarchy
- 📋 Sequence diagrams for typical execution flows
