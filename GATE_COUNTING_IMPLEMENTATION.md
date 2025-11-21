# Gate Counting and Memory Tracking Feature Implementation

## Summary

Added gate type counting and optional memory tracking to circuit construction for PyTKET, Qiskit, and Braket implementations of the exact cover algorithm. 

- **Gate counting** is **enabled by default** and tracks fundamental gates directly during circuit construction.
- **Memory tracking** is **disabled by default** and intended for advanced/development use when profiling or debugging large problem sizes.

## Implementation Details

### 1. GateCounter Class
- Simple helper class with `increment()`, `add_counter()`, and `to_dict()` methods
- Tracks gates with clear naming convention:
  - Single qubit: `H`, `X`, `Measure`
  - Multi-controlled X: `CX` (1 control), `CCX` (2 controls), `C3X` (3 controls), etc.
  - Multi-controlled Z: `CZ` (1 control), `CCZ` (2 controls), `C3Z` (3 controls), etc.

### 2. Circuit Builder Integration

#### PyTKET Implementation (`pytket_impl.py`)
- Added `GateCounter` class
- Modified `_add_multi_control_gate()` to accept and use counter
- Updated all builder functions to track gates:
  - `_build_counter_pytket()`: tracks MCX gates
  - `_build_oracle_pytket()`: tracks X and MCX gates
  - `_build_diffuser_pytket()`: tracks H, X, and CnZ gates
- Main function aggregates counts accounting for Grover iterations
- Returns tuple: `(circuit, gate_counts_dict)`

#### Qiskit Implementation (`qiskit_impl.py`)
- Matching `GateCounter` class implementation
- Updated all builder functions:
  - `_build_counter_qiskit()`: tracks X and MCX gates
  - `_build_oracle_qiskit()`: tracks X and MCX gates
  - `_build_diffuser_qiskit()`: tracks H, X, and Z gates (via H-MCX-H decomposition)
- Returns tuple: `(circuit, gate_counts_dict)`

#### Braket Implementation (`braket_impl.py`)
- Complete `GateCounter` class implementation
- All builder functions track gates:
  - `_build_counter_braket()`: tracks X and MCX gates
  - `_build_oracle_braket()`: tracks X and MCX gates
  - `_build_diffuser_braket()`: tracks H, X, and CnZ gates with optional decomposition
- Supports `decompose_cnz` parameter for consistency with PyTKET
- Returns tuple: `(circuit, gate_counts_dict)`

### 3. Solver Integration

#### ExactCoverQuantumSolver (`exact_cover_solver.py`)
- Added `gate_counts` attribute initialized to `None` (always populated)
- Added `memory_usage` attribute initialized to `None` (only populated if `track_memory=True`)
- Added `track_memory` parameter (default: `False`) for optional memory profiling
- Modified `_build_sdk_circuit()` to handle tuple returns and conditionally track memory
- Stores gate counts after circuit building (always)
- Stores memory stats after circuit building (only if enabled)
- Uses `MemoryTracker` context manager when `track_memory=True`

#### QuantumSolver Base Class (`quantum_solver.py`)
- Added `get_gate_counts()` method for accessing gate counts (always available)
- Added `get_memory_usage()` method for accessing memory statistics (available if enabled)
- Updated `_get_circuit_resources()` to include gate counts (always) and memory usage (if available) in metadata
- Gate counts automatically saved with circuit metadata
- Memory usage saved with metadata only when tracking is enabled

## Usage

### Basic Usage (Gate Counting Only)

```python
from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

# Create puzzle and set solver
puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

# Build circuit (gate counting happens automatically)
circuit = puzzle.build_circuit(sdk="pytket")

# Access gate counts (always available)
gate_counts = puzzle._solver.get_gate_counts()
print(gate_counts)  # {'H': 9, 'X': 5, 'CX': 17, 'C8X': 1, 'Measure': 2}
```

### Advanced Usage (With Memory Tracking)

```python
# Enable memory tracking for development/profiling (disabled by default)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple", track_memory=True)

# Build circuit
circuit = puzzle.build_circuit(sdk="pytket")

# Access memory usage (only available if track_memory=True)
memory_usage = puzzle._solver.get_memory_usage()
if memory_usage:
    print(f"Memory delta: {memory_usage['delta_mb']:.2f} MB")
    print(f"Peak memory: {memory_usage['peak_mb']:.2f} MB")
```

## Testing

Created comprehensive test suite (`test_gate_counting.py`) that verifies:
- ✅ PyTKET gate counting works correctly
- ✅ Qiskit gate counting works correctly
- ✅ Both implementations successfully count gates
- ✅ Gate counts accessible via `get_gate_counts()` method
- ✅ All fundamental gate types are tracked

## Key Design Decisions

1. **Simplicity**: Used simple dictionary-based counter rather than complex class hierarchy
2. **Accuracy**: Count gates during construction (not post-analysis) for infallible accuracy
3. **Query-friendly naming**: Clear convention (CX, CCX, C3X) enables easy filtering and analysis
4. **Default enabled (gates)**: Gate counting active by default since overhead is minimal
5. **Default disabled (memory)**: Memory tracking opt-in for advanced/dev use to avoid production overhead
6. **SDK-agnostic storage**: Store in solver attributes accessible regardless of SDK used
7. **Lightweight profiling**: Memory tracking uses psutil with minimal performance impact when enabled

## Implementation Differences

Small differences exist between SDK gate counts due to decomposition strategies:
- **PyTKET**: Uses `CnZ` gates directly in diffuser (optional `decompose_cnz` parameter for consistency)
- **Qiskit**: Implements controlled-Z as `H + MCX + H` (adds 2 H gates, 1 CX gate per CZ)
- **Braket**: Uses `H + MCX + H` decomposition like Qiskit (supports `decompose_cnz` parameter)

All implementations are functionally equivalent and correctly count the gates they actually use. With `decompose_cnz=True` (default), PyTKET and Braket match Qiskit's gate counts exactly.

## Files Modified

- `src/sudoku_nisq/circuits/exact_cover/pytket_impl.py` - Added GateCounter and gate tracking
- `src/sudoku_nisq/circuits/exact_cover/qiskit_impl.py` - Added GateCounter and gate tracking
- `braket_impl.py` - Complete implementation with GateCounter and gate tracking
- `src/sudoku_nisq/solvers/exact_cover_solver.py` - Handle tuple returns, store gate counts and memory usage
- `src/sudoku_nisq/quantum_solver.py` - Added get_gate_counts() and get_memory_usage() methods, metadata integration
- `src/sudoku_nisq/utils/memory_tracker.py` - New lightweight MemoryTracker utility class

## Files Created

- `src/sudoku_nisq/utils/__init__.py` - Utils package initialization
- `src/sudoku_nisq/utils/memory_tracker.py` - Lightweight memory tracking using psutil
- `test_gate_counting.py` - Comprehensive test suite for gate counting feature
- `example_gate_counting.py` - Example script demonstrating gate counting
- `example_memory_tracking.py` - Example script demonstrating memory tracking

## Future Enhancements

- Add gate counting to graph coloring solver when implemented
- Consider adding gate counts to visualization/plotting functions
- Optionally add gate count comparison utilities for algorithm analysis
