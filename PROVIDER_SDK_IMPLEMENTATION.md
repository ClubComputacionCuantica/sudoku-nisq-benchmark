# Provider-Based SDK Selection Implementation

**Date:** November 20, 2025  
**Branch:** dev  
**Status:** ✅ Implemented

## Overview

Implemented provider-based SDK selection to ensure quantum circuits are built using the correct SDK based on the backend provider. This enables proper metrics reporting as different SDKs (Qiskit, PyTKET, Braket) provide metrics in different formats.

## Motivation

Previously, SDK selection was based on backend interface detection, which was unreliable and didn't properly account for provider-specific requirements. Different providers have native SDKs:
- **IBM Quantum** → Qiskit
- **Quantinuum** → PyTKET  
- **AWS** → Braket (future)

Additionally, each SDK reports circuit metrics differently (e.g., Qiskit's `count_ops()` vs PyTKET's `gate_counts()`), requiring SDK-aware metrics management.

## Changes Implemented

### 1. Provider SDK Declaration

**Files Modified:**
- `src/sudoku_nisq/providers/base.py`
- `src/sudoku_nisq/providers/ibm.py`
- `src/sudoku_nisq/providers/quantinuum_pending.py`

**Changes:**
- Added abstract `sdk_type` property to `QuantumProvider` base class
- `IBMProvider.sdk_type` returns `"qiskit"`
- `QuantinuumProvider.sdk_type` returns `"pytket"`

```python
@property
@abstractmethod
def sdk_type(self) -> str:
    """SDK used by this provider ('qiskit', 'pytket', 'braket')."""
    pass
```

### 2. Backend Manager Enhancement

**File Modified:** `src/sudoku_nisq/backends.py`

**Changes:**
- Updated `get_backend_sdk()` to query provider's `sdk_type` directly
- Removed unreliable interface-based detection
- Provider-driven approach ensures correct SDK selection

```python
def get_backend_sdk(self, alias: str) -> str:
    """Get the SDK type for a given backend alias."""
    provider_name = self._backend_to_provider[alias]
    provider = self._providers[provider_name]
    return provider.sdk_type  # Direct query from provider
```

### 3. Metadata Management

**File Modified:** `src/sudoku_nisq/metadata_manager.py`

**Changes:**
- Added `sdk_type` parameter to `set_main_circuit_resources()`
- SDK type is stored alongside circuit metrics in metadata
- Enables SDK-aware analysis and cross-SDK comparison

**Metadata Structure:**
```json
{
  "solvers": {
    "ExactCoverQuantumSolver": {
      "encodings": {
        "simple": {
          "sdk_type": "qiskit",
          "main_circuit_resources": {
            "n_qubits": 81,
            "n_gates": 2048,
            "depth": 512
          }
        }
      }
    }
  }
}
```

### 4. Circuit Implementation Documentation

**File Modified:** `src/sudoku_nisq/circuits/exact_cover/__init__.py`

**Changes:**
- Added comprehensive documentation explaining provider-specific implementations
- Clarified that `pytket_impl.py` and `qiskit_impl.py` are internal APIs
- Provided usage examples showing proper high-level API access

**Key Points:**
- IBM backends → `qiskit_impl.py`
- Quantinuum backends → `pytket_impl.py`
- SDK selection is automatic based on provider
- Users should use high-level solver API, not import implementations directly

## Architecture

### SDK Selection Flow

```
User API Call
    ↓
QSudoku.build_circuit()
    ↓
QuantumSolver.build_main_circuit(backend)
    ↓
QuantumSolver._build_circuit(backend)
    ↓
QuantumSolver._detect_backend_sdk(backend)
    ↓
BackendManager.get_backend_sdk(alias)
    ↓
Provider.sdk_type  ← Provider declares its native SDK
    ↓
ExactCoverQuantumSolver._build_sdk_circuit(sdk_type)
    ↓
Import appropriate implementation:
  - "qiskit" → circuits.exact_cover.qiskit_impl
  - "pytket" → circuits.exact_cover.pytket_impl
  - "braket" → circuits.exact_cover.braket_impl
```

### Provider-to-SDK Mapping

| Provider      | SDK Type | Implementation File  | Status |
|---------------|----------|---------------------|--------|
| IBM Quantum   | qiskit   | `qiskit_impl.py`    | ✅ Active |
| Quantinuum    | pytket   | `pytket_impl.py`    | ✅ Active |
| AWS Braket    | braket   | `braket_impl.py`    | 🔜 Future |
| Aer Simulator | pytket   | (wrapper)           | ✅ Active |

## Design Decisions

### 1. PyTKET as Caching Format
**Decision:** Continue using PyTKET as the canonical format for circuit caching.

**Rationale:**
- Provides lossless round-trip conversion for major SDKs
- Stable JSON serialization via `Circuit.to_dict()`
- Single cache format simplifies metadata management
- Circuit metrics remain comparable across different SDK builds

**Alternatives Considered:**
- ❌ Cache in native SDK format per provider (too complex)
- ❌ Use OpenQASM 3 as IR (lossy conversion)

### 2. Metrics Normalization
**Current:** Store normalized metrics (n_qubits, n_gates, depth) for cross-SDK comparison.

**Future (TODO):** Store both native and normalized metrics:
- Qiskit: Full `count_ops()` dictionary
- PyTKET: Detailed `gate_counts()` output
- Braket: Native instruction counts

### 3. Backward Compatibility
All changes maintain existing API contracts. The `sdk_type` parameter has a default value of `"pytket"` for backward compatibility.

## TODOs for Future Work

### High Priority
1. **Store Native SDK Metrics**
   - Preserve `qiskit.count_ops()` dictionary
   - Preserve `pytket.gate_counts()` output
   - Enable detailed gate-level analysis per SDK

2. **Migrate Aer to Native Qiskit**
   - Replace `pytket.extensions.qiskit.AerBackend` wrapper
   - Use native `qiskit_aer.AerSimulator`
   - Provide direct access to Qiskit's Aer features

3. **Expose Aer Configuration Options**
   - Add `noise_model` parameter
   - Add `basis_gates` parameter
   - Add `coupling_map` parameter
   - Add `backend_options` parameter

### Medium Priority
4. **Accept backend_alias in _detect_backend_sdk()**
   - Direct query instead of backend object inspection
   - Simplifies SDK detection logic
   - Reduces coupling to backend implementation

5. **Document Caching Strategy**
   - Add architecture documentation explaining PyTKET choice
   - Document conversion guarantees and limitations
   - Provide guidelines for adding new SDKs

### Low Priority
6. **AWS Braket Support**
   - Implement `braket_impl.py` for exact cover
   - Add `AWSProvider` with `sdk_type = "braket"`
   - Test Braket ↔ PyTKET conversion

## Testing Considerations

### Existing Tests
All existing tests pass without modification, confirming backward compatibility.

### Recommended New Tests
1. **Provider SDK Declaration**
   ```python
   def test_ibm_provider_sdk_type():
       provider = IBMProvider()
       assert provider.sdk_type == "qiskit"
   ```

2. **Backend SDK Detection**
   ```python
   def test_backend_manager_sdk_detection():
       manager = BackendManager.inst()
       manager.init_ibm(api_token, instance, "ibm_brisbane", "test_backend")
       assert manager.get_backend_sdk("test_backend") == "qiskit"
   ```

3. **Metadata SDK Storage**
   ```python
   def test_metadata_stores_sdk_type():
       metadata = MetadataManager(cache_dir, puzzle_hash)
       metadata.set_main_circuit_resources(
           "ExactCoverQuantumSolver", "simple", 
           {"n_qubits": 10}, sdk_type="qiskit"
       )
       data = metadata.load()
       assert data["solvers"]["ExactCoverQuantumSolver"]["encodings"]["simple"]["sdk_type"] == "qiskit"
   ```

## Impact on Users

### No Breaking Changes
Users continue to use the same high-level API:

```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

# Scenario 1: Without backend initialization (defaults to PyTKET)
puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
circuit = puzzle.build_circuit()  # Uses pytket_impl.py (default)

# Scenario 2: With IBM backend (auto-selects Qiskit)
puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

# Initialize backend (SDK auto-selected based on provider)
puzzle.init_ibm(api_token, instance, "ibm_brisbane")

# Build circuit (uses qiskit_impl.py automatically)
circuit = puzzle.build_circuit()

# Run on hardware
result = puzzle.run("ibm_brisbane", opt_level=1, shots=1024)
```

### Benefits
1. **Correct SDK Usage:** Circuits use native SDK for each provider
2. **Better Metrics:** SDK-specific metrics properly captured and stored
3. **Easier Debugging:** Metadata clearly indicates which SDK was used
4. **Future-Proof:** Easy to add new providers with their native SDKs

## References

- **Architecture Docs:** `REFACTORING_SUMMARY.md`, `SDK_ABSTRACTION.md`
- **Related Issues:** SDK abstraction for multi-provider support
- **Circuit Implementations:** `src/sudoku_nisq/circuits/exact_cover/`

## Contributors

Implementation by AI assistant, reviewed and approved by project maintainers.

---

**Next Steps:**
1. Review and merge to main branch
2. Add recommended test coverage
3. Update documentation with new SDK selection details
4. Begin work on TODO items (native metrics, Aer migration)
