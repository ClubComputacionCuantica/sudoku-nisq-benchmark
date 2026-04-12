# Quick Start: Stage Metadata Manager Development

**For developers implementing Phases 1-7**

---

## Project Structure

```
src/sudoku_nisq/metadata/
├── __init__.py                   # Package exports
├── base.py                       # StageMetadataManager abstract base
├── config.py                     # MetadataConfig with feature flags
├── instance.py                   # Stage 1: InstanceMetadataManager
├── logical_ir.py                 # Stage 2a: LogicalIRMetadataManager
├── ir_policy.py                  # Stage 2b: IRPolicyMetadataManager
├── compilation.py                # Stage 3: CompilationMetadataManager
├── executable.py                 # Stage 4: ExecutableMetadataManager
├── execution.py                  # Stage 5: ExecutionMetadataManager
└── metrics.py                    # Stage 6-7: MetricsMetadataManager

tests/metadata/
├── __init__.py                   # Config/import tests
├── test_base.py                  # Base class utility tests
├── test_placeholders.py          # Placeholder manager tests
├── test_logical_ir.py            # Phase 1 tests (create during Phase 1)
├── test_compilation.py           # Phase 2 tests (create during Phase 2)
└── ...
```

---

## Implementing a Stage Manager

### Step 1: Replace `NotImplementedError` Stubs

Open the relevant manager file (e.g., `logical_ir.py`) and implement:

```python
def record(self, **kwargs) -> str:
    """Record stage-specific data.
    
    Example for Stage 2a:
    :param solver_name: Solver identifier
    :param encoding: Encoding type
    :param circuit: Circuit object (PyTKET/Qiskit/Braket)
    :returns: circuit_hash
    """
    # 1. Extract metadata from domain objects
    resources = self._extract_resources(circuit)
    
    # 2. Generate unique ID
    circuit_hash = self._compute_circuit_hash(circuit)
    
    # 3. Load existing data
    data = self._load_json()
    
    # 4. Update nested structure
    data = self._set_nested(
        [solver_name, encoding],
        {"circuit_hash": circuit_hash, "resources": resources},
        data
    )
    
    # 5. Save atomically
    self._save_json(data)
    
    return circuit_hash

def query(self, **filters) -> Union[List[Dict], Dict, None]:
    """Query records by filter criteria."""
    data = self._load_json()
    
    # Apply filters and return matches
    results = []
    # ... filtering logic ...
    return results
```

### Step 2: Add Helper Methods

Implement stage-specific extraction logic:

```python
def _extract_resources(self, circuit: Any) -> Dict[str, int]:
    """Auto-extract resources from circuit object."""
    try:
        from pytket import Circuit
        if isinstance(circuit, Circuit):
            return {
                "n_qubits": circuit.n_qubits,
                "n_gates": circuit.n_gates,
                "depth": circuit.depth(),
            }
    except ImportError:
        pass
    
    try:
        from qiskit import QuantumCircuit
        if isinstance(circuit, QuantumCircuit):
            return {
                "n_qubits": circuit.num_qubits,
                "n_gates": sum(circuit.count_ops().values()),
                "depth": circuit.depth(),
            }
    except ImportError:
        pass
    
    raise ValueError(f"Cannot extract resources from {type(circuit)}")
```

### Step 3: Write Tests

Create `tests/metadata/test_<stage>.py`:

```python
import pytest
from pathlib import Path

@pytest.fixture
def manager(tmp_path):
    """Fixture for manager instance."""
    from sudoku_nisq.metadata import LogicalIRMetadataManager
    return LogicalIRMetadataManager(tmp_path, "test_puzzle_hash")

def test_record_creates_storage(manager):
    """Test record() creates storage file."""
    # Mock circuit object
    mock_circuit = MockCircuit(n_qubits=4, n_gates=10, depth=5)
    
    circuit_hash = manager.record(
        solver_name="exact_cover",
        encoding="pattern",
        circuit=mock_circuit
    )
    
    assert circuit_hash is not None
    assert manager.exists()

def test_query_filters_by_solver(manager):
    """Test query() filters by solver_name."""
    # Setup: record multiple circuits
    manager.record(solver_name="solver1", encoding="enc1", circuit=mock1)
    manager.record(solver_name="solver2", encoding="enc2", circuit=mock2)
    
    results = manager.query(solver_name="solver1")
    assert len(results) == 1
    assert results[0]["solver_name"] == "solver1"
```

### Step 4: Update Integration Points

Find where old `MetadataManager` method is called and add feature flag check:

```python
# In quantum_solver.py
def build_main_circuit(self, ...):
    # ... circuit construction ...
    
    if MetadataConfig.ENABLE_NEW_ARCHITECTURE:
        from sudoku_nisq.metadata import LogicalIRMetadataManager
        manager = LogicalIRMetadataManager(self.cache_base, self.puzzle_hash)
        circuit_hash = manager.record(
            solver_name=self.__class__.__name__,
            encoding=self.encoding,
            circuit=circuit,
        )
        self._circuit_hash = circuit_hash  # Store for Stage 3
    else:
        # Legacy path
        self._metadata.set_main_circuit_resources(...)
```

---

## Using Base Class Utilities

### JSON Operations

```python
# Load JSON file
data = self._load_json()

# Save JSON file (atomic)
self._save_json(data)

# Navigate nested dict
value = self._get_nested(["key1", "key2", "key3"], data, default=None)

# Set nested value
data = self._set_nested(["key1", "key2"], "value", data)
```

### JSONL Operations

```python
# Load all records
records = self._load_jsonl()

# Append single record (efficient)
self._append_jsonl({"id": "123", "data": "..."})

# Bulk save (atomic, use for updates)
self._save_jsonl(records)
```

---

## Testing Guidelines

### Test Structure

```python
@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary cache directory."""
    return tmp_path / "test_cache"

@pytest.fixture
def manager(tmp_cache):
    """Manager instance with temp storage."""
    return YourMetadataManager(tmp_cache, "test_hash")

def test_basic_record(manager):
    """Test basic record operation."""
    result = manager.record(param1="value1", param2="value2")
    assert result is not None
    assert manager.exists()

def test_query_empty(manager):
    """Test query on empty storage."""
    results = manager.query()
    assert results == [] or results is None

def test_query_filters(manager):
    """Test query filtering."""
    # Setup: record multiple items
    manager.record(...)
    manager.record(...)
    
    # Test: filter by criteria
    results = manager.query(filter_key="value")
    assert len(results) == expected_count
```

### Running Tests

```bash
# Run all metadata tests
poetry run pytest tests/metadata/ -v

# Run specific stage tests
poetry run pytest tests/metadata/test_logical_ir.py -v

# Check coverage
poetry run pytest tests/metadata/ --cov=src/sudoku_nisq/metadata --cov-report=html
```

---

## Common Patterns

### Pattern 1: Auto-Extraction from Domain Objects

```python
def _extract_from_circuit(self, circuit: Any) -> Dict:
    """Extract metadata from circuit object."""
    # Try each SDK
    for sdk_module, extractor in self._extractors.items():
        try:
            if isinstance(circuit, sdk_module.CircuitType):
                return extractor(circuit)
        except ImportError:
            continue
    
    raise ValueError(f"Unknown circuit type: {type(circuit)}")
```

### Pattern 2: Unique ID Generation

```python
import uuid

def record(self, ...):
    unique_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    # ... store with unique_id ...
    return unique_id
```

### Pattern 3: JSONL Query with Filters

```python
def query(self, **filters) -> List[Dict]:
    """Query with filters."""
    records = self._load_jsonl()
    
    results = []
    for rec in records:
        if all(rec.get(k) == v for k, v in filters.items()):
            results.append(rec)
    
    return results
```

### Pattern 4: Timestamp Recording

```python
from datetime import datetime

def record(self, ...):
    timestamp = datetime.now().isoformat()
    record = {
        "timestamp": timestamp,
        # ... other fields ...
    }
    self._append_jsonl(record)
```

---

## Debugging Tips

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Base class logs warnings for corrupt files
```

### Inspect Storage Files

```bash
# View JSON (pretty-printed)
python -m json.tool .quantum_solver_cache/abc123/stage_2a_logical_ir.json

# View JSONL (one record per line)
cat .quantum_solver_cache/abc123/stage_3_compilation.jsonl | python -m json.tool
```

### Test Feature Flag

```python
from sudoku_nisq.metadata import MetadataConfig

# Enable in tests
MetadataConfig.enable_new_architecture()

# Check status
print(MetadataConfig.ENABLE_NEW_ARCHITECTURE)
```

---

## Performance Considerations

### DO

- ✅ Use `_append_jsonl()` for single record appends (O(1))
- ✅ Use `_load_jsonl()` with filters for small datasets (<1000 records)
- ✅ Implement lazy loading (only load when needed)
- ✅ Use atomic writes for data integrity

### DON'T

- ❌ Load full JSONL for single record lookup (use filters)
- ❌ Save JSON after every field update (batch updates)
- ❌ Store large binary data in JSON (use separate files + references)

---

## Integration Checklist

Before marking phase complete:

- [ ] All `NotImplementedError` stubs replaced
- [ ] Helper methods implemented with SDK-specific logic
- [ ] Comprehensive tests written (>90% coverage)
- [ ] Feature flag integration in calling code
- [ ] Backward compatibility verified (old tests pass)
- [ ] Performance tested with realistic data
- [ ] Documentation updated (docstrings + guides)
- [ ] Code reviewed by team

---

## Getting Help

**Questions?** Check:
1. `docs/internal/architecture/stage_metadata_manager_migration_plan.md` - Full plan
2. `docs/internal/architecture/phase_0_status.md` - Phase 0 details
3. `tests/metadata/test_base.py` - Base class usage examples
4. Existing implementations in earlier phases

**Issues?** File in project tracker with:
- Phase number
- Stage manager affected
- Error message/stack trace
- Minimal reproduction case
