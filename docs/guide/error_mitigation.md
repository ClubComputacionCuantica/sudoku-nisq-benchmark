# Error Mitigation for Sudoku NISQ Solver

## Overview

Zero Noise Extrapolation (ZNE) and Probabilistic Error Cancellation (PEC) are integrated by wrapping the Grover search’s bitstring output into a scalar expectation value (success probability). This document reflects the current implementation in `src/sudoku_nisq/mitigation/`.

---

## From Bitstrings to an Expectation Value

We do not measure an observable directly. Instead we define:

f(bitstring) = 1 if the bitstring encodes a valid exact cover
f(bitstring) = 0 otherwise

Then the estimated success probability is the expectation value:
⟨f⟩ = Σ_b P(b) f(b)

Implemented in `mitigation/expectation_wrapper.py`:

```python
def compute_success_expectation(counts, validator):
    if not counts:
        return 0.0
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp = 0.0
    for b, c in counts.items():
        exp += (c / total) * (1.0 if validator(b) else 0.0)
    return exp
```

The solver supplies `solver._is_valid_solution(bitstring)`.

---

## Circuit Flow (Mitiq ↔ pytket ↔ Backend)

Logical Grover circuit (pytket.Circuit)
  ↓ tk_to_qiskit
Qiskit circuit passed to Mitiq (ZNE/PEC)
  ↓ (Mitiq folds / samples) → executor(qiskit_circuit_variant)
Executor:
  • qiskit_to_tk (convert back)
  • backend.process_circuit(pytket_circuit, n_shots=shots)
  • result = backend.get_result(handle)
  • counts = result.get_counts()
  • success_prob = compute_success_expectation(counts, solver._is_valid_solution)
Return success_prob (float in [0,1]) to Mitiq
Mitiq extrapolates / reconstructs mitigated value

Mitiq never sees the backend. Backend never sees Mitiq. The executor is the bridge.

---

## ZNE Usage (Current Implementation)

Prefer the convenience function:

```python
from sudoku_nisq.mitigation.executors import apply_zne

mitigated_prob = apply_zne(
    circuit=pytket_circuit,
    backend=backend,
    solver=solver,
    shots=4096,
    # Optional (currently passthrough, defaults used if None):
    scale_noise=None,
    factory=None,
)
```

Internally:
1. Converts `pytket_circuit` → Qiskit (tk_to_qiskit).
2. Builds an executor via `create_zne_executor`.
3. Calls `mitiq.zne.execute_with_zne(qiskit_circuit, executor, scale_noise=..., factory=...)`.
4. Executor converts each folded Qiskit circuit back to pytket before submission.

Access mitigated value (if run through `QuantumSolver.run(use_zne=True)`):

```python
result._mitigated_success_prob  # float
```

Attribute name matches implementation.

---

## PEC Usage (Skeleton)

```python
from sudoku_nisq.mitigation.executors import apply_pec

mitigated_prob = apply_pec(
    circuit=pytket_circuit,
    backend=backend,
    solver=solver,
    representations=representations,  # Required OperationRepresentation list
    shots=4096,
)
```

Representations must be constructed externally (TODO: auto-generation). Executor does not consume them; `mitiq.pec.execute_with_pec` does.

---

## Executor (Consistent with Code)

```python
def create_zne_executor(backend, solver, shots=1024, **kwargs):
    def executor(circuit):
        from pytket import Circuit
        if not isinstance(circuit, Circuit):
            from qiskit import QuantumCircuit
            if isinstance(circuit, QuantumCircuit):
                from pytket.extensions.qiskit import qiskit_to_tk
                circuit = qiskit_to_tk(circuit)
            else:
                raise TypeError(f"Unsupported circuit type: {type(circuit)}")
        handle = backend.process_circuit(circuit, n_shots=shots, **kwargs)
        result = backend.get_result(handle)
        counts = result.get_counts()
        return compute_success_expectation(counts, solver._is_valid_solution)
    return executor
```

---

## Integration in `QuantumSolver.run`

When `use_zne=True`:
1. Build / retrieve main circuit (pytket).
2. Call `apply_zne(...)` to get mitigated success probability.
3. Perform a standard execution for counts.
4. Attach `result._mitigated_success_prob`.

Similar for `use_pec=True` (mutually exclusive at present).

---

## Backends Supported Now

Backend types usable through this bridge (pytket interface):
- IBM Quantum (via pytket-qiskit)
- Quantinuum (native pytket)
- Aer simulator (pytket-qiskit)

Not yet integrated:
- AWS Braket: requires Qiskit ↔ Braket or direct Braket circuit path (TODO).
- Other providers needing non-pytket native circuits.

---

## Limitations

- Only overall success probability mitigated (not per-bitstring distribution).
- ZNE shot cost grows with number of scale factors; PEC can be much more expensive.
- Combined ZNE+PEC not implemented (TODO).
- Noise scaling / factory selection left at defaults (configurable in future).
- Private method `_is_valid_solution` used; may expose a public `validator` later.

---

## When to Use

Use ZNE:
- Moderate circuits (few hundred gates).
- No detailed noise model available.
- Need quick improvement metric.

Use PEC:
- Small circuits.
- You have a reliable noise model (calibration data).
- Can afford higher sampling overhead.

---

## Future TODOs (Tracked in Code)

- Auto selection of scale factors / factories.
- Braket support (conversion layer).
- Public validator injection instead of private attribute access.
- Combined mitigation pipeline.
- Cache conversions for repeated executes.

---

## References

- Mitiq: https://mitiq.readthedocs.io/
- Expectation wrapper: `src/sudoku_nisq/mitigation/expectation_wrapper.py`
- Executors: `src/sudoku_nisq/mitigation/executors.py`
- Solver integration: `src/sudoku_nisq/quantum_solver.py`