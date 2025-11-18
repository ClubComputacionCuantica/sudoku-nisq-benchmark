# Architecture

This project separates concerns across four layers. This makes it easier to extend with new encodings, SDK circuit builders, and hardware providers.

## Layers and Responsibilities

- Encodings (what to solve)
  - Input: a puzzle (grid, prefilled cells).
  - Output: abstract exact-cover problem (universe elements + subsets).
  - No references to circuits, SDKs, or providers.
  - Example: `ExactCoverEncoding(puzzle)` → `.universe`, `.simple_subsets`, `.pattern_subsets`.

- Circuits (how to solve)
  - Input: encoding (universe, subsets) and solver parameters.
  - Output: SDK-native circuit (PyTKET, Qiskit, TODO: Braket native).
  - No cloud/provider auth or device logic; only circuit assembly.
  - Implementation pattern:
    - `sudoku_nisq.circuits.exact_cover.qiskit_impl.build_exact_cover_circuit(solver_ctx)`
    - `sudoku_nisq.circuits.exact_cover.pytket_impl.build_exact_cover_circuit(solver_ctx)`

- Providers/Backends (where to run)
  - Input: circuit + execution params (shots, optimization level).
  - Output: execution results (counts, job metadata).
  - Responsibilities: auth, device selection, transpilation presets, retries, job submission, result retrieval.
  - Suggested base interface (pseudo):
    ```python
    class BaseProvider(Protocol):
        def authenticate(self, *args, **kwargs) -> None: ...
        def list_available_devices(self, *args, **kwargs) -> list[str]: ...
        def add_device(self, device: str, alias: str, **kwargs) -> Any: ...
        def get_backend(self, alias: str) -> Any: ...
        def run(self, circuit: Any, shots: int, **kwargs) -> dict: ...
        def backend_info(self) -> dict[str, dict[str, Any]]: ...
        def clear_backends(self) -> None: ...
    ```

- Solvers (orchestrate)
  - Input: puzzle (or precomputed encoding), solver params, and a target backend alias.
  - Flow: build encoding → build circuit → submit to provider → post-process counts to solution.
  - Delegates circuit construction to `circuits`, provider calls to `providers`.

## Dependency Direction

```
encodings  →  circuits  →  providers
      \           ^
       \__________|     (solvers orchestrate across all)
```

- `encodings` has no dependency on circuits/providers.
- `circuits` depends on encoding outputs (universe, subsets).
- `providers` depend on SDK-native circuits only.
- `solvers` coordinate everything end-to-end.

## Error Handling & Results

- Providers normalize result shapes: at minimum `{ "counts": dict[str,int], "metadata": dict }`.
- Solvers convert raw counts to Sudoku solutions and attach timing/job metadata.

## Extension Points

- Add a new encoding: implement a class exposing `.universe` and `.subsets` mapping(s).
- Add a new circuit backend: create a builder in `sudoku_nisq.circuits.<problem>.<sdk>_impl`.
- Add a new provider: subclass the provider base, implement auth/list/add/get/run/info/clear.
- Add a new solver: combine an encoding + circuit builder + provider submission, with resource estimation.

## TODOs

- [ ] Promote a formal `BaseProvider` Protocol in `sudoku_nisq.providers.base`.
- [ ] Add Braket-native circuit builder instead of PyTKET fallback.
- [ ] Document result normalization contract and error taxonomy.
- [ ] Add sequence diagrams (encoding → circuit → provider → solver) for a typical run.
