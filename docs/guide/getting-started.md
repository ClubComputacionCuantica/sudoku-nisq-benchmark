# Getting Started

This guide walks through solving a Sudoku puzzle using the quantum exact cover solver.

## 1. Create or Load a Puzzle
```python
from sudoku_nisq import QSudoku
puzzle = QSudoku.generate(subgrid_size=3, num_missing_cells=25)
# TODO: Demonstrate loading from a file or predefined pattern.
```

## 2. Choose Encoding Strategy
Two strategies are available:
- `simple` (default): One subset per (row, col, digit) possibility.
- `pattern`: Row-wise digit placement patterns (can reduce subset count).

```python
from sudoku_nisq import ExactCoverQuantumSolver
solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="simple")
```

## 3. Resource Estimation (Optional)
```python
resources = solver.resource_estimation()
print(resources)
# TODO: Interpret resource metrics in docs (n_qubits, MCX_gates, etc.)
```

## 4. Attach a Backend (Provider Abstraction)
```python
# TODO: Replace placeholders with a real flow once auth examples are finalized
# manager = BackendManager.inst()
# manager.authenticate_ibm(api_token="<token>", instance="<instance>")
# manager.add_ibm_device("ibm_brisbane", alias="brisbane")
```

## 5. Run the Solver
```python
# TODO: Confirm public method naming (run_local vs run) after API review
# result = puzzle.run("brisbane", opt_level=1, shots=512)
# print(result["counts"])  # Example processed output
```

## 6. Decode / Verify Solution
```python
# TODO: Add solution verification helper once finalized
# assert puzzle.is_solved()
```

## Next Steps
- Try the `pattern` encoding to compare resource usage.
- Enable verbose logging to inspect circuit build phases. (TODO: add logging docs)
- Run on different providers for comparative benchmarking. (TODO: benchmarking guide)

```{toctree}
:maxdepth: 1

architecture
```

## FAQ (TODO)
- How are subsets constructed?
- How does pattern encoding reduce search space?
- What causes large MCX counts?
- How to add a custom provider?

## Troubleshooting (TODO)
- Authentication failures
- Circuit too large for backend
- No solutions found / multiple solutions ambiguity
