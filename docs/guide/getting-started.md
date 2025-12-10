# Getting Started

This guide walks through solving a Sudoku puzzle using the quantum exact cover solver.

```{note}
Use 2x2 or 4x4 puzzles for realistic quantum execution. Larger puzzles (9x9 and beyond) are largely unsolvable on any existing quantum hardware.
```

## 1. Create or Load a Puzzle
```python
from sudoku_nisq import QSudoku

# For quick simulations, use small instances
puzzle = QSudoku.generate(size=4, num_missing_cells=3)

# For more realistic benchmarks, use 4x4
board_4x4 = [
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 4, 0, 0],
    [2, 0, 0, 3],
]
puzzle = QSudoku.from_board(board_4x4)
```

## 2. Choose Encoding Strategy and Set Solver
Two strategies are available:
- `simple` (default): One subset per (row, col, digit) possibility.
- `pattern`: Row-wise digit placement patterns (can reduce subset count).

```python
from sudoku_nisq import ExactCoverQuantumSolver
solver = puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
```

## 3. Resource Estimation (Optional)
```python
resources = solver.resource_estimation()
print(resources)
```

## 4. Attach a Backend (Simple IBM Flow)
```python
# Replace with your credentials and device
# alias = puzzle.init_ibm(api_token="<token>", instance="<crn>", device="ibm_brisbane")
```

(backend-configuration)=
## Backend Configuration

```{admonition} Backend Configuration
:class: tip
See {doc}`providers` for backend setup details and configuration examples.
```

## 5. Run the Solver
```python
# Local simulator (no credentials)
result = puzzle.run_aer(shots=512)

# Or on hardware/simulator via provider (requires init_ibm above)
# result = puzzle.run(alias, opt_level=1, shots=512)
```

## 6. Visualize and Summarize
```python
puzzle.counts_plot(result, backend_alias="Local", shots=512)
summary = puzzle.report_resources()
```