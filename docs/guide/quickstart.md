# Quickstart

A minimal end-to-end example using the exact cover solver and local simulation.

```{note}
Start with 2x2 or 4x4 puzzles. Larger puzzles (9x9 and beyond) are currently unsolvable on any existing quantum hardware due to circuit size and qubit requirements.
```

```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

# 1) Create a 2x2 puzzle (simulatable on local machines)
puzzle = QSudoku.generate(size=2, num_missing_cells=2, subgrid_size=1)

# 2) Choose a solver
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

# 3) Build circuit and run locally (Aer)
circuit = puzzle.build_circuit()
result = puzzle.run_aer(shots=256)
print("counts:", result.get_counts())

# 4) Plot counts (optional)
puzzle.counts_plot(result, backend_alias="Local", shots=256)

# 5) Resource summary (from metadata)
summary = puzzle.report_resources()
print("summary keys:", summary.keys())
```

## 4x4 Puzzle Example
```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

# Use a pre-defined 4x4 board with 2x2 subgrids
board_4x4 = [
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 4, 0, 0],
    [2, 0, 0, 3],
]

puzzle = QSudoku.from_board(board_4x4)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

circuit = puzzle.build_circuit()
result = puzzle.run_aer(shots=512)
print("counts:", result.get_counts())
```

## Running on a hardware backend (outline)
```python
# IBM Quantum example (outline)
# Use 4x4 puzzles for hardware - 2x2 may work on some systems
# alias = puzzle.init_ibm(api_token="<token>", instance="<crn>", device="ibm_brisbane")
# result = puzzle.run(alias, opt_level=1, shots=1024)

# Braket example (outline)
# Rebuild circuit with sdk="braket" for native construction if desired:
# braket_circuit = puzzle.build_circuit(sdk="braket")
# Submit via your Braket workflow (execution handled outside QSudoku helper for now).
```

```{note}
- For 2x2 puzzles with no real subgrids, use `QSudoku.generate(size=2, subgrid_size=1, ...)`.
- For 4x4 puzzles with proper 2x2 subgrids, use `QSudoku.generate(size=4, ...)` or load from board.
- For Quantinuum via TKET, see `QSudoku.init_quantinuum(...)`.
- The docs mock SDKs during build; you'll need the real SDKs for execution.
- Native Braket circuit builder is available via `build_circuit(sdk="braket")`; transpilation occurs server-side.
```
