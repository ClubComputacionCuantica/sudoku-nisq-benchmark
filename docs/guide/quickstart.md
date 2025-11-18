# Quickstart

A minimal end-to-end example using the exact cover solver and local simulation.

```python
from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

# 1) Create a 9x9 puzzle (~20 blanks)
puzzle = QSudoku.generate(subgrid_size=3, num_missing_cells=20)

# 2) Choose a solver
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

# 3) Build circuit and run locally (Aer-like)
#    Tip: use run() with a real backend alias once initialized
circuit = puzzle.build_circuit()
result = puzzle.run_aer(shots=256)
print("counts:", result.get("counts"))

# 4) Plot counts (optional)
puzzle.counts_plot(result, backend_alias="Local", shots=256)

# 5) Resource summary (from metadata)
summary = puzzle.report_resources()
print("summary keys:", summary.keys())
```

## Running on a hardware backend (outline)
```python
# IBM Quantum example (outline)
# alias = puzzle.init_ibm(api_token="<token>", instance="<crn>", device="ibm_brisbane")
# result = puzzle.run(alias, opt_level=1, shots=1024)
```

```{note}
- For Quantinuum via TKET, see `QSudoku.init_quantinuum(...)`.
- The docs mock SDKs during build; you’ll need the real SDKs for execution.
```
