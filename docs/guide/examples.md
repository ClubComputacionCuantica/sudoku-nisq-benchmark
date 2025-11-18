# Examples

A few small, runnable patterns you can adapt.

## Generate, solve locally, and visualize
```python
from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

p = QSudoku.generate(subgrid_size=3, num_missing_cells=30)
p.set_solver(ExactCoverQuantumSolver, encoding="simple")

c = p.build_circuit()
res = p.run_aer(shots=512)
p.counts_plot(res, backend_alias="Local", shots=512)
```

## Use an existing board
```python
board_4x4 = [
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 4, 0, 0],
    [2, 0, 0, 3],
]

p = QSudoku.from_board(board_4x4)
# choose solver etc.
```

## Transpile-only (analyze resources per level)
```python
alias = "brisbane"  # after init_ibm(...)
res0 = p.transpile(alias, opt_level=0)
res2 = p.transpile(alias, opt_level=2)
print(res0.get("n_gates"), res2.get("n_gates"))
```

```{todo}
Add a worked hardware example once test credentials and device availability are standardized.
```
