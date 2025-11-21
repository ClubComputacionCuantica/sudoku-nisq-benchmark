# Examples

A few small, runnable patterns you can adapt.

## Generate, solve locally, and visualize
```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

p = QSudoku.generate(size=9, num_missing_cells=30)
p.set_solver(ExactCoverQuantumSolver, encoding="simple")

c = p.build_circuit()
res = p.run_aer(shots=512)
p.counts_plot(res, backend_alias="Local", shots=512)
```

## Use an existing board
```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

board_4x4 = [
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 4, 0, 0],
    [2, 0, 0, 3],
]

p = QSudoku.from_board(board_4x4)
p.set_solver(ExactCoverQuantumSolver)
```

## Transpile and compare metrics by level
```python
# After initializing a backend alias (e.g., via init_ibm)
# alias = p.init_ibm(api_token="<token>", instance="<crn>", device="ibm_brisbane")

_ = p.transpile(alias, opt_level=0)
_ = p.transpile(alias, opt_level=2)

summary = p.report_resources()
ec = summary["solvers"]["ExactCoverQuantumSolver"]["simple"]["backends"][alias]
print("opt0 n_gates:", ec[0]["n_gates"], "opt2 n_gates:", ec[2]["n_gates"])  # keys are ints
```
