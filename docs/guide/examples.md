# Examples

A few small, runnable patterns you can adapt.

```{note}
These examples use 2x2 and 4x4 Sudoku puzzles. Larger puzzles (9x9 and beyond) are currently unsolvable on any existing quantum hardware due to circuit size and qubit requirements. Use 2x2 for quick simulations and 4x4 for more realistic benchmarks.
```

## Generate, solve locally, and visualize (2x2)
```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

# 2x2 puzzle is simulatable on local machines
p = QSudoku.generate(size=2, num_missing_cells=2, subgrid_size=1)
p.set_solver(ExactCoverQuantumSolver, encoding="simple")

c = p.build_circuit()
res = p.run_aer(shots=512)
p.counts_plot(res, backend_alias="Local", shots=512)

# Decode and display
formatted = p.format_result(res)
print(f"Success rate: {formatted['success_rate']:.1%}")
top = formatted['solutions'][0]
print("Top assignments:", top['assignments'][:5])
if top['board'] is not None:
    print("Filled board preview:", top['board'][:2])
```

## Use an existing 4x4 board
```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

# 4x4 puzzle with 2x2 subgrids
board_4x4 = [
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 4, 0, 0],
    [2, 0, 0, 3],
]

p = QSudoku.from_board(board_4x4)
p.set_solver(ExactCoverQuantumSolver, encoding="simple")
```

## Transpile and compare metrics by level (4x4)
```python
# Use 4x4 puzzle for hardware testing
board_4x4 = [
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 4, 0, 0],
    [2, 0, 0, 3],
]

p = QSudoku.from_board(board_4x4)
p.set_solver(ExactCoverQuantumSolver, encoding="simple")

# After initializing a backend alias (e.g., via init_ibm)
# alias = p.init_ibm(api_token="<token>", instance="<crn>", device="ibm_brisbane")

_ = p.transpile(alias, opt_level=0)
_ = p.transpile(alias, opt_level=2)

summary = p.report_resources()
ec = summary["solvers"]["ExactCoverQuantumSolver"]["simple"]["backends"][alias]
print("opt0 n_gates:", ec[0]["n_gates"], "opt2 n_gates:", ec[2]["n_gates"])  # keys are ints
```
