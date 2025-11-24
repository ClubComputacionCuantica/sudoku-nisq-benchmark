from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.puzzle import SudokuPuzzle
from pytket.extensions.qiskit import AerBackend
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverSolver

puzzle = SudokuPuzzle([[0, 0, 3, 0], [0, 0, 0, 2], [1, 0, 0, 0], [0, 4, 0, 0]], block_rows=2, block_cols=2)
q = QSudoku(puzzle)
q.set_solver(ExactCoverSolver, encoding='binary')
backend = AerBackend()
q._attached_backends['aer'] = backend

print("Backend attached:", 'aer' in q._attached_backends)
print("Solver set:", q._solver is not None)

try:
    result = q.get_transpiled_circuit('aer', opt_level=0)
    print('Result type:', type(result))
    print('ERROR: Should have raised FileNotFoundError!')
except FileNotFoundError as e:
    print('SUCCESS: Raised FileNotFoundError:', e)
except Exception as e:
    print('Other error:', type(e).__name__, str(e))
