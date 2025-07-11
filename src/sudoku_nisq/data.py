
from sudoku_nisq.q_sudoku import Sudoku
from sudoku_nisq.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.graph_coloring_solver import GraphColoringQuantumSolver
from sudoku_nisq.backtracking_solver import BacktrackingQuantumSolver
import pandas as pd

def puzzle_generator(n, num_missing_cells=9, subgrid_size=2):
    """
    Yield multiple Sudoku puzzles with the given number of missing cells and subgrid size.
    Args:
        n (int): Number of puzzles to generate.
        num_missing_cells (int): Number of missing cells in each puzzle.
        subgrid_size (int): Size of the Sudoku subgrid.
    Yields:
        Sudoku: A canonical Sudoku puzzle instance.
    """
    for _ in range(n):
        yield Sudoku(subgrid_size=subgrid_size, num_missing_cells=num_missing_cells, canonicalize=True)

def init_solvers(puzzle):
    """Initialize solvers for a given puzzle. Extend as needed."""
    exact_solver = ExactCoverQuantumSolver(sudoku=puzzle, simple=True)

    # graph_solver = GraphColoringQuantumSolver(sudoku=puzzle)
    # backtrack_solver = BacktrackingQuantumSolver(sudoku=puzzle)
    return [exact_solver] #, graph_solver, backtrack_solver]

def extract_resources(solver):
    circuit = solver._get_or_build_main_circuit()
    return {
        'n_qubits': circuit.n_qubits,
        'n_gates': circuit.n_gates,
        'depth': circuit.depth()
    }

def main(num_puzzles=10, num_missing_cells=9, subgrid_size=2):
    import gc
    results = []
    for puzzle in puzzle_generator(num_puzzles, num_missing_cells=num_missing_cells, subgrid_size=subgrid_size):
        solvers = init_solvers(puzzle)
        for solver in solvers:
            resources = extract_resources(solver)
            results.append({
                'puzzle_hash': puzzle.get_hash(),
                'solver_type': solver.__class__.__name__,
                'missing_cells': puzzle.num_missing_cells,
                'n_qubits': resources['n_qubits'],
                'n_gates': resources['n_gates'],
                'depth': resources['depth'],
            })
            # Explicitly clear solver cache to free memory
            solver.clear_cache("all")
        # Optional: clean up references to free memory
        del solvers
        del puzzle
        gc.collect()
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    df = main(num_puzzles=10, num_missing_cells=4, subgrid_size=2)