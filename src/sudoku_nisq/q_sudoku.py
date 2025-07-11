from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sudoku import Sudoku as pysudoku
from sudoku_py import SudokuGenerator as sudokupy
from sudoku_nisq.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.graph_coloring_solver import GraphColoringQuantumSolver
from sudoku_nisq.backtracking_solver import BacktrackingQuantumSolver
import json
import hashlib

class Sudoku():
    """
    Sudoku puzzle class supporting multiple generation methods, canonicalization, plotting, and quantum solver initialization.
    
    Attributes:
        subgrid_size (int): Size of the subgrid (e.g., 3 for 9x9 Sudoku).
        board_size (int): Size of the board (subgrid_size * subgrid_size).
        total_cells (int): Total number of cells in the board.
        difficulty (float): Difficulty parameter for puzzle generation (py-sudoku).
        num_missing_cells (int): Number of cells to remove (sudoku-py).
        canonicalize (bool): Whether to canonicalize the puzzle board.
        puzzle: Underlying puzzle object (py-sudoku or sudoku-py).
        open_tuples (list): List of open cell possibilities.
        pre_tuples (list): List of preset (pre-filled) cell tuples.
    
    Methods:
        plot(title=None): Plot the Sudoku grid and return a matplotlib Figure.
        init_exactcover(simple=True, pattern=False): Initialize ExactCoverQuantumSolver.
        init_graphcoloring(): Initialize GraphColoringQuantumSolver.
        init_backtracking(): Initialize BacktrackingQuantumSolver.
        find_preset_tuples(): Return list of preset cell tuples (i, j, value).
        find_open_tuples(): Return list of open cell possibilities (i, j, digit).
        count_solutions(): Recursively count all complete solutions.
        get_hash(): Return the canonical hash of the puzzle.
    """
    def __init__(self, board=[], subgrid_size=2, sudopy=True, num_missing_cells=6, pysudo=False, difficulty=0.4, seed=100, canonicalize=False):
        """
        Initialize a Sudoku puzzle instance.
        
        Args:
            board (list, optional): Custom board as a matrix. If provided, used directly.
            subgrid_size (int, optional): Size of the subgrid (default: 2).
            sudopy (bool, optional): Use sudoku-py generator (default: True).
            num_missing_cells (int, optional): Number of cells to remove (sudoku-py).
            pysudo (bool, optional): Use py-sudoku generator (default: False).
            difficulty (float, optional): Difficulty for py-sudoku (default: 0.4).
            seed (int, optional): Seed for puzzle generation (default: 100).
            canonicalize (bool, optional): Canonicalize the board after generation (default: False).
        """
        self.subgrid_size = subgrid_size
        self.board_size = self.subgrid_size*self.subgrid_size
        self.total_cells = self.board_size * self.board_size
        self.difficulty = difficulty
        self.num_missing_cells = num_missing_cells
        self.canonicalize = canonicalize
        
        # Optionally use 
        #   custom board as a matrix
        #   py-sudoku: allows to generate puzzles from a seed.
        #   sudoku-py: allows to generate puzzles by number of blank cells.
        
        if board:
            self.puzzle = pysudoku(self.subgrid_size,self.subgrid_size,board=board)
        elif pysudo is True:
            self.puzzle = pysudoku(self.subgrid_size,seed=seed).difficulty(self.difficulty)
        elif sudopy is True:
            puzzle = sudokupy(board_size=self.board_size)
            cells_to_remove = self.num_missing_cells
            puzzle.generate(cells_to_remove=cells_to_remove)
            puzzle.board_exchange_values({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9})
            self.puzzle = puzzle
        # Canonicalize the board if requested, regardless of how it was generated
        if canonicalize:
            # For sudoku-py, self.puzzle.board may not be a list of lists of int, so convert if needed
            board_matrix = [list(row) for row in self.puzzle.board]
            canonical_board = canonicalize_puzzle(board_matrix)
            self.puzzle.board = canonical_board

        self.open_tuples = self.find_open_tuples()
        self.pre_tuples = self.find_preset_tuples()
    
    def plot(self,title=None):
        """
        Plot the Sudoku grid using matplotlib.
        
        Args:
            title (str, optional): Title for the plot.
        
        Returns:
            matplotlib.figure.Figure: The matplotlib Figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)
        
        minor_ticks = range(0, self.board_size + 1)
        major_ticks = range(0, self.board_size + 1, int(self.board_size**0.5))
        
        for tick in minor_ticks:
            ax.plot([tick, tick], [0, self.board_size], 'k', linewidth=0.5)
            ax.plot([0, self.board_size], [tick, tick], 'k', linewidth=0.5)
        
        for tick in major_ticks:
            ax.plot([tick, tick], [0, self.board_size], 'k', linewidth=3)
            ax.plot([0, self.board_size], [tick, tick], 'k', linewidth=3)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add numbers to the grid
        for (i, j, value) in self.pre_tuples:
            if value == 0:  # Check if the value is zero
                continue  # Skip the rest of the loop for this iteration
            ax.text(j + 0.5, self.board_size - 0.5 - i, str(value),
                    ha='center', va='center', fontsize=100/self.board_size)

            
        if title:
            plt.title(title, fontsize=20)
        
        plt.close(fig)

        return fig
    
    def init_exactcover(self, simple: bool = True, pattern: bool = False):
        """
        Initialize the ExactCoverQuantumSolver for this Sudoku instance.
        
        Args:
            simple (bool, optional): Use simple encoding (default: True).
            pattern (bool, optional): Use pattern-based encoding (default: False).
        """
        self.exact = ExactCoverQuantumSolver(sudoku=self, simple=simple, pattern=pattern)

    def init_graphcoloring(self):
        """
        Initialize the GraphColoringQuantumSolver for this Sudoku instance.
        """
        self.coloring = GraphColoringQuantumSolver(sudoku=self)
    
    def init_backtracking(self):
        """
        Initialize the BacktrackingQuantumSolver for this Sudoku instance.
        """
        self.backtracking = BacktrackingQuantumSolver(sudoku=self)

    def find_preset_tuples(self):
        """
        Find and return a list of preset (pre-filled) cell tuples.
        
        Returns:
            list: Tuples of the form (i, j, value) for pre-filled cells.
        """
        preset_tuples = []
        for i in range(self.subgrid_size*self.subgrid_size):  # Loop over each row
            for j in range(self.subgrid_size*self.subgrid_size):  # Loop over each column in the row
                element = self.puzzle.board[i][j]
                if element is not None: # Check if the cell is pre-filled
                    preset_tuples.append((i,j,element)) # Store pre-filled cell as tuple
        return preset_tuples

    ## Find open cells and store them in tuples
    def find_open_tuples(self):
        """
        Find and return a list of open cell possibilities.
        
        Returns:
            list: Tuples of the form (i, j, digit) for each possible digit in each empty cell.
        """
        open_tuples = []
        for i in range(self.subgrid_size*self.subgrid_size):  # Loop over each row
            for j in range(self.subgrid_size*self.subgrid_size):  # Loop over each column in the row
                element = self.puzzle.board[i][j]
                if element is None or element == 0: # Check if the cell is empty
                    digits = list(range(1, self.subgrid_size*self.subgrid_size +1)) # Possible digits for the cell
                    # Discard digits based on the column constraint
                    for p in range(self.subgrid_size*self.subgrid_size):
                        if self.puzzle.board[p][j] is not None and self.puzzle.board[p][j] != 0 and self.puzzle.board[p][j] in digits:
                            digits.remove(self.puzzle.board[p][j])
                    # Discard digits based on the row constraint
                    for q in range(self.subgrid_size*self.subgrid_size):
                        if self.puzzle.board[i][q] is not None and self.puzzle.board[i][q] != 0 and self.puzzle.board[i][q] in digits:
                            digits.remove(self.puzzle.board[i][q])
                    # Discard digits based on the subfield
                    subgrid_row_start = self.subgrid_size * (i // self.subgrid_size)
                    subgrid_col_start = self.subgrid_size * (j // self.subgrid_size)
                    for x in range(subgrid_row_start, subgrid_row_start + self.subgrid_size):
                        for y in range(subgrid_col_start, subgrid_col_start + self.subgrid_size):
                            if self.puzzle.board[x][y] is not None and self.puzzle.board[x][y] != 0 and self.puzzle.board[x][y] in digits:
                                digits.remove(self.puzzle.board[x][y])

                    # Store a tuple for each remaining possibility for the given cell
                    for digit in digits:
                        open_tuples.append((i, j, digit))
        return open_tuples

    # The following functions are used to count the solutions of any given puzzle
    # ------------------------------------------------------------
    def _set_cell(self, i, j, value):
        """
        Set the value of a cell in the puzzle board.
        
        Args:
            i (int): Row index.
            j (int): Column index.
            value (int): Value to set.
        """
        self.puzzle.board[i][j] = value
    
    def _is_correct(self):
        """
        Check if the current board is valid (no duplicate values in rows, columns, or subgrids).
        
        Returns:
            bool: True if the board is valid, False otherwise.
        """
        board = self.puzzle.board
        size = self.board_size
        # Check rows for duplicates
        for i in range(size):
            seen = set()
            for j in range(size):
                num = board[i][j]
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        # Check columns for duplicates
        for j in range(size):
            seen = set()
            for i in range(size):
                num = board[i][j]
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        # Check subgrids (blocks) for duplicates
        for block_row in range(0, size, self.subgrid_size):
            for block_col in range(0, size, self.subgrid_size):
                seen = set()
                for i in range(block_row, block_row + self.subgrid_size):
                    for j in range(block_col, block_col + self.subgrid_size):
                        num = board[i][j]
                        if num != 0:
                            if num in seen:
                                return False
                            seen.add(num)
        return True
    
    def _find_empty(self):
        """
        Find the next empty cell in the board.
        
        Returns:
            tuple or None: (i, j) indices of the empty cell, or None if full.
        """
        board = self.puzzle.board
        size = self.board_size
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def count_solutions(self):
        """
        Recursively count all complete solutions for the puzzle using backtracking.
        
        Returns:
            int: Number of complete solutions.
        """
        empty = self._find_empty()
        if not empty:
            return 1  # Found a complete solution
        i, j = empty
        count = 0
        for num in range(1, self.board_size + 1):
            self._set_cell(i, j, num)
            if self._is_correct():
                count += self.count_solutions()
            self._set_cell(i, j, 0)  # Backtrack
        return count

    def get_hash(self):
        """
        Return the canonical hash of the puzzle (SHA-256 of canonicalized board).
        
        Returns:
            str: Hash string.
        """
        return hash_canonical_puzzle(self.puzzle.board, already_canonicalized=self.canonicalize)

# --- Utility functions for canonicalization and hashing ---
def canonicalize_puzzle(matrix: list[list[int]]) -> list[list[int]]:
    """Relabels digits in a Sudoku matrix to ensure canonical form up to permutation.
    
    The function creates a standardized representation of a Sudoku puzzle by relabeling
    the non-zero digits sequentially as they appear, while preserving the relative
    relationships between numbers. This ensures that puzzles that are equivalent up to
    digit permutation will have the same canonical form.
    
    Args:
        matrix (list[list[int]]): A nxn Sudoku matrix where 0 represents empty cells
            and 1-n represent filled cells.
    
    Returns:
        list[list[int]]: A canonicalized version of the input matrix where numbers are
            relabeled according to their first appearance.
    
    Example:
        Input matrix with numbers [2,5,7] would be canonicalized to [1,2,3]
        maintaining their relative positions but using sequential numbering.
        
    Args:
        matrix (list of list of int): Sudoku board matrix.
    
    Returns:
        list of list of int: Canonicalized board matrix.
    """
    mapping = {}  # Maps original numbers to their canonical form
    current = 1   # Next available canonical number
    canonical = []

    for row in matrix:
        new_row = []
        for val in row:
            if val == 0:
                new_row.append(0)  # Preserve empty cells
            else:
                if val not in mapping:
                    # First time seeing this number, assign next sequential value
                    mapping[val] = current
                    current += 1
                new_row.append(mapping[val])
        canonical.append(new_row)

    return canonical

def hash_canonical_puzzle(matrix: list[list[int]], already_canonicalized: bool = False) -> str:
    """
    Generate a unique hash for a Sudoku puzzle in its canonical form.
    
    Args:
        matrix (list of list of int): Sudoku board matrix.
        already_canonicalized (bool, optional): If True, skip canonicalization (default: False).
    
    Returns:
        str: SHA-256 hash string of the canonicalized puzzle.
    """
    if not already_canonicalized:
        canonical = canonicalize_puzzle(matrix)
    else:
        canonical = matrix
    serialized = json.dumps(canonical)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
