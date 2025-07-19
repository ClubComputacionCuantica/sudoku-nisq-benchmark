from dataclasses import dataclass
import math
import json
import hashlib
from typing import List, Optional
from sudoku_py import SudokuGenerator

@dataclass
class SudokuPuzzle:
    board: List[List[int]]
    subgrid_size: int
    board_size: int
    num_missing_cells: int
    canonicalize: bool = False

    @classmethod
    def from_board(cls, board: List[List[int]], canonicalize: bool = False) -> "SudokuPuzzle":
        """
        Create a SudokuPuzzle instance from an existing board.

        Args:
            board (List[List[int]]): The Sudoku board.
            canonicalize (bool): Whether to canonicalize the board.

        Returns:
            SudokuPuzzle: A new SudokuPuzzle instance.
        """
        subgrid_size = int(math.isqrt(len(board)))
        board_size = subgrid_size * subgrid_size
        num_missing_cells = sum(1 for row in board for cell in row if cell == 0 or cell is None)

        if canonicalize:
            board = cls._canonicalize(board)

        return cls(
            board=board,
            subgrid_size=subgrid_size,
            board_size=board_size,
            num_missing_cells=num_missing_cells,
            canonicalize=canonicalize
        )

    @classmethod
    def generate(cls, subgrid_size: int, num_missing_cells: int, canonicalize: bool = False) -> "SudokuPuzzle":
        """
        Generate a new Sudoku puzzle.

        Args:
            subgrid_size (int): Size of the subgrid (e.g., 3 for 9x9 Sudoku).
            num_missing_cells (int): Number of cells to remove.
            canonicalize (bool): Whether to canonicalize the puzzle.

        Returns:
            SudokuPuzzle: A new SudokuPuzzle instance.
        """
        board_size = subgrid_size * subgrid_size
        generator = SudokuGenerator(board_size=board_size)
        generator.generate(cells_to_remove=num_missing_cells)
        generator.board_exchange_values({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9})
        board = generator.board

        if canonicalize:
            board = cls._canonicalize(board)

        return cls(
            board=board,
            subgrid_size=subgrid_size,
            board_size=board_size,
            num_missing_cells=num_missing_cells,
            canonicalize=canonicalize
        )

    def plot(self, title: Optional[str] = None):
        """
        Plot the Sudoku grid using matplotlib.
        
        Example usage:
        puzzle = SudokuPuzzle.generate(subgrid_size=3, num_missing_cells=20)
        fig = puzzle.plot(title="Sudoku Puzzle")
        fig.savefig("sudoku_plot.png")  # Save the plot to a file

        Args:
            title (str, optional): Title for the plot.

        Returns:
            matplotlib.figure.Figure: The matplotlib Figure object for the plot.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)
        minor_ticks = range(0, self.board_size + 1)
        major_ticks = range(0, self.board_size + 1, self.subgrid_size)
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
            if value == 0:  # Skip empty cells
                continue
            ax.text(j + 0.5, self.board_size - 0.5 - i, str(value),
                    ha='center', va='center', fontsize=100 / self.board_size)

        # Add title if provided
        if title:
            plt.title(title, fontsize=20)
        plt.close(fig)

        return fig
    
    def get_hash(self) -> str:
        """Generate a SHA-256 hash of the puzzle."""
        board_str = json.dumps(self.board)
        return hashlib.sha256(board_str.encode("utf-8")).hexdigest()
    
    @property
    def num_solutions(self) -> int:
        """
        Calculate and return the number of valid solutions for the puzzle.

        This property uses the `_count_solutions` method to perform a backtracking
        search and count all possible solutions.

        Returns:
            int: The number of valid solutions for the puzzle.
        """
        return self._count_solutions()
    
    @property
    def pre_tuples(self):
        """
        Find and return a list of preset (pre-filled) cell tuples.
        
        Returns:
            list: Tuples of the form (i, j, value) for pre-filled cells.
        """
        preset_tuples = []
        for i in range(self.subgrid_size*self.subgrid_size):  # Loop over each row
            for j in range(self.subgrid_size*self.subgrid_size):  # Loop over each column in the row
                element = self.board[i][j]
                if element is not None: # Check if the cell is pre-filled
                    preset_tuples.append((i,j,element)) # Store pre-filled cell as tuple
        return preset_tuples

    @property
    def open_tuples(self):
        """
        Find and return a list of open cell possibilities.
        
        Returns:
            list: Tuples of the form (i, j, digit) for each possible digit in each empty cell.
        """
        open_tuples = []
        for i in range(self.subgrid_size*self.subgrid_size):  # Loop over each row
            for j in range(self.subgrid_size*self.subgrid_size):  # Loop over each column in the row
                element = self.board[i][j]
                if element is None or element == 0: # Check if the cell is empty
                    digits = list(range(1, self.subgrid_size*self.subgrid_size +1)) # Possible digits for the cell
                    # Discard digits based on the column constraint
                    for p in range(self.subgrid_size*self.subgrid_size):
                        if self.board[p][j] is not None and self.board[p][j] != 0 and self.board[p][j] in digits:
                            digits.remove(self.board[p][j])
                    # Discard digits based on the row constraint
                    for q in range(self.subgrid_size*self.subgrid_size):
                        if self.board[i][q] is not None and self.board[i][q] != 0 and self.board[i][q] in digits:
                            digits.remove(self.board[i][q])
                    # Discard digits based on the subfield
                    subgrid_row_start = self.subgrid_size * (i // self.subgrid_size)
                    subgrid_col_start = self.subgrid_size * (j // self.subgrid_size)
                    for x in range(subgrid_row_start, subgrid_row_start + self.subgrid_size):
                        for y in range(subgrid_col_start, subgrid_col_start + self.subgrid_size):
                            if self.board[x][y] is not None and self.board[x][y] != 0 and self.board[x][y] in digits:
                                digits.remove(self.board[x][y])

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
        self.board[i][j] = value
    
    def _is_correct(self):
        """
        Check if the current board is valid (no duplicate values in rows, columns, or subgrids).
        
        Returns:
            bool: True if the board is valid, False otherwise.
        """
        board = self.board
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
        board = self.board
        size = self.board_size
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _count_solutions(self):
        """
        Recursively count all complete solutions for the puzzle using backtracking.
        
        Returns:
            int: Number of complete solutions.
        """
        empty = self._find_empty()
        if not empty:
            # Check if the complete board is actually a valid solution
            if self._is_correct():
                return 1  # Found a valid complete solution
            else:
                return 0  # Board is full but invalid
        i, j = empty
        count = 0
        for num in range(1, self.board_size + 1):
            self._set_cell(i, j, num)
            if self._is_correct():
                count += self._count_solutions()
            self._set_cell(i, j, 0)  # Backtrack
        return count
    
    @staticmethod
    def _canonicalize(matrix: list[list[int]]) -> list[list[int]]:
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