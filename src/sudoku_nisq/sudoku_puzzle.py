from dataclasses import dataclass
import math
import json
import hashlib
from typing import List, Optional
from sudoku_py import SudokuGenerator


@dataclass
class SudokuPuzzle:
    """Represents a Sudoku puzzle with creation, validation, and analysis helpers.

    Provides methods to generate puzzles, derive metadata (hash, counts, solution
    enumeration), and basic visualization. Use `SudokuPuzzle.generate` for new
    puzzles or `SudokuPuzzle.from_board` to wrap an existing grid.
    """
    board: List[List[int]]
    subgrid_size: int
    board_size: int
    num_missing_cells: int
    canonicalize: bool = False

    @classmethod
    def from_board(cls, board: List[List[int]], canonicalize: bool = False) -> "SudokuPuzzle":
        """Creates a SudokuPuzzle instance from an existing board.

        Args:
            board (List[List[int]]): A 2D list representing the Sudoku board.
                Empty cells should be represented as 0 or None.
            canonicalize (bool, optional): Whether to canonicalize the board to
                ensure a standardized digit representation. Defaults to False.

        Returns:
            SudokuPuzzle: A new SudokuPuzzle instance with calculated properties
                including subgrid size, board size, and missing cell count.
        """
        board_size = len(board)
        subgrid_size = int(math.isqrt(board_size))
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
    def generate(
        cls,
        subgrid_size: int,
        num_missing_cells: int,
        canonicalize: bool = False,
        *,
        size: Optional[int] = None,
    ) -> "SudokuPuzzle":
        """Generates a new random Sudoku puzzle with specified parameters.

        Creates a complete Sudoku puzzle and then removes the specified number
        of cells to create the puzzle challenge. Uses the SudokuGenerator library
        to ensure valid puzzle generation.

        Args:
            subgrid_size (int): Size of each subgrid (e.g., 3 for a 9x9 Sudoku).
                The total board size will be subgrid_size² when ``size`` is not provided.
            num_missing_cells (int): Number of cells to remove from the complete
                puzzle to create the challenge.
            canonicalize (bool, optional): Whether to canonicalize the puzzle to
                ensure standardized digit representation. Defaults to False.
            size (Optional[int]): Overall board size N. If provided, supports N=2
                (treated as subgrid_size=1, i.e., no real subgrids) or perfect squares
                like 4, 9, 16 (subgrid_size=sqrt(N)).

        Returns:
            SudokuPuzzle: A new randomly generated SudokuPuzzle instance with
                the specified difficulty level.
        """
        if size is not None:
            if size == 2:
                board_size = 2
                subgrid_size = 1
            else:
                k = int(math.isqrt(size))
                if k * k != size:
                    raise ValueError(f"size must be 2 or a perfect square (e.g., 4, 9, 16); got {size}")
                subgrid_size = k
                board_size = size
        else:
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
        """Creates a visual representation of the Sudoku puzzle using matplotlib.
        
        Generates a matplotlib figure showing the Sudoku grid with proper gridlines
        and filled numbers. Major gridlines separate subgrids while minor gridlines
        separate individual cells.
        
        Args:
            title (str, optional): Title to display at the top of the plot.
                If None, no title will be shown.

        Returns:
            matplotlib.figure.Figure: The matplotlib Figure object containing the
                plot. This can be saved to a file or displayed.
                
        Example:
            >>> puzzle = SudokuPuzzle.generate(subgrid_size=3, num_missing_cells=20)
            >>> fig = puzzle.plot(title="My Sudoku Puzzle")
            >>> fig.savefig("sudoku_plot.png")
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
        """Generates a unique SHA-256 hash identifier for the puzzle.
        
        Creates a deterministic hash based on the current board state that can
        be used for puzzle identification, caching, or comparison purposes.
        
        Returns:
            str: A 64-character hexadecimal SHA-256 hash of the puzzle board.
        """
        board_str = json.dumps(self.board)
        return hashlib.sha256(board_str.encode("utf-8")).hexdigest()
    
    @property
    def num_solutions(self) -> int:
        """Calculates the total number of valid solutions for the puzzle.

        Uses backtracking algorithm to exhaustively search all possible
        completions of the puzzle and count valid solutions. This can be
        computationally expensive for puzzles with many empty cells.

        Returns:
            int: The number of valid complete solutions. Returns 1 for a
                well-formed puzzle with a unique solution, 0 for unsolvable
                puzzles, or >1 for puzzles with multiple solutions.
        """
        return self._count_solutions()
    
    @property
    def pre_tuples(self):
        """Returns all pre-filled (given) cells as coordinate tuples.
        
        Scans the board and identifies all cells that contain initial values
        (non-zero, non-None values). These represent the clues provided with
        the puzzle.
        
        Returns:
            List[Tuple[int, int, int]]: List of tuples in the format (row, col, value)
                for each pre-filled cell. Coordinates are 0-indexed.
        """
        preset_tuples = []
        for i in range(self.board_size):  # Loop over each row
            for j in range(self.board_size):  # Loop over each column in the row
                element = self.board[i][j]
                if element is not None: # Check if the cell is pre-filled
                    preset_tuples.append((i,j,element)) # Store pre-filled cell as tuple
        return preset_tuples

    @property
    def open_tuples(self):
        """Returns all valid digit possibilities for empty cells.
        
        For each empty cell, determines which digits (1 to board_size) can be
        legally placed based on Sudoku constraints (no duplicates in rows,
        columns, or subgrids).
        
        Returns:
            List[Tuple[int, int, int]]: List of tuples in the format (row, col, digit)
                representing each valid digit placement possibility. Each empty
                cell may contribute multiple tuples if multiple digits are valid.
        """
        open_tuples = []
        for i in range(self.board_size):  # Loop over each row
            for j in range(self.board_size):  # Loop over each column in the row
                element = self.board[i][j]
                if element is None or element == 0: # Check if the cell is empty
                    digits = list(range(1, self.board_size + 1)) # Possible digits for the cell
                    # Discard digits based on the column constraint
                    for p in range(self.board_size):
                        if self.board[p][j] is not None and self.board[p][j] != 0 and self.board[p][j] in digits:
                            digits.remove(self.board[p][j])
                    # Discard digits based on the row constraint
                    for q in range(self.board_size):
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
    
    # ------------------------------------------------------------
    # The following functions are used to count the solutions of any given puzzle
    def _set_cell(self, i, j, value):
        """Sets the value of a specific cell in the puzzle board.
        
        Args:
            i (int): Zero-indexed row position.
            j (int): Zero-indexed column position.
            value (int): Value to place in the cell (typically 1 to board_size,
                or 0 for empty).
        """
        self.board[i][j] = value
    
    def _is_correct(self):
        """Validates the current board state against Sudoku constraints.
        
        Checks that no duplicate non-zero values exist in any row, column,
        or subgrid. This validation works with partially filled boards.
        
        Returns:
            bool: True if the current board state is valid according to
                Sudoku rules, False if any constraint violations are found.
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
        """Locates the next empty cell in the board for backtracking.
        
        Scans the board row by row to find the first cell containing 0,
        which represents an empty position.
        
        Returns:
            Tuple[int, int] or None: Tuple of (row, col) indices for the first
                empty cell found, or None if the board is completely filled.
        """
        board = self.board
        size = self.board_size
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _count_solutions(self):
        """Recursively counts all valid complete solutions using backtracking.
        
        Implements a depth-first search algorithm that tries all possible
        digit placements in empty cells and counts how many lead to valid
        complete solutions. Uses backtracking to efficiently explore the
        solution space.
        
        Returns:
            int: Total number of valid complete solutions found. A well-formed
                Sudoku puzzle should return 1 for a unique solution.
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
    
    def enumerate_solutions(self, max_solutions: int = 100) -> list[list[list[int]]]:
        """Enumerates all valid complete solutions using backtracking.
        
        Similar to _count_solutions but collects actual solution boards instead
        of just counting. Includes a configurable limit to prevent excessive
        computation for puzzles with many solutions.
        
        Args:
            max_solutions: Maximum number of solutions to collect. Stops early
                if this limit is reached. Default 100 is reasonable for 2×2 and
                4×4 puzzles. For 9×9+ puzzles, enumeration may be infeasible.
        
        Returns:
            List of solution boards, where each board is a 2D list of integers.
            Returns empty list if puzzle is unsolvable. May return fewer than
            max_solutions if puzzle has fewer valid solutions.
        
        Example:
            >>> puzzle = SudokuPuzzle.from_board([[1, 0], [0, 1]])
            >>> solutions = puzzle.enumerate_solutions(max_solutions=10)
            >>> len(solutions)
            2  # 2×2 puzzle with 2 givens typically has 2 solutions
        
        Note:
            For large puzzles (9×9+) or puzzles with few clues, enumeration
            can be computationally expensive. Consider using count_solutions
            first to check feasibility.
        """
        from typing import List as ListType
        solutions: ListType[ListType[ListType[int]]] = []
        
        def _enumerate_recursive():
            """Helper function for recursive backtracking with solution collection."""
            if len(solutions) >= max_solutions:
                return  # Early stopping
            
            empty = self._find_empty()
            if not empty:
                # Check if the complete board is actually a valid solution
                if self._is_correct():
                    # Deep copy the current board state
                    solution = [row[:] for row in self.board]
                    solutions.append(solution)
                return
            
            i, j = empty
            for num in range(1, self.board_size + 1):
                self._set_cell(i, j, num)
                if self._is_correct():
                    _enumerate_recursive()
                    if len(solutions) >= max_solutions:
                        self._set_cell(i, j, 0)
                        return  # Early stopping
                self._set_cell(i, j, 0)  # Backtrack
        
        _enumerate_recursive()
        return solutions
    # ------------------------------------------------------------
    
    @staticmethod
    def _canonicalize(matrix: list[list[int]]) -> list[list[int]]:
        """Converts a Sudoku matrix to canonical form with sequential digit labeling.
        
        Creates a standardized representation by relabeling non-zero digits
        sequentially (1, 2, 3, ...) based on their first appearance when scanning
        left-to-right, top-to-bottom. This ensures that puzzles equivalent up to
        digit permutation will have identical canonical representations.
        
        Args:
            matrix (List[List[int]]): A square Sudoku matrix where 0 represents
                empty cells and positive integers represent filled cells.
        
        Returns:
            List[List[int]]: Canonicalized matrix with digits relabeled sequentially.
                Empty cells (0) are preserved unchanged.
        
        Example:
            A matrix with digits [2, 5, 7] appearing in that order would be
            canonicalized to [1, 2, 3], maintaining all structural relationships.
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
    
    def create_validation_context(
        self,
        encoding_type: str = 'simple',
        max_solutions: int = 100
    ):
        """Create a ValidationContext for metrics computation from solution enumeration.
        
        Enumerates valid solutions, converts them to bitstrings, and packages them
        into a ValidationContext suitable for automatic metrics computation during
        quantum execution.
        
        Args:
            encoding_type: Either 'simple' or 'pattern' to specify which encoding
                to use for bitstring conversion. Must match the encoding used when
                building the quantum circuit. Default 'simple'.
            max_solutions: Maximum number of solutions to enumerate. Default 100
                is reasonable for 2×2 and small 4×4 puzzles. For 9×9+ puzzles,
                enumeration is typically infeasible.
        
        Returns:
            ValidationContext dataclass with:
            - valid_solutions: List of bitstrings (one per valid solution)
            - total_valid_count: Number of valid solutions found (may be capped)
            - solution_validator: Function to check if a bitstring is valid
        
        Raises:
            ValueError: If encoding_type is invalid or if puzzle has too many
                solutions to enumerate (exceeds max_solutions limit).
        
        Example:
            >>> from sudoku_nisq import QSudoku
            >>> puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=2)
            >>> context = puzzle.puzzle.create_validation_context('simple')
            >>> len(context.valid_solutions)
            2  # Example: 2×2 puzzle with 2 solutions
            >>> puzzle.set_validation_context(context.valid_solutions)
            >>> result = puzzle.run_aer(shots=1024)  # Auto-computes metrics
        
        Warning:
            For puzzles with many empty cells or large board sizes, solution
            enumeration can be computationally expensive. Check num_solutions
            property first for feasibility. Generally:
            - 2×2 puzzles: Always feasible (<10 solutions typically)
            - 4×4 puzzles: Often feasible if well-constrained
            - 9×9+ puzzles: Usually infeasible (billions of solutions)
        
        Note:
            Results are cached in _cached_validation_context attribute to avoid
            redundant computation. Pass force_recompute=True to clear cache (future).
        """
        from sudoku_nisq.metrics.data_models import ValidationContext
        from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding
        
        if encoding_type not in ('simple', 'pattern'):
            raise ValueError(f"encoding_type must be 'simple' or 'pattern', got {encoding_type}")
        
        # Warn for potentially large solution spaces
        if self.board_size >= 9 and self.num_missing_cells > 30:
            import warnings
            warnings.warn(
                f"Puzzle has {self.num_missing_cells} empty cells on {self.board_size}×{self.board_size} board. "
                f"Solution enumeration may be slow or infeasible. Consider using count_solutions first.",
                UserWarning
            )
        
        # Enumerate solutions
        solution_boards = self.enumerate_solutions(max_solutions=max_solutions)
        
        if not solution_boards:
            raise ValueError("Puzzle has no valid solutions. Cannot create ValidationContext.")
        
        # Convert solutions to bitstrings using exact cover encoding
        encoder = ExactCoverEncoding(self)
        valid_bitstrings = []
        
        for solution_board in solution_boards:
            try:
                bitstring = encoder.solution_to_bitstring(solution_board, encoding_type)
                valid_bitstrings.append(bitstring)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to convert solution to bitstring: {e}. Skipping this solution.",
                    UserWarning
                )
        
        if not valid_bitstrings:
            raise ValueError("Could not convert any solutions to bitstrings. Check encoding configuration.")
        
        # Create validator function
        valid_set = set(valid_bitstrings)
        def validator(bitstring: str) -> bool:
            return bitstring in valid_set
        
        # Create and return ValidationContext
        context = ValidationContext(
            valid_solutions=valid_bitstrings,
            total_valid_count=len(valid_bitstrings),
            solution_validator=validator
        )
        
        # Cache for future use
        if not hasattr(self, '_cached_validation_contexts'):
            self._cached_validation_contexts = {}
        self._cached_validation_contexts[encoding_type] = context
        
        return context
