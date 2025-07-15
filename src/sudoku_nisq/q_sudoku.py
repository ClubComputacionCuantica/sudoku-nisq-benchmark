import json
import hashlib
import math
import matplotlib.pyplot as plt
import gc
import warnings
from sudoku_py import SudokuGenerator as sudokupy
from typing import Literal, Dict, Any, Optional, Type, TYPE_CHECKING
from pathlib import Path

# Import solvers for backward compatibility
from sudoku_nisq.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.graph_coloring_solver import GraphColoringQuantumSolver
from sudoku_nisq.backtracking_solver import BacktrackingQuantumSolver
from sudoku_nisq.metadata_manager import MetadataManager
from sudoku_nisq.backends import BackendManager

if TYPE_CHECKING:
    from sudoku_nisq.quantum_solver import QuantumSolver

class QSudoku():
    """
    Sudoku puzzle class with quantum solver integration and backend management.
    
    Supports both traditional puzzle operations and quantum algorithm research workflows.
    Features single active solver architecture with automatic memory management.
    
    Attributes:
        board (list): 2D list representing the Sudoku board.
        subgrid_size (int): Size of the subgrid (e.g., 2 for 4x4 Sudoku, 3 for 9x9 Sudoku).
        board_size (int): Size of the board (subgrid_size * subgrid_size).
        total_cells (int): Total number of cells in the board.
        num_missing_cells (int): Number of empty cells in the board.
        canonicalize (bool): Whether to canonicalize the puzzle board.
        open_tuples (list): List of open cell possibilities.
        pre_tuples (list): List of preset (pre-filled) cell tuples.
    
    Phase 2 Methods (Recommended):
        generate(subgrid_size, num_missing_cells): Factory method for puzzle creation.
        set_solver(solver_class, encoding, **kwargs): Set active solver with auto-cleanup.
        build_circuit(): Build quantum circuit using active solver.
        attach_backend(alias): Attach backend from global registry.
        init_ibm/init_quantinuum(): One-step backend initialization and attachment.
        transpile(backend_alias, opt_levels): Transpile circuit for hardware backend.
        run(backend_alias, opt_level, shots): Execute on hardware backend.
        run_aer(shots): Execute on Aer simulator.
        report_resources(): Get comprehensive resource summary.
        drop_solver(): Explicitly free solver memory.
    
    Legacy Methods (Deprecated):
        init_exactcover(): Use set_solver(ExactCoverQuantumSolver, encoding) instead.
        init_graphcoloring(): Use set_solver(GraphColoringQuantumSolver) instead.
        init_backtracking(): Use set_solver(BacktrackingQuantumSolver) instead.
        
    Utility Methods:
        plot(title): Plot the Sudoku grid.
        count_solutions(): Count all possible solutions.
        get_hash(): Return SHA-256 hash of the puzzle.
        get_canon_hash(): Return hash of canonicalized puzzle.
    """
    def __init__(self, board=[], subgrid_size=2, num_missing_cells=6, canonicalize=False, csv_path=None, cache_base=None):
        """
        Initialize a QSudoku puzzle instance.
        
        Args:
            board (list, optional): Custom board as a matrix. If provided, used directly.
            subgrid_size (int, optional): Size of the subgrid (default: 2).
            num_missing_cells (int, optional): Number of cells to remove (sudoku-py).
            canonicalize (bool, optional): Canonicalize the board after generation (default: False).
            csv_path (str, optional): Path to CSV file for logging experiment results.
            cache_base (str, optional): Base directory for caching circuits and metadata.
        """
        
        # Optionally use 
        #   custom board as a matrix
        #   sudoku-py: allows to generate puzzles by number of blank cells.
        
        if board:
            self.board = board
            self.subgrid_size = int(math.isqrt(len(board)))
            self.board_size = self.subgrid_size*self.subgrid_size
            self.num_missing_cells = sum(1 for row in self.board for cell in row if cell == 0 or cell is None)
        else:
            self.subgrid_size = subgrid_size
            self.board_size = self.subgrid_size*self.subgrid_size
            puzzle = sudokupy(board_size=self.board_size)
            self.num_missing_cells = num_missing_cells
            puzzle.generate(cells_to_remove=self.num_missing_cells)
            puzzle.board_exchange_values({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9})
            self.board = puzzle.board
            
        self.total_cells = self.board_size * self.board_size
            
        self.canonicalize = canonicalize
        # Canonicalize the board if requested
        if canonicalize:
            self.board = canonicalize_puzzle(self.board)
        
        self._solver: Optional["QuantumSolver"] = None
        self._attached_backends: Dict[str, Any] = {}
        
        self._metadata = MetadataManager(
            cache_base=Path(cache_base) if cache_base else Path(".quantum_solver_cache"),
            puzzle_hash=self.get_hash(),
            log_csv_path=Path(csv_path) if csv_path else None
        )
    
    @classmethod
    def generate(cls, subgrid_size: int, num_missing_cells: int, **kwargs):
        """Factory method for puzzle generation"""
        return cls(subgrid_size=subgrid_size, num_missing_cells=num_missing_cells, **kwargs)
    
    def _swap_solver(self, new_solver: "QuantumSolver") -> None:
        """Internal: clean up old solver and install new one"""
        if self._solver:
            del self._solver
            gc.collect()
        self._solver = new_solver
    
    def set_solver(self, solver_class: Type["QuantumSolver"], encoding: str = None, **solver_kwargs) -> "QuantumSolver":
        """Replace current solver with new one (auto-cleanup previous)"""
        # Create new solver
        new_solver = solver_class(
            sudoku=self,
            encoding=encoding,
            **solver_kwargs
        )
        
        # Swap in new solver (cleanup handled internally)
        self._swap_solver(new_solver)
        
        # Record puzzle metadata once per solver
        self._metadata.ensure_puzzle_fields(
            size=self.board_size,
            num_missing_cells=self.num_missing_cells,
            board=self.board  # It's already a Python list, no .tolist() needed
        )
        
        # Flush metadata to disk immediately
        self._metadata.save()
        
        return self._solver
    
    def drop_solver(self) -> None:
        """Explicitly free solver memory without replacement"""
        if self._solver:
            del self._solver
            self._solver = None
            gc.collect()
    
    def build_circuit(self):
        """Delegate to active solver"""
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        return self._solver.build_main_circuit()
    
    def attach_backend(self, alias: str) -> None:
        """Attach a backend from global registry to this puzzle"""
        # Fail fast validation
        backend = BackendManager.get(alias)  # Raises if not found
        self._attached_backends[alias] = backend
        
    def init_ibm(self, api_token: str, instance: str, device: str, alias: str = None) -> str:
        """Initialize and attach IBM backend to this puzzle"""
        alias = BackendManager.init_ibm(api_token, instance, device, alias)
        self.attach_backend(alias)
        return alias
        
    def init_quantinuum(self, service, token: str, device: str, alias: str = None) -> str:
        """(Placeholder) Initialize and attach Quantinuum backend to this puzzle"""
        alias = BackendManager.init_quantinuum(service, token, device, alias)
        self.attach_backend(alias)
        return alias
    
    def transpile(self, backend_alias: str, opt_levels: list, **kwargs):
        """Transpile using attached backend (fail fast if not attached)"""
        if backend_alias not in self._attached_backends:
            attached = list(self._attached_backends.keys())
            raise ValueError(f"Backend '{backend_alias}' not attached. Attached: {attached}")
        
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
            
        return self._solver.transpile_and_analyze(backend_alias, opt_levels, **kwargs)
    
    def run(self, backend_alias: str, opt_level: int, shots: int, **kwargs):
        """Run on attached hardware backend"""
        if backend_alias not in self._attached_backends:
            attached = list(self._attached_backends.keys())
            raise ValueError(f"Backend '{backend_alias}' not attached. Attached: {attached}")
            
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
            
        return self._solver.run(backend_alias, shots, opt_level, **kwargs)
    
    def run_aer(self, shots: int = 1024, **kwargs):
        """Run logical circuit on Aer simulator (no transpilation needed)"""
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        return self._solver.run_aer(shots, **kwargs)
    
    def report_resources(self):
        """Get resource summary from metadata"""
        return self._metadata.get_resource_summary()

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
    
    # Backward compatibility wrappers (DEPRECATED)
    def init_exactcover(self, encoding: Literal["simple", "pattern"] = "simple"):
        """DEPRECATED: Use set_solver(ExactCoverQuantumSolver, encoding) instead"""
        warnings.warn(
            "init_exactcover is deprecated. Use set_solver(ExactCoverQuantumSolver, encoding) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        solver = self.set_solver(ExactCoverQuantumSolver, encoding=encoding)
        # For backward compatibility, also set as attribute
        setattr(self, f"exact_{encoding}", solver)
        return solver

    def init_graphcoloring(self):
        """DEPRECATED: Use set_solver(GraphColoringQuantumSolver) instead"""
        warnings.warn(
            "init_graphcoloring is deprecated. Use set_solver(GraphColoringQuantumSolver) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        solver = self.set_solver(GraphColoringQuantumSolver)
        # For backward compatibility, also set as attribute
        self.coloring = solver
        return solver
    
    def init_backtracking(self):
        """DEPRECATED: Use set_solver(BacktrackingQuantumSolver) instead"""
        warnings.warn(
            "init_backtracking is deprecated. Use set_solver(BacktrackingQuantumSolver) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        solver = self.set_solver(BacktrackingQuantumSolver)
        # For backward compatibility, also set as attribute  
        self.backtracking = solver
        return solver
    
    # Backward compatibility attribute access
    @property
    def exact_simple(self):
        """DEPRECATED: Access solver directly via set_solver() instead"""
        warnings.warn("Direct solver access is deprecated", DeprecationWarning, stacklevel=2)
        if isinstance(self._solver, ExactCoverQuantumSolver) and self._solver.encoding == "simple":
            return self._solver
        return None
    
    @property  
    def exact_pattern(self):
        """DEPRECATED: Access solver directly via set_solver() instead"""
        warnings.warn("Direct solver access is deprecated", DeprecationWarning, stacklevel=2)
        if isinstance(self._solver, ExactCoverQuantumSolver) and self._solver.encoding == "pattern":
            return self._solver
        return None

    def get_hash(self):
        """
        Return the hash of the puzzle (SHA-256 of original board).
        
        Returns:
            str: Hash string.
        """
        serialized = json.dumps(self.board)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    
    def get_canon_hash(self):
        """
        Return the canonical hash of the puzzle (SHA-256 of canonicalized board).
        
        Returns:
            str: Hash string.
        """
        return hash_canonical_puzzle(self.board, already_canonicalized=self.canonicalize)

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

    def count_solutions(self):
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
                count += self.count_solutions()
            self._set_cell(i, j, 0)  # Backtrack
        return count

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