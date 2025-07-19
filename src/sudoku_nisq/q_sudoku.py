import gc
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
from pathlib import Path

from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
from sudoku_nisq.metadata_manager import MetadataManager
from sudoku_nisq.backends import BackendManager

if TYPE_CHECKING:
    from sudoku_nisq.quantum_solver import QuantumSolver
    from pytket.extensions.quantinuum.backends.credential_storage import CredentialStorage

class QSudoku():
    """
    Sudoku puzzle class with quantum solver integration and backend management.

    Supports both traditional puzzle operations and quantum algorithm research workflows.
    Features single active solver architecture with automatic memory management.

    Attributes:
        _solver (QuantumSolver): Active quantum solver instance.
        _attached_backends (dict): Dictionary of attached backends.
        _metadata (MetadataManager): Metadata manager for caching and logging.

    Methods:
        __init__(board, subgrid_size, num_missing_cells, canonicalize, cache_base): Initialize a QSudoku puzzle instance.
        generate(subgrid_size, num_missing_cells, **kwargs): Factory method for puzzle creation.
        set_solver(solver_class, encoding, **kwargs): Set active solver with auto-cleanup.
        drop_solver(): Explicitly free solver memory.
        build_circuit(): Build quantum circuit using active solver.
        draw_circuit(circuit=None): Draw the quantum circuit.
        attach_backend(alias): Attach backend from global registry.
        init_ibm(api_token, instance, device, alias=None): Initialize and attach IBM backend to this puzzle.
        init_quantinuum(device, alias=None, token_store=None, provider=None): Initialize and attach Quantinuum backend to this puzzle.
        transpile(backend_alias, opt_levels, **kwargs): Transpile circuit for hardware backend.
        run(backend_alias, opt_level, shots, **kwargs): Execute on hardware backend.
        run_aer(shots=1024, **kwargs): Execute on Aer simulator.
        counts_plot(counts=None, backend_alias=None, shots=None, top_n=20, show_valid_only=False, figsize=(12, 6), show_summary=True): Create a bar plot of measurement counts with Sudoku-specific enhancements.
        report_resources(): Get comprehensive resource summary.
        get_hash(): Return SHA-256 hash of the current puzzle.
    """
    def __init__(self, puzzle: SudokuPuzzle, cache_base: Optional[str] = None):
        """
        Initialize a QSudoku puzzle instance.

        Args:
            puzzle: SudokuPuzzle instance to wrap
            cache_base (str, optional): Base directory for caching circuits and metadata.
        """
        
        self.puzzle = puzzle
        
        # Initialize solver management
        self._solver: Optional["QuantumSolver"] = None
        self._attached_backends: Dict[str, Any] = {}
        
        # Metadata manager for caching and logging
        self._metadata = MetadataManager(
            cache_base=Path(cache_base) if cache_base else Path(".quantum_solver_cache"),
            puzzle_hash=self.puzzle.get_hash()
        )
    
    @classmethod
    def generate(cls, subgrid_size: int, num_missing_cells: int, canonicalize: bool = False, cache_base: Optional[str] = None) -> "QSudoku":
        """
        Factory method for generating a QSudoku instance.

        Args:
            subgrid_size (int): Size of the subgrid (e.g., 3 for 9x9 Sudoku).
            num_missing_cells (int): Number of cells to remove.
            canonicalize (bool): Whether to canonicalize the puzzle.
            cache_base (str, optional): Base directory for caching circuits and metadata.

        Returns:
            QSudoku: A new QSudoku instance with a generated puzzle.
        """
        # Generate the puzzle using SudokuPuzzle
        puzzle = SudokuPuzzle.generate(subgrid_size=subgrid_size, num_missing_cells=num_missing_cells, canonicalize=canonicalize)

        # Wrap the puzzle in a QSudoku instance
        return cls(puzzle=puzzle, cache_base=cache_base)

    @classmethod
    def from_board(cls, board: List[List[int]], canonicalize: bool = False, cache_base: Optional[str] = None) -> "QSudoku":
        """
        Factory method for creating a QSudoku instance from an existing board.

        Args:
            board (List[List[int]]): Existing Sudoku board.
            canonicalize (bool): Whether to canonicalize the board.
            cache_base (str, optional): Base directory for caching circuits and metadata.

        Returns:
            QSudoku: A new QSudoku instance with the provided board.
        """
        # Create the puzzle using SudokuPuzzle.from_board()
        puzzle = SudokuPuzzle.from_board(board=board, canonicalize=canonicalize)

        # Wrap the puzzle in a QSudoku instance
        return cls(puzzle=puzzle, cache_base=cache_base)

    @property
    def board(self) -> List[List[int]]:
        """Get the current puzzle board."""
        return self.puzzle.board
    
    @property
    def board_size(self) -> int:
        """Get the size of the board (e.g., 4 for 4x4, 9 for 9x9)."""
        return self.puzzle.board_size
    
    @property
    def subgrid_size(self) -> int:
        """Get the subgrid size (e.g., 2 for 4x4, 3 for 9x9)."""
        return self.puzzle.subgrid_size
    
    @property
    def num_missing_cells(self) -> int:
        """Get the number of missing cells in the puzzle."""
        return self.puzzle.num_missing_cells

    def plot_puzzle(self) -> None:
        """Plot the current Sudoku board."""
        return self.puzzle.plot()

    def init_ibm(self, api_token: str, instance: str, device: str, alias: Optional[str] = None) -> str:
        """Initialize and attach IBM backend to this puzzle"""
        alias = BackendManager.init_ibm(api_token, instance, device, alias)
        self.attach_backend(alias)
        return alias

    def init_quantinuum(self, device: str, alias: Optional[str] = None, token_store: Optional['CredentialStorage'] = None, provider: Optional[str] = None) -> str:
        """Initialize and attach Quantinuum backend to this puzzle"""
        alias = BackendManager.init_quantinuum(device, alias, token_store, provider)
        self.attach_backend(alias)
        return alias
    
    def set_solver(self, solver_class: Type["QuantumSolver"], encoding: Optional[str] = None, **solver_kwargs) -> Optional["QuantumSolver"]:
        """Replace current solver with new one (auto-cleanup previous)"""
        # Create new solver
        new_solver = solver_class(
            puzzle=self.puzzle,
            metadata_manager=self._metadata,
            encoding=encoding,
            **solver_kwargs
        )
        
        # Swap in new solver (cleanup handled internally)
        self._swap_solver(new_solver)
        
        # Record puzzle metadata once per solver
        self._metadata.ensure_puzzle_fields(
            size=self.board_size,
            num_missing_cells=self.num_missing_cells,
            board=self.board
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
    
    def _swap_solver(self, new_solver: "QuantumSolver") -> None:
        """Internal: clean up old solver and install new one"""
        if self._solver:
            del self._solver
            gc.collect()
        self._solver = new_solver
    
    def build_circuit(self):
        """Delegate to active solver"""
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        return self._solver.build_main_circuit()
    
    def draw_circuit(self, circuit = None):
        """
        Delegate to active solver's draw method.
        If no circuit is provided, uses the main circuit from current solver.
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        
        if circuit is None:
            circuit = self._solver.build_main_circuit()

        return self._solver.draw_circuit(circuit)

    def attach_backend(self, alias: str) -> None:
        """Attach a backend from global registry to this puzzle"""
        # Fail fast validation
        backend = BackendManager.get(alias)  # Raises if not found
        self._attached_backends[alias] = backend
    
    def transpile(self, backend_alias: str, opt_level: int, **kwargs):
        """Transpile using attached backend (fail fast if not attached)"""
        if backend_alias not in self._attached_backends:
            attached = list(self._attached_backends.keys())
            raise ValueError(f"Backend '{backend_alias}' not attached. Attached: {attached}")
        
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")

        return self._solver.transpile_and_analyze(self._attached_backends[backend_alias], backend_alias, opt_level, **kwargs)

    def run(self, backend_alias: str, opt_level: int, shots: int, **kwargs):
        """Run on attached hardware backend"""
        if backend_alias not in self._attached_backends:
            attached = list(self._attached_backends.keys())
            raise ValueError(f"Backend '{backend_alias}' not attached. Attached: {attached}")
            
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
            
        return self._solver.run(self._attached_backends[backend_alias], backend_alias, shots, force_run = False, optimisation_level = opt_level, **kwargs)
    
    def run_aer(self, shots: int = 1024, **kwargs):
        """Run logical circuit on Aer simulator (no transpilation needed)"""
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
        return self._solver.run_aer(shots, **kwargs)
    
    def counts_plot(self, counts=None, backend_alias=None, shots=None, top_n=20, 
                    show_valid_only=False, figsize=(12, 6), show_summary=True):
        """
        Create a bar plot of measurement counts with Sudoku-specific enhancements.
        
        This method provides an easy interface to visualize quantum execution results
        with automatic validation highlighting and summary statistics.
        
        Args:
            counts: Dictionary of measurement outcomes, pytket Result object, or None
            backend_alias: Name of the backend used (for title display)
            shots: Total number of shots (for title and percentage calculation)
            top_n: Show only the top N most frequent outcomes (default: 20)
            show_valid_only: If True, only show outcomes that represent valid Sudoku solutions
            figsize: Figure size tuple (width, height)
            show_summary: If True, print summary statistics (default: True)
            
        Examples:
            # Plot results from Aer simulation
            result = puzzle.run_aer(shots=1024)
            puzzle.counts_plot(result, backend_alias="Aer", shots=1024)
            
            # Plot only valid solutions from hardware run
            result = puzzle.run("ibm_brisbane", opt_level=1, shots=100)
            puzzle.counts_plot(result, backend_alias="IBM Brisbane", show_valid_only=True)
            
            # Plot with custom counts dictionary
            puzzle.counts_plot(my_counts, backend_alias="Custom", shots=500, top_n=10)
            
            # Plot without summary statistics
            puzzle.counts_plot(result, backend_alias="Aer", show_summary=False)
        """
        if not self._solver:
            raise ValueError("No solver set. Call set_solver() first.")
            
        # Delegate to the solver's counts_plot method with QSudoku-specific defaults
        if backend_alias is None:
            backend_alias = "Unknown Backend"
            
        return self._solver.counts_plot(
            counts=counts,
            backend_alias=backend_alias, 
            shots=shots,
            top_n=top_n,
            show_valid_only=show_valid_only,
            figsize=figsize,
            show_summary=show_summary
        )
    
    def report_resources(self):
        """Get resource summary from metadata"""
        return self._metadata.get_resource_summary()

    def get_hash(self) -> str:
        """Delegate to SudokuPuzzle."""
        return self.puzzle.get_hash()