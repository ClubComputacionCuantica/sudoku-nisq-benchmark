from collections import defaultdict
from sudoku_nisq.sudoku_pattern_generation import PatternGeneration

class ExactCoverEncoding:
    """Transforms Sudoku puzzles into exact cover problems for quantum solving.

    This class encodes Sudoku constraint satisfaction problems as exact cover problems,
    which can then be solved using quantum algorithms. The exact cover formulation
    represents Sudoku constraints as a universe of elements that must be covered
    exactly once by selected subsets.

    The encoding supports two main approaches:

    * Simple encoding: Direct mapping of cell-digit assignments to constraints
    * Pattern-based encoding: Uses row patterns to reduce the problem size

    Both standard Sudoku (3x3+ subgrids) and 2x2 mini-Sudoku are supported with
    specialized constraint generation methods.

    :ivar int subgrid_size: Size of the Sudoku subgrid (e.g., 3 for 9x9 Sudoku).
    :ivar list[tuple[int,int,int]] open_tuples: Available cell-digit possibilities
        from the puzzle as ``(row, col, digit)`` tuples.
    :ivar list[tuple[int,int,int]] set_tuples: Pre-filled cells from the puzzle
        as ``(row, col, digit)`` tuples.
    :ivar dict[str, list] simple_subsets: Subsets for the simple encoding approach.
    :ivar dict[str, list] pattern_subsets: Subsets for the pattern-based encoding.
    :ivar list universe: Complete set of constraints for standard Sudoku.
    :ivar list universe2x2: Complete set of constraints for 2x2 mini-Sudoku.

    Example:
        .. code-block:: python

            from sudoku_nisq.sudoku_puzzle import SudokuPuzzle

            # Create a puzzle
            puzzle = SudokuPuzzle.generate(subgrid_size=3, num_missing_cells=20)

            # Generate exact cover encoding
            encoding = ExactCoverEncoding(puzzle)

            # Access the generated constraints and subsets
            print(f"Universe size: {len(encoding.universe)}")
            print(f"Simple subsets: {len(encoding.simple_subsets)}")
            print(f"Pattern subsets: {len(encoding.pattern_subsets)}")

    Note:
        Pattern encoding strategy is due to Weiß, Maximilian. 2022. 
        Encoding strategies to solve Sudoku with Quantum Computers. 
        Bachelorarbeit, Institut für Informatik, Ludwig-Maximilians-Universität München. 
        Submitted 19 July 2022. Available at: https://elib.dlr.de/193653/1/Abgabe_weiss22.pdf
    """
    def __init__(self, puzzle):
        """Initialize the exact cover encoding for a given Sudoku puzzle.
        
        Sets up the exact cover formulation by extracting puzzle constraints and
        generating both simple and pattern-based subset encodings. The appropriate
        universe of constraints is selected based on the puzzle size.
        
        Args:
            puzzle (SudokuPuzzle): The Sudoku puzzle instance to encode. Must have
                the following properties:
                - subgrid_size: Size of the subgrid (e.g., 3 for 9x9)
                - open_tuples: List of (row, col, digit) possibilities for empty cells
                - pre_tuples: List of (row, col, digit) for pre-filled cells
                
        Note:
            For puzzles with subgrid_size >= 2, uses the standard constraint set
            including subgrid constraints. For 2x2 puzzles, uses a simplified
            constraint set without subgrid constraints due to overlap with
            row/column constraints.
        """
        self.subgrid_size = puzzle.subgrid_size
        self.open_tuples = puzzle.open_tuples
        self.set_tuples = puzzle.pre_tuples

        # Subset and universe generation differs for 2x2 (subgrid_size=1) special case
        if self.subgrid_size >= 2:
            self.simple_subsets = self.gen_simple_subsets()
            possible_patterns = PatternGeneration(puzzle=puzzle)
            self.pattern_subsets = self.gen_patterns_subsets(
                possible_patterns=possible_patterns.patterns,
                fixed_tuples=self.set_tuples
            )
            self.universe = []
            self.gen_universe()
        else:
            # 2x2 uses reduced constraint set (no subgrid constraints) and matching subsets
            self.simple_subsets = self.gen_simple_subsets2x2()
            possible_patterns = PatternGeneration(puzzle=puzzle)
            self.pattern_subsets = self.gen_patterns_subsets2x2(
                possible_patterns=possible_patterns.patterns,
                fixed_tuples=self.set_tuples
            )
            self.universe2x2 = []
            self.gen_universe2x2()
            
    def _cell_const(self):
        """Generate cell occupancy constraints.
        
        Creates constraints ensuring that each empty cell is filled with exactly
        one digit. Each constraint represents a unique (row, column) position
        that must be covered by exactly one subset in the solution.
        
        Returns:
            List[Tuple[int, int]]: List of (row, col) tuples representing cell
                positions that need to be filled.
                
        Example:
            For a puzzle with empty cells at (0,1) and (2,3), returns:
            [(0, 1), (2, 3)]
        """
        unique_pairs = set((x, y) for x, y, _ in self.open_tuples)
        return list(unique_pairs)

    def _row_const(self):
        """Generate row uniqueness constraints.
        
        Creates constraints ensuring that each digit appears exactly once in each
        row. Each constraint represents a specific (row, digit) combination that
        must be covered by exactly one subset.
        
        Returns:
            List[Tuple[str, int, int]]: List of ('row', row_index, digit) tuples
                representing row-digit constraints.
                
        Example:
            For open possibilities including digit 5 in row 2, includes:
            ('row', 2, 5)
        """
        _row_constraints = {('row', row, digit) for row, _, digit in self.open_tuples}
        return list(_row_constraints)

    def _col_const(self):
        """Generate column uniqueness constraints.
        
        Creates constraints ensuring that each digit appears exactly once in each
        column. Each constraint represents a specific (column, digit) combination
        that must be covered by exactly one subset.
        
        Returns:
            List[Tuple[str, int, int]]: List of ('col', column_index, digit) tuples
                representing column-digit constraints.
                
        Example:
            For open possibilities including digit 7 in column 1, includes:
            ('col', 1, 7)
        """
        _col_constraints = {('col', column, digit) for _, column, digit in self.open_tuples}
        return list(_col_constraints)

    def _subgrid_const(self):
        """Generate subgrid uniqueness constraints.
        
        Creates constraints ensuring that each digit appears exactly once in each
        subgrid (box). Each constraint represents a specific (subgrid, digit)
        combination that must be covered by exactly one subset.
        
        Returns:
            List[Tuple[str, int, int, int]]: List of ('subgrid', top_row, left_col, digit)
                tuples representing subgrid-digit constraints, where top_row and left_col
                specify the top-left corner of the subgrid.
                
        Example:
            For digit 3 in the top-left 3x3 subgrid, includes:
            ('subgrid', 0, 0, 3)
        """
        _subgrid_constraints = set()
        for tup in self.open_tuples:
            i, j, value = tup
            subgrid_row_start = (i // self.subgrid_size) * self.subgrid_size
            subgrid_col_start = (j // self.subgrid_size) * self.subgrid_size
            _subgrid_constraint = ('subgrid', subgrid_row_start, subgrid_col_start, value)
            _subgrid_constraints.add(_subgrid_constraint)
        return list(_subgrid_constraints)

    def gen_universe(self):
        """Generate the complete universe of constraints for standard Sudoku.
        
        Combines all constraint types (cell, row, column, and subgrid) to create
        the complete set of elements that must be covered exactly once in the
        exact cover problem. This represents all the Sudoku rules as constraints.
        
        The universe is stored in self.universe and contains:
        - Cell constraints: Each empty cell must be filled
        - Row constraints: Each digit must appear once per row  
        - Column constraints: Each digit must appear once per column
        - Subgrid constraints: Each digit must appear once per subgrid
        
        Note:
            This method modifies self.universe in place by extending it with
            constraints from all constraint generation methods.
        """
        self.universe.extend(self._cell_const())
        self.universe.extend(self._row_const())
        self.universe.extend(self._col_const())
        self.universe.extend(self._subgrid_const())

    def gen_simple_subsets(self):
        """Generate simple subsets for the exact cover problem.
        
        Creates one subset for each possible cell-digit assignment in the puzzle.
        Each subset contains all the constraints that would be satisfied by
        placing a specific digit in a specific cell.
        
        Returns:
            Dict[str, List]: Dictionary mapping subset keys ('S_0', 'S_1', etc.)
                to lists of constraints covered by that subset. Each subset contains:
                - Cell constraint: (row, col)
                - Row constraint: ('row', row, digit)  
                - Column constraint: ('col', col, digit)
                - Subgrid constraint: ('subgrid', subgrid_top, subgrid_left, digit)
                
        Example:
            For placing digit 5 at position (1, 2), creates subset:
            'S_0': [(1, 2), ('row', 1, 5), ('col', 2, 5), ('subgrid', 0, 0, 5)]
            
        Note:
            The number of subsets equals the number of open tuples (possible
            digit placements) in the puzzle.
        """
        subsets = defaultdict(list)
        i = 0
        for tuple in self.open_tuples:
            x, y, z = tuple
            cell = (x, y)
            row = ('row', x, z)
            col = ('col', y, z)
            subgrid = ('subgrid', (x // self.subgrid_size) * self.subgrid_size, (y // self.subgrid_size) * self.subgrid_size, z)
            key = f'S_{i}'
            subsets[key].append(cell)
            subsets[key].append(row)
            subsets[key].append(col)
            subsets[key].append(subgrid)
            i += 1
        return dict(subsets)

    def gen_patterns_subsets(self, possible_patterns, fixed_tuples):
        """Generate pattern-based subsets for the exact cover problem.
        
        Creates subsets based on complete row patterns for each digit, potentially
        reducing the subgrid_size of the exact cover problem. Each subset represents a
        valid way to place all instances of a specific digit across the entire
        puzzle, following Sudoku constraints.
        
        Args:
            possible_patterns (Dict[int, List[List[int]]]): Dictionary mapping each
                digit to lists of valid row patterns, where each pattern is a list
                indicating which row each column should contain that digit.
            fixed_tuples (List[Tuple[int, int, int]]): Pre-filled cells as
                (row, col, digit) tuples that should be excluded from pattern
                generation to avoid conflicts.
                
        Returns:
            Dict[str, List]: Dictionary mapping subset keys ('S_0', 'S_1', etc.)
                to lists of constraints covered by that subset. Each subset contains
                constraints for all cells where a digit is placed according to the
                pattern.
                
        Note:
            This encoding can significantly reduce the problem subgrid_size by considering
            complete placements of each digit rather than individual cell assignments.
            Cells that are already filled (fixed_tuples) are omitted to prevent
            constraint conflicts.
        """
        omitted_tuples = defaultdict(list)
        for tup in fixed_tuples:
            x, y, z = tup
            omitted_tuples[z].append((x, y))
        subsets = defaultdict(list)
        i = 0
        for digit, patterns_list in possible_patterns.items():
            for pattern in patterns_list:
                key = f'S_{i}'
                for col in range(len(pattern)):
                    a, b = pattern[col], col
                    cell = (a, b)
                    if cell not in omitted_tuples[digit]:
                        subsets[key].append(cell)
                        row = ('row', a, digit)
                        subsets[key].append(row)
                        col_item = ('col', b, digit)
                        subsets[key].append(col_item)
                        subgrid = ('subgrid', (a // self.subgrid_size) * self.subgrid_size, (b // self.subgrid_size) * self.subgrid_size, digit)
                        subsets[key].append(subgrid)
                i += 1
        return dict(subsets)

    def gen_universe2x2(self):
        """Generate the complete universe of constraints for 2x2 mini-Sudoku.
        
        Creates a simplified constraint set for 2x2 Sudoku puzzles that excludes
        subgrid constraints since they would be redundant with row and column
        constraints in this special case.
        
        The universe2x2 contains:
        - Cell constraints: Each empty cell must be filled
        - Row constraints: Each digit must appear once per row
        - Column constraints: Each digit must appear once per column
        
        Note:
            Subgrid constraints are omitted because in 2x2 Sudoku, the subgrids
            are equivalent to individual cells, making subgrid constraints
            redundant with cell constraints.
        """
        self.universe2x2.extend(self._cell_const())
        self.universe2x2.extend(self._row_const())
        self.universe2x2.extend(self._col_const())

    def gen_simple_subsets2x2(self):
        """Generate simple subsets for 2x2 mini-Sudoku exact cover problem.
        
        Creates simplified subsets for 2x2 Sudoku that exclude subgrid constraints.
        Each subset represents placing a specific digit in a specific cell and
        contains the corresponding cell, row, and column constraints.
        
        Returns:
            Dict[str, List]: Dictionary mapping subset keys ('S_0', 'S_1', etc.)
                to lists of constraints covered by that subset. Each subset contains:
                - Cell constraint: (row, col)
                - Row constraint: ('row', row, digit)
                - Column constraint: ('col', col, digit)
                
        Note:
            This is the 2x2 version of gen_simple_subsets() without subgrid
            constraints due to their redundancy in 2x2 puzzles.
        """
        subsets = defaultdict(list)
        i = 0
        for tuple in self.open_tuples:
            x, y, z = tuple
            cell = (x, y)
            row = ('row', x, z)
            col = ('col', y, z)
            key = f'S_{i}'
            subsets[key].append(cell)
            subsets[key].append(row)
            subsets[key].append(col)
            i += 1
        return dict(subsets)

    def gen_patterns_subsets2x2(self, possible_patterns, fixed_tuples):
        """Generate pattern-based subsets for 2x2 mini-Sudoku exact cover problem.
        
        Creates pattern-based subsets for 2x2 Sudoku that exclude subgrid constraints.
        Each subset represents a complete row pattern for placing all instances of
        a specific digit, covering the corresponding cell, row, and column constraints.
        
        Args:
            possible_patterns (Dict[int, List[List[int]]]): Dictionary mapping each
                digit to lists of valid row patterns for 2x2 Sudoku.
            fixed_tuples (List[Tuple[int, int, int]]): Pre-filled cells as
                (row, col, digit) tuples that should be excluded from patterns.
                
        Returns:
            Dict[str, List]: Dictionary mapping subset keys to lists of constraints
                covered by that subset. Each subset contains cell, row, and column
                constraints but excludes subgrid constraints.
                
        Note:
            This is the 2x2 version of gen_patterns_subsets() without subgrid
            constraints.
        """
        omitted_tuples = defaultdict(list)
        for tup in fixed_tuples:
            x, y, z = tup
            omitted_tuples[z].append((x, y))
        subsets = defaultdict(list)
        i = 0
        for digit, patterns_list in possible_patterns.items():
            for pattern in patterns_list:
                key = f'S_{i}'
                for col in range(len(pattern)):
                    a, b = pattern[col], col
                    cell = (a, b)
                    if cell not in omitted_tuples[digit]:
                        subsets[key].append(cell)
                        row = ('row', a, digit)
                        subsets[key].append(row)
                        col_item = ('col', b, digit)
                        subsets[key].append(col_item)
                i += 1
        return dict(subsets)