from collections import defaultdict
from sudoku_nisq.sudoku_pattern_generation import PatternGeneration

# The `ExactCoverEncoding` class generates constraints for an exact cover problem based on open tuples for any given sudoku puzzle

class ExactCoverEncoding:
    def __init__(self, sudoku):
        self.size = sudoku.subgrid_size
        self.open_tuples = sudoku.open_tuples
        self.set_tuples = sudoku.pre_tuples
        
        self.simple_subsets = self.gen_simple_subsets()
        possible_patterns = PatternGeneration(sudoku=sudoku)
        self.pattern_subsets = self.gen_patterns_subsets(possible_patterns=possible_patterns.patterns,fixed_tuples=self.set_tuples)
        
        if self.size >= 2:    
            self.universe = []
            self.gen_universe()
        else:
            self.universe2x2 = []
            self.gen_universe2x2()
            
    # Constraint generation for Universe
    
    def _cell_const(self):
        unique_pairs = set((x, y) for x, y, _ in self.open_tuples)
        return list(unique_pairs)

    def _row_const(self):
        _row_constraints = {('row', row, digit) for row, _, digit in self.open_tuples}
        return list(_row_constraints)

    def _col_const(self):
        _col_constraints = {('col', column, digit) for _, column, digit in self.open_tuples}
        return list(_col_constraints)

    def _subgrid_const(self):
        _subgrid_constraints = set()
        for tup in self.open_tuples:
            i, j, value = tup
            subgrid_row_start = (i // self.size) * self.size
            subgrid_col_start = (j // self.size) * self.size
            _subgrid_constraint = ('subgrid', subgrid_row_start, subgrid_col_start, value)
            _subgrid_constraints.add(_subgrid_constraint)
        return list(_subgrid_constraints)

    def gen_universe(self):
        self.universe.extend(self._cell_const())
        self.universe.extend(self._row_const())
        self.universe.extend(self._col_const())
        self.universe.extend(self._subgrid_const())

    def gen_simple_subsets(self):
        subsets = defaultdict(list)
        i = 0
        for tuple in self.open_tuples:
            x, y, z = tuple
            cell = (x, y)
            row = ('row', x, z)
            col = ('col', y, z)
            subgrid = ('subgrid', (x // self.size) * self.size, (y // self.size) * self.size, z)
            key = f'S_{i}'
            subsets[key].append(cell)
            subsets[key].append(row)
            subsets[key].append(col)
            subsets[key].append(subgrid)
            i += 1
        return dict(subsets)

    def gen_patterns_subsets(self, possible_patterns, fixed_tuples):
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
                        subgrid = ('subgrid', (a // self.size) * self.size, (b // self.size) * self.size, digit)
                        subsets[key].append(subgrid)
                i += 1
        return dict(subsets)

    """ 2x2 case """

    def gen_universe2x2(self):
        self.universe2x2.extend(self._cell_const())
        self.universe2x2.extend(self._row_const())
        self.universe2x2.extend(self._col_const())

    def gen_simple_subsets2x2(self):
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