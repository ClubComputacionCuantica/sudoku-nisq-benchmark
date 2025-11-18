from collections import defaultdict
import copy

class PatternGeneration:
    """Generates and manages patterns for Sudoku puzzle solving.
    
    This class creates patterns for each digit based on the puzzle's constraints,
    including fixed positions and possible placements. It's designed to work with
    quantum solving algorithms that require pattern-based representations.
    
    Attributes:
        open_tuples (list): List of tuples representing open positions (row, col, digit).
        fixed_tuples (list): List of tuples representing fixed positions (row, col, digit).
        length (int): The total number of cells in the puzzle (size * size).
        patterns (dict): Dictionary mapping each digit to its possible patterns.
    """
    
    def __init__(self, puzzle) -> None:
        """Initializes the PatternGeneration with a Sudoku puzzle.
        
        Args:
            puzzle: A Sudoku puzzle object containing open_tuples, pre_tuples,
                    and subgrid_size attributes.
        """
        self.open_tuples = puzzle.open_tuples
        self.fixed_tuples = puzzle.pre_tuples
        size = puzzle.subgrid_size
        self.length = size * size
        self.patterns = self.generate_patterns_dict()
        self.patterns = self.pattern_cleanup()

    def pattern_cleanup(self):
        """Cleans up generated patterns by removing invalid ones.
        
        Filters out patterns that are not lists or don't contain all unique values
        from 1 to length. This ensures that only valid, complete patterns are kept.
        
        Returns:
            dict: Dictionary with the same structure as self.patterns but containing
            only valid patterns for each digit.
        """
        final_patterns = {}
        for key, patterns in self.patterns.items():
            valid_patterns = []
            for pattern in patterns:
                if isinstance(pattern, list) and len(set(pattern)) == self.length:
                    valid_patterns.append(pattern)
            final_patterns[key] = valid_patterns
        return final_patterns
    
    def generate_patterns_dict(self):
        """Generates all possible patterns for each digit in the puzzle.
        
        Creates a dictionary where each digit maps to a list of possible patterns.
        Each pattern is a list representing where that digit can be placed in each
        column of the puzzle. The method starts with fixed positions and then
        generates all possible combinations for open positions.
        
        Returns:
            dict: Dictionary mapping each digit (1 to length) to a list of possible
                patterns. Each pattern is a list of length 'length' where the
                value at index i represents the row position of the digit in column i.
        """
        digits = [digit for digit in range(1, self.length + 1)]
        patterns = {digit: [] for digit in digits}
        
        # Initialize patterns with fixed positions
        for digit in patterns:
            pattern = [None] * self.length
            for tup in self.fixed_tuples:
                if tup[2] == digit:
                    pattern[tup[1]] = tup[0]
            patterns[digit].append(pattern)
        
        # Organize open tuples by digit and column for efficient processing
        organized_open_tuples = defaultdict(lambda: defaultdict(list))
        for row, col, digit in self.open_tuples:
            organized_open_tuples[digit][col].append(row)
        
        # Generate all possible pattern combinations
        for digit in organized_open_tuples:
            for col in organized_open_tuples[digit]:
                new_patterns = []
                patterns_to_remove = []
                
                for i in range(len(patterns[digit])):
                    # Create new patterns for each possible row in this column
                    for row in organized_open_tuples[digit][col]:
                        pattern_copy = copy.deepcopy(patterns[digit][i])
                        pattern_copy[col] = row
                        new_patterns.append(pattern_copy)
                    
                    # Mark original pattern for removal
                    pattern_to_remove = copy.deepcopy(patterns[digit][i])
                    patterns_to_remove.append(pattern_to_remove)
                
                # Replace old patterns with new ones
                for item in patterns_to_remove:
                    patterns[digit].remove(item)
                for item in new_patterns:
                    patterns[digit].append(item)

        return patterns