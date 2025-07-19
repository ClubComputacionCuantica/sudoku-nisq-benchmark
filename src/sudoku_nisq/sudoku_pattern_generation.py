from collections import defaultdict
import copy

class PatternGeneration:
    def __init__(self,puzzle) -> None:
        self.open_tuples = puzzle.open_tuples
        self.fixed_tuples = puzzle.pre_tuples
        size = puzzle.subgrid_size
        self.length = size*size
        self.patterns = self.generate_patterns_dict()
        self.patterns = self.pattern_cleanup()

    def pattern_cleanup(self):
        final_patterns = {}
        for key, patterns in self.patterns.items():
            valid_patterns = []
            for pattern in patterns:
                if isinstance(pattern, list) and len(set(pattern)) == self.length:
                    valid_patterns.append(pattern)
            final_patterns[key] = valid_patterns
        return final_patterns
    
    def generate_patterns_dict(self):
        digits = [digit for digit in range(1, self.length + 1)]
        patterns = {digit: [] for digit in digits}
        for digit in patterns:
            pattern = [None] * self.length
            for tup in self.fixed_tuples:
                if tup[2] == digit:
                    pattern[tup[1]] = tup[0]
            patterns[digit].append(pattern)
        organized_open_tuples = defaultdict(lambda:defaultdict(list))
        for row, col, digit in self.open_tuples:
            organized_open_tuples[digit][col].append(row)
        for digit in organized_open_tuples:
            for col in organized_open_tuples[digit]:
                ram = []
                ram_remove = []
                for i in range(len(patterns[digit])):
                    for row in organized_open_tuples[digit][col]:
                        patt_copy = copy.deepcopy(patterns[digit][i])
                        patt_copy[col] = row
                        ram.append(patt_copy)
                    patt_remove = copy.deepcopy(patterns[digit][i])
                    
                    ram_remove.append(patt_remove)
                for item in ram_remove:
                    patterns[digit].remove(item)
                for item in ram:
                    patterns[digit].append(item)

        return patterns