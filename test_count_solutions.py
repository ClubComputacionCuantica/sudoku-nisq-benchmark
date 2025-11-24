"""Test count_solutions method"""

from sudoku_nisq import ExactCoverProblem

# Test the small example
problem = ExactCoverProblem.create_small_example()
print('Universe:', problem.universe)
print('Subsets:')
for k, v in problem.subsets.items():
    print(f'  {k}: {v}')
print()

# Check claimed solutions
print('Checking claimed solutions:')
solutions = [
    ['S_0', 'S_2'],  # {0,3} + {1,2} = {0,1,2,3}
    ['S_4', 'S_5', 'S_3'],  # {0} + {1,3} + {2,3} - has overlap at 3
    ['S_1', 'S_3']  # {0,1,2} + {2,3} - has overlap at 2
]

for i, sol in enumerate(solutions, 1):
    elements_covered = []
    print(f'Solution {i}: {sol}')
    for s in sol:
        elements_covered.extend(problem.subsets[s])
    print(f'  Covers: {elements_covered}')
    is_unique = len(elements_covered) == len(set(elements_covered))
    is_complete = set(elements_covered) == set(problem.universe)
    print(f'  Valid exact cover: {is_unique and is_complete}')
    print()

# Count actual solutions
print('Running count_solutions()...')
n = problem.count_solutions()
print(f'Algorithm found: {n} solution(s)')
print()

# Test on simple problem
print('Testing on simple 2-element problem:')
universe = [0, 1]
subsets = {
    'S_0': [0],
    'S_1': [1]
}
simple = ExactCoverProblem(universe, subsets)
n_simple = simple.count_solutions()
print(f'  Universe: {universe}')
print(f'  Subsets: {subsets}')
print(f'  Solutions found: {n_simple} (expected 1)')
