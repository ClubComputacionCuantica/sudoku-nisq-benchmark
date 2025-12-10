#!/usr/bin/env python3
"""Debug canonical matrix sorting."""

from src.sudoku_nisq.exact_cover_problem import ExactCoverProblem

# Test case: 2×2 identity matrix
matrix = [[1, 0], [0, 1]]
problem = ExactCoverProblem.from_incidence_matrix(matrix)

print("Original matrix:")
for row in matrix:
    print(f"  {row}")

canonical_matrix, ordered_universe = problem.to_canonical_matrix()

print("\nCanonical matrix:")
for row in canonical_matrix:
    print(f"  {row}")

print(f"\nOrdered universe: {ordered_universe}")

# The issue: columns [1,0] and [0,1] are being sorted
# [0,1] comes before [1,0] lexicographically
# So canonical matrix is [[0,1], [1,0]]
# This gives bits: 0,1,1,0 instead of 1,0,0,1

# Let's trace through the algorithm
print("\nTracing algorithm:")
universe = [0, 1]
subsets = {'S_0': [0], 'S_1': [1]}

problem2 = ExactCoverProblem(universe, subsets)
canonical_matrix2, _ = problem2.to_canonical_matrix()

print(f"Universe: {universe}")
print(f"Subsets: {subsets}")
print("Canonical matrix:")
for row in canonical_matrix2:
    print(f"  {row}")

# Column for S_0=[0] is [1,0] (bit 1 at index 0)
# Column for S_1=[1] is [0,1] (bit 1 at index 1)
# After sorting: [0,1] < [1,0], so columns are swapped
print("\nColumn bitvectors before sorting:")
print("  S_0=[0] → [1, 0]")
print("  S_1=[1] → [0, 1]")
print("\nAfter lexicographic sort:")
print("  [0, 1] < [1, 0]")
print("  So columns become: [[0,1], [1,0]]")