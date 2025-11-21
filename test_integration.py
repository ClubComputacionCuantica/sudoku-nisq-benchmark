"""Quick integration test for exact cover solver with mitigation."""

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

# Generate a simple 2x2 Sudoku
print("Generating 2x2 Sudoku puzzle...")
qs = QSudoku.generate(size=2, num_missing=2, seed=42)
print(f"Puzzle:\n{qs.puzzle}\n")

# Set solver
print("Setting ExactCoverQuantumSolver...")
qs.set_solver(ExactCoverQuantumSolver, encoding='simple')

# Verify validation method exists
print(f"Solver has _is_valid_solution: {hasattr(qs.solver, '_is_valid_solution')}")
print(f"Solver universe: {qs.solver.universe}")
print(f"Solver subsets: {list(qs.solver.subsets.keys())}")

# Test validation with example bitstrings
print("\nTesting validation logic:")
test_bitstrings = [
    "0000",  # No subsets selected
    "1111",  # All subsets selected
    "1100",  # First two subsets
    "0011",  # Last two subsets
]

for bitstring in test_bitstrings:
    is_valid = qs.solver._is_valid_solution(bitstring)
    print(f"  Bitstring '{bitstring}': {'✓ VALID' if is_valid else '✗ INVALID'}")

# Run on Aer simulator
print("\nRunning on Aer simulator...")
result = qs.run_aer(shots=1024)
counts = result.get_counts()

# Analyze results
from sudoku_nisq.mitigation.expectation_wrapper import compute_success_expectation

success_prob = compute_success_expectation(counts, qs.solver._is_valid_solution)
print(f"\nSuccess probability: {success_prob:.4f}")

# Show top results
print("\nTop 5 measurement outcomes:")
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
for bitstring, count in sorted_counts[:5]:
    is_valid = qs.solver._is_valid_solution(bitstring)
    prob = count / sum(counts.values())
    status = "✓" if is_valid else "✗"
    print(f"  {bitstring}: {count:4d} ({prob:6.2%}) {status}")

print("\n✅ Integration test complete!")
