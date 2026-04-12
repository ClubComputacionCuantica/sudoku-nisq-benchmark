"""
Quick test of generic exact cover functionality.
"""

from sudoku_nisq import ExactCoverProblem, QExactCover

def test_basic_problem():
    """Test creating and validating a basic exact cover problem."""
    print("Testing ExactCoverProblem creation...")
    
    universe = [0, 1, 2]
    subsets = {'S_0': [0, 1], 'S_1': [1, 2]}
    problem = ExactCoverProblem(universe, subsets)
    
    assert len(problem.universe) == 3
    assert len(problem.subsets) == 2
    print("  ✓ Basic problem creation works")
    
    # Test canonicalization
    n, canonical = problem.canonicalize()
    assert n == 3
    assert len(canonical) == 2
    print("  ✓ Canonicalization works")
    
    # Test hashing
    hash1 = problem.get_hash()
    assert len(hash1) == 64  # SHA256 hex is 64 chars
    print("  ✓ Hashing works")
    
    # Test incidence matrix
    matrix = problem.to_incidence_matrix()
    assert len(matrix) == 3  # 3 rows
    assert len(matrix[0]) == 2  # 2 columns
    print("  ✓ Incidence matrix conversion works")


def test_qexact_cover():
    """Test QExactCover interface."""
    print("\nTesting QExactCover interface...")
    
    problem = ExactCoverProblem.create_small_example()
    qec = QExactCover(problem)
    
    assert len(qec.universe) == 4
    assert len(qec.subsets) == 6
    print("  ✓ QExactCover initialization works")
    
    # Test resource estimation
    resources = qec.report_resources()
    assert 'problem' in resources
    assert 'estimated' in resources
    print(f"  ✓ Resource estimation works (needs {resources['estimated']['n_qubits']} qubits)")


def test_count_solutions():
    """Test solution counting."""
    print("\nTesting solution counting...")
    
    # Test simple case
    universe = [0, 1]
    subsets = {'S_0': [0], 'S_1': [1]}
    problem = ExactCoverProblem(universe, subsets)
    count = problem.count_solutions()
    assert count == 1
    print(f"  ✓ Simple problem: {count} solution")
    
    # Test multiple solutions
    universe = [0, 1, 2]
    subsets = {
        'S_0': [0],
        'S_1': [1], 
        'S_2': [2],
        'S_3': [1, 2]
    }
    problem = ExactCoverProblem(universe, subsets)
    count = problem.count_solutions()
    assert count == 2  # Solutions: {S_0, S_1, S_2} and {S_0, S_3}
    print(f"  ✓ Multiple solutions: {count} solutions")
    
    # Test max_solutions
    count_limited = problem.count_solutions(max_solutions=1)
    assert count_limited >= 1
    print(f"  ✓ Early termination: stopped at {count_limited}")
    
    # Test small example (with limit to prevent long execution)
    example = ExactCoverProblem.create_small_example()
    count_example = example.count_solutions(max_solutions=10)
    print(f"  ✓ Small example: {count_example} solution(s) found (may be capped at 10)")


def test_circuit_building():
    """Test building quantum circuit."""
    print("\nTesting circuit building...")
    
    # Use a small problem
    universe = [0, 1]
    subsets = {'S_0': [0], 'S_1': [1]}
    problem = ExactCoverProblem(universe, subsets, num_solutions=1)
    
    qec = QExactCover(problem)
    circuit = qec.build_circuit(sdk="qiskit")
    
    assert circuit is not None
    assert hasattr(circuit, 'num_qubits')
    print(f"  ✓ Circuit built successfully ({circuit.num_qubits} qubits)")


def test_enumeration():
    """Test problem enumeration."""
    print("\nTesting problem enumeration...")
    
    count = 0
    # Use smaller limits to avoid combinatorial explosion
    for problem in ExactCoverProblem.enumerate_instances(max_n=2, max_m=2, max_total=4):
        count += 1
        if count >= 5:
            break
    
    assert count >= 5
    print(f"  ✓ Problem enumeration works (generated {count} instances)")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERIC EXACT COVER - QUICK TEST")
    print("=" * 60)
    print()
    
    try:
        test_basic_problem()
        test_qexact_cover()
        test_count_solutions()
        test_circuit_building()
        test_enumeration()
        
        print()
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("The generic exact cover functionality is working correctly.")
        print("Try running: python examples/exact_cover_benchmark.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
