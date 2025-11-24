#!/usr/bin/env python3
"""
Test script for canonical encoding methods in ExactCoverProblem.

Validates:
1. to_canonical_matrix() produces canonical form
2. to_canonical_encoding() implements un(n);un(m);bits(A*) correctly
3. canonical_order_index() computes shortlex position
4. Equivalence: same canonical matrix → same encoding
"""

from src.sudoku_nisq.exact_cover_problem import ExactCoverProblem


def test_canonical_matrix():
    """Test canonical matrix computation."""
    print("\n=== Test 1: Canonical Matrix ===")
    
    # Test case: universe with non-canonical order
    universe = ['b', 'a', 'c']
    subsets = {
        'S_0': ['a', 'b'],
        'S_1': ['b', 'c'],
        'S_2': ['a', 'b']  # Duplicate of S_0
    }
    problem = ExactCoverProblem(universe, subsets)
    
    canonical_matrix, ordered_universe = problem.to_canonical_matrix()
    
    print(f"Original universe: {problem.universe}")
    print(f"Ordered universe: {ordered_universe}")
    print(f"Canonical matrix ({len(canonical_matrix)}×{len(canonical_matrix[0]) if canonical_matrix else 0}):")
    for row in canonical_matrix:
        print(f"  {row}")
    
    # Verify universe is sorted
    assert ordered_universe == ['a', 'b', 'c'], f"Universe not sorted: {ordered_universe}"
    
    # Verify columns are sorted and deduplicated
    # S_0 and S_2 are duplicates, should have only 2 columns
    assert len(canonical_matrix[0]) == 2, f"Expected 2 columns, got {len(canonical_matrix[0])}"
    
    print("✓ Canonical matrix test passed")
    return problem, canonical_matrix, ordered_universe


def test_canonical_encoding():
    """Test canonical encoding scheme."""
    print("\n=== Test 2: Canonical Encoding ===")
    
    # Test case: 2×2 identity matrix
    # Original matrix = [[1,0], [0,1]]
    # After canonicalization: columns are sorted lexicographically
    # Column [1,0] and column [0,1] → sorted gives [0,1], [1,0]
    # So canonical matrix = [[0,1], [1,0]]
    matrix = [[1, 0], [0, 1]]
    problem = ExactCoverProblem.from_incidence_matrix(matrix)
    
    encoding = problem.to_canonical_encoding()
    
    # Expected: un(2);un(2);bits(canonical_matrix)
    # un(2) = "110"
    # un(2) = "110" 
    # Canonical matrix after column sort: [[0,1], [1,0]]
    # bits = "0110" (row-major: 0,1,1,0)
    expected = "1101100110"
    
    print(f"Problem: 2×2 identity matrix")
    print(f"Encoding: {encoding}")
    print(f"Expected: {expected}")
    
    assert encoding == expected, f"Encoding mismatch: got {encoding}, expected {expected}"
    print("✓ Canonical encoding test passed")
    
    # Test minimal case (1×1 matrix with single 1)
    minimal_problem = ExactCoverProblem.from_incidence_matrix([[1]])
    minimal_encoding = minimal_problem.to_canonical_encoding()
    # un(1) = "10", un(1) = "10", bits = "1"
    expected_minimal = "10101"
    print(f"\nMinimal problem (1×1 matrix [1]) encoding: {minimal_encoding}")
    print(f"Expected: {expected_minimal} (un(1);un(1);1)")
    assert minimal_encoding == expected_minimal, f"Minimal encoding wrong: {minimal_encoding}"
    print("✓ Minimal encoding test passed")


def test_canonical_order_index():
    """Test shortlex order index computation."""
    print("\n=== Test 3: Canonical Order Index ===")
    
    # Small test cases with known indices (non-empty strings only)
    test_cases = [
        # (encoding_string, expected_index)
        ("0", 1),    # First string of length 1
        ("1", 2),    # Second string of length 1
        ("00", 3),   # 2^2 - 1 + 0 = 3
        ("01", 4),   # 2^2 - 1 + 1 = 4
        ("10", 5),   # 2^2 - 1 + 2 = 5
        ("11", 6),   # 2^2 - 1 + 3 = 6
    ]
    
    for enc_str, expected_idx in test_cases:
        # Compute index using formula
        length = len(enc_str)
        count_shorter = (1 << length) - 1
        lex_position = int(enc_str, 2)
        computed_idx = count_shorter + lex_position
        
        print(f"Encoding: '{enc_str}' → Index: {computed_idx} (expected: {expected_idx})")
        assert computed_idx == expected_idx, f"Index mismatch for '{enc_str}'"
    
    print("✓ Canonical order index test passed")
    
    # Now test with actual problem
    problem = ExactCoverProblem.from_incidence_matrix([[1, 0], [0, 1]])
    idx = problem.canonical_order_index()
    enc = problem.to_canonical_encoding()
    print(f"\n2×2 identity problem:")
    print(f"  Encoding: {enc} (length {len(enc)})")
    print(f"  Index: {idx}")
    
    # Verify index is consistent with encoding
    length = len(enc)
    expected_idx = (1 << length) - 1 + int(enc, 2)
    assert idx == expected_idx, f"Index inconsistent with encoding"
    print("✓ Problem index test passed")


def test_equivalence():
    """Test that equivalent problems have same canonical encoding."""
    print("\n=== Test 4: Equivalence Test ===")
    
    # Two representations of the same exact cover instance
    # Universe {0, 1, 2}, subsets {0,1}, {1,2}
    
    # Representation 1: canonical ordering
    problem1 = ExactCoverProblem(
        universe=[0, 1, 2],
        subsets={'S_0': [0, 1], 'S_1': [1, 2]}
    )
    
    # Representation 2: different universe ordering
    problem2 = ExactCoverProblem(
        universe=[2, 0, 1],
        subsets={'S_a': [1, 0], 'S_b': [2, 1]}
    )
    
    enc1 = problem1.to_canonical_encoding()
    enc2 = problem2.to_canonical_encoding()
    
    print(f"Problem 1 encoding: {enc1}")
    print(f"Problem 2 encoding: {enc2}")
    
    assert enc1 == enc2, "Equivalent problems should have same canonical encoding"
    
    idx1 = problem1.canonical_order_index()
    idx2 = problem2.canonical_order_index()
    
    print(f"Problem 1 index: {idx1}")
    print(f"Problem 2 index: {idx2}")
    
    assert idx1 == idx2, "Equivalent problems should have same index"
    print("✓ Equivalence test passed")


def test_sudoku_canonical_encoding():
    """Test canonical encoding for Sudoku puzzles (via exact cover)."""
    print("\n=== Test 5: Sudoku Canonical Encoding ===")
    
    # Import Sudoku puzzle if available
    try:
        from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding
        from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
        
        # Create minimal 2×2 Sudoku
        grid = [[1, 0], [0, 0]]
        puzzle = SudokuPuzzle.from_board(grid)
        
        # Get exact cover encoding
        encoder = ExactCoverEncoding(puzzle)
        
        # Get canonical encodings for both types
        enc_simple = encoder.to_canonical_encoding('simple')
        enc_pattern = encoder.to_canonical_encoding('pattern')
        
        print(f"Simple encoding length: {len(enc_simple)}")
        print(f"Pattern encoding length: {len(enc_pattern)}")
        print(f"Simple encoding (first 80 chars): {enc_simple[:80]}...")
        print(f"Pattern encoding (first 80 chars): {enc_pattern[:80]}...")
        
        # They should be different (different exact cover instances)
        assert enc_simple != enc_pattern, "Simple and pattern encodings should differ"
        
        print("✓ Sudoku canonical encoding test passed")
        
    except ImportError as e:
        print(f"⚠ Skipping Sudoku test (import failed): {e}")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Testing Canonical Encoding Implementation")
    print("=" * 60)
    
    try:
        test_canonical_matrix()
        test_canonical_encoding()
        test_canonical_order_index()
        test_equivalence()
        test_sudoku_canonical_encoding()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
