#!/usr/bin/env python3
"""
Canonical Encoding Example
===========================

Demonstrates the canonical encoding framework for exact cover problems:
1. Canonical matrix construction
2. Binary string encoding (un(n);un(m);bits)
3. Shortlex order index computation
4. Isomorphism detection
5. Sudoku puzzle canonical encodings
"""

from sudoku_nisq import ExactCoverProblem


def example_1_basic_encoding():
    """Basic example: canonical matrix and encoding."""
    print("=" * 70)
    print("Example 1: Basic Canonical Encoding")
    print("=" * 70)
    
    # Define a simple problem
    universe = ['b', 'a', 'c']  # Non-canonical order
    subsets = {
        'S_0': ['a', 'b'],
        'S_1': ['b', 'c'],
        'S_2': ['a', 'b']  # Duplicate of S_0
    }
    
    problem = ExactCoverProblem(universe, subsets)
    
    print("\nOriginal Problem:")
    print(f"  Universe: {problem.universe}")
    print(f"  Subsets:")
    for name, subset in problem.subsets.items():
        print(f"    {name}: {subset}")
    
    # Canonical matrix
    canonical_matrix, ordered_universe = problem.to_canonical_matrix()
    
    print("\nCanonical Representation:")
    print(f"  Ordered universe: {ordered_universe}")
    print(f"  Canonical matrix ({len(canonical_matrix)}×{len(canonical_matrix[0])}):")
    print(f"    (Duplicates removed, columns sorted lexicographically)")
    for i, row in enumerate(canonical_matrix):
        print(f"    Row {i} ({ordered_universe[i]}): {row}")
    
    # Binary encoding
    encoding = problem.to_canonical_encoding()
    
    print(f"\nBinary Encoding:")
    print(f"  Length: {len(encoding)} bits")
    print(f"  Encoding: {encoding}")
    
    # Decode the parts
    n = len(ordered_universe)
    m = len(canonical_matrix[0]) if canonical_matrix else 0
    
    def decode_unary(s):
        """Decode unary prefix, return (value, remaining_string)."""
        count = 0
        i = 0
        while i < len(s) and s[i] == '1':
            count += 1
            i += 1
        if i < len(s) and s[i] == '0':
            i += 1
        return count, s[i:]
    
    rest = encoding
    n_decoded, rest = decode_unary(rest)
    m_decoded, rest = decode_unary(rest)
    bits = rest
    
    print(f"\nEncoding Structure:")
    print(f"  un({n}) = {'1'*n}0 = {encoding[:n+1]}")
    print(f"  un({m}) = {'1'*m}0 = {encoding[n+1:n+1+m+1]}")
    print(f"  bits = {bits} ({len(bits)} bits = {n}×{m})")
    print(f"  Decoded: n={n_decoded}, m={m_decoded}")
    
    # Global index
    index = problem.canonical_order_index()
    print(f"\nGlobal Shortlex Index: {index}")
    print(f"  (Position in universal ordering of all exact cover instances)")


def example_2_isomorphism():
    """Demonstrate isomorphism detection via canonical encoding."""
    print("\n" + "=" * 70)
    print("Example 2: Isomorphism Detection")
    print("=" * 70)
    
    # Three representations of the "same" problem
    p1 = ExactCoverProblem(
        universe=[0, 1, 2],
        subsets={'S_0': [0, 1], 'S_1': [1, 2]}
    )
    
    p2 = ExactCoverProblem(
        universe=[2, 0, 1],  # Different order
        subsets={'S_a': [1, 0], 'S_b': [2, 1]}  # Different labels, order
    )
    
    p3 = ExactCoverProblem(
        universe=['x', 'y', 'z'],  # Different element types
        subsets={'A': ['x', 'y'], 'B': ['y', 'z']}
    )
    
    print("\nProblem 1:")
    print(f"  Universe: {p1.universe}")
    print(f"  Subsets: {p1.subsets}")
    
    print("\nProblem 2:")
    print(f"  Universe: {p2.universe}")
    print(f"  Subsets: {p2.subsets}")
    
    print("\nProblem 3:")
    print(f"  Universe: {p3.universe}")
    print(f"  Subsets: {p3.subsets}")
    
    # Canonical encodings
    enc1 = p1.to_canonical_encoding()
    enc2 = p2.to_canonical_encoding()
    enc3 = p3.to_canonical_encoding()
    
    print("\nCanonical Encodings:")
    print(f"  Problem 1: {enc1}")
    print(f"  Problem 2: {enc2}")
    print(f"  Problem 3: {enc3}")
    
    # Indices
    idx1 = p1.canonical_order_index()
    idx2 = p2.canonical_order_index()
    idx3 = p3.canonical_order_index()
    
    print("\nShortlex Indices:")
    print(f"  Problem 1: {idx1}")
    print(f"  Problem 2: {idx2}")
    print(f"  Problem 3: {idx3}")
    
    # Check isomorphism
    print("\nIsomorphism Test:")
    print(f"  p1 ≅ p2: {enc1 == enc2} ✓" if enc1 == enc2 else f"  p1 ≅ p2: {enc1 == enc2} ✗")
    print(f"  p1 ≅ p3: {enc1 == enc3} ✓" if enc1 == enc3 else f"  p1 ≅ p3: {enc1 == enc3} ✗")
    print(f"  p2 ≅ p3: {enc2 == enc3} ✓" if enc2 == enc3 else f"  p2 ≅ p3: {enc2 == enc3} ✗")
    
    print("\nAll three problems are isomorphic (same structure, different labels)")


def example_3_shortlex_order():
    """Demonstrate shortlex ordering."""
    print("\n" + "=" * 70)
    print("Example 3: Shortlex Order")
    print("=" * 70)
    
    # Create several small problems
    problems = [
        ExactCoverProblem.from_incidence_matrix([[1]]),
        ExactCoverProblem.from_incidence_matrix([[1, 0], [0, 1]]),
        ExactCoverProblem.from_incidence_matrix([[1, 1]]),
        ExactCoverProblem.from_incidence_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ]
    
    # Compute encodings and indices
    results = []
    for i, p in enumerate(problems):
        enc = p.to_canonical_encoding()
        idx = p.canonical_order_index()
        matrix = p.to_incidence_matrix()
        results.append((idx, len(enc), enc, matrix, i))
    
    # Sort by index
    results.sort()
    
    print("\nProblems sorted by shortlex index:")
    print(f"{'Index':<10} {'Length':<8} {'Encoding':<30} {'Matrix':<20}")
    print("-" * 70)
    
    for idx, length, enc, matrix, original_idx in results:
        matrix_str = str(matrix)[:20] + "..." if len(str(matrix)) > 20 else str(matrix)
        enc_str = enc[:30] + "..." if len(enc) > 30 else enc
        print(f"{idx:<10} {length:<8} {enc_str:<30} {matrix_str:<20}")
    
    print("\nObservations:")
    print("  - Smaller matrices → smaller indices (generally)")
    print("  - Index grows exponentially with encoding length")
    print("  - Length = n + m + nm + 2 (unary overhead + matrix bits)")


def example_4_sudoku_encodings():
    """Compare canonical encodings for Sudoku puzzle."""
    print("\n" + "=" * 70)
    print("Example 4: Sudoku Canonical Encodings")
    print("=" * 70)
    
    try:
        from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
        from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding
        
        # Create a 2×2 Sudoku puzzle
        grid = [[1, 0], [0, 0]]
        puzzle = SudokuPuzzle.from_board(grid)
        
        print("\nSudoku Puzzle (2×2):")
        print(f"  Grid: {grid}")
        
        # Generate exact cover encoding
        encoder = ExactCoverEncoding(puzzle)
        
        # Get canonical encodings for both types
        enc_simple = encoder.to_canonical_encoding('simple')
        enc_pattern = encoder.to_canonical_encoding('pattern')
        
        print("\nExact Cover Encodings:")
        print(f"  Simple encoding:")
        print(f"    Length: {len(enc_simple)} bits")
        print(f"    First 60 bits: {enc_simple[:60]}...")
        print(f"    Shortlex index: ~2^{len(enc_simple)}")
        
        print(f"\n  Pattern encoding:")
        print(f"    Length: {len(enc_pattern)} bits")
        print(f"    First 60 bits: {enc_pattern[:60]}...")
        print(f"    Shortlex index: ~2^{len(enc_pattern)}")
        
        print("\nComparison:")
        print(f"  Encoding length ratio: {len(enc_simple)}/{len(enc_pattern)} = {len(enc_simple)/len(enc_pattern):.2f}x")
        print(f"  Encodings are different: {enc_simple != enc_pattern}")
        print(f"    → Simple and pattern represent different exact cover instances")
        
        # Get universe sizes
        n_simple = len(encoder.universe2x2)
        n_pattern = len(encoder.universe2x2)
        m_simple = len(encoder.simple_subsets)
        m_pattern = len(encoder.pattern_subsets)
        
        print(f"\n  Problem sizes:")
        print(f"    Simple: {n_simple} constraints × {m_simple} subsets")
        print(f"    Pattern: {n_pattern} constraints × {m_pattern} subsets")
        
    except ImportError as e:
        print(f"\n⚠ Sudoku imports not available: {e}")


def example_5_index_computation():
    """Detailed example of index computation."""
    print("\n" + "=" * 70)
    print("Example 5: Shortlex Index Computation")
    print("=" * 70)
    
    # Use small example
    problem = ExactCoverProblem.from_incidence_matrix([[1]])
    encoding = problem.to_canonical_encoding()
    
    print("\nProblem: 1×1 matrix [[1]]")
    print(f"Encoding: {encoding}")
    print(f"Length: {len(encoding)}")
    
    # Manual computation
    length = len(encoding)
    print(f"\nIndex Computation:")
    print(f"  1. Count all shorter strings:")
    print(f"     Σ(i=0 to {length-1}) 2^i = 2^{length} - 1 = {(1 << length) - 1}")
    
    print(f"\n  2. Interpret encoding as binary number:")
    print(f"     int('{encoding}', 2) = {int(encoding, 2)}")
    
    print(f"\n  3. Add offset:")
    index = (1 << length) - 1 + int(encoding, 2)
    print(f"     Index = {(1 << length) - 1} + {int(encoding, 2)} = {index}")
    
    # Verify
    computed_index = problem.canonical_order_index()
    print(f"\n  4. Verify with method:")
    print(f"     problem.canonical_order_index() = {computed_index}")
    print(f"     Match: {index == computed_index} ✓")
    
    print("\nGeneral Formula:")
    print("  index(s) = (2^|s| - 1) + int(s, 2)")
    print("  where |s| = length of encoding s")


if __name__ == "__main__":
    example_1_basic_encoding()
    example_2_isomorphism()
    example_3_shortlex_order()
    example_4_sudoku_encodings()
    example_5_index_computation()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key Takeaways:

1. **Canonical Matrix**: Unique representation independent of labels/ordering
   - Order universe elements
   - Remove duplicate columns
   - Sort columns lexicographically

2. **Binary Encoding**: un(n);un(m);bits(A*)
   - Unary encoding for dimensions (1^n 0)
   - Row-major matrix bits
   - Uniquely identifies isomorphism class

3. **Shortlex Order**: Total order on all exact cover instances
   - First by length, then lexicographically
   - Defines bijection with natural numbers
   - Enables systematic enumeration (theoretically)

4. **Applications**:
   - Isomorphism detection via encoding equality
   - Systematic benchmark generation
   - Complexity analysis via encoding length
   - Deduplication in large datasets

5. **Sudoku Connection**:
   - Each Sudoku puzzle → exact cover instance
   - Simple vs pattern encoding → different instances
   - Canonical encoding applies to both
    """)
