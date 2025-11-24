# Canonical Encoding Quick Reference

## Quick Start

```python
from sudoku_nisq import ExactCoverProblem

# Create problem
universe = ['a', 'b', 'c']
subsets = {'S_0': ['a', 'b'], 'S_1': ['b', 'c']}
problem = ExactCoverProblem(universe, subsets)

# Get canonical representation
canonical_matrix, ordered_universe = problem.to_canonical_matrix()
# → ordered_universe = ['a', 'b', 'c']
# → canonical_matrix = [[0,1], [1,1], [1,0]] (deduped + sorted columns)

# Get binary encoding
encoding = problem.to_canonical_encoding()
# → "1110110011110" (un(3);un(2);bits)

# Get global index
index = problem.canonical_order_index()
# → 15773 (position in shortlex order)
```

## Three Core Methods

### 1. `to_canonical_matrix()`

**Returns**: `(canonical_matrix, ordered_universe)`

**Algorithm**:
1. Sort universe elements
2. Convert subsets → column bitvectors
3. Remove duplicate columns
4. Sort columns lexicographically

**Use Case**: Get unique matrix representation independent of labels

### 2. `to_canonical_encoding()`

**Returns**: `str` (binary string)

**Format**: `un(n) ; un(m) ; bits(A*)`
- `un(k) = "1"*k + "0"` (unary encoding)
- `bits(A*)` = row-major matrix bits

**Use Case**: Unique string identifier for isomorphism class

### 3. `canonical_order_index()`

**Returns**: `int` (global position)

**Formula**: `index = (2^length - 1) + int(encoding, 2)`

**Use Case**: Position in universal ordering of all exact cover instances

## Sudoku Integration

```python
from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding

puzzle = SudokuPuzzle.from_board([[1, 0], [0, 0]])
encoder = ExactCoverEncoding(puzzle)

# Get canonical encodings
enc_simple = encoder.to_canonical_encoding('simple')
enc_pattern = encoder.to_canonical_encoding('pattern')

# Different encodings (different EC instances)
assert enc_simple != enc_pattern
```

## Common Patterns

### Isomorphism Check

```python
p1 = ExactCoverProblem([0,1,2], {'S_0': [0,1]})
p2 = ExactCoverProblem(['a','b','c'], {'S_0': ['a','b']})

# Same structure?
is_isomorphic = p1.to_canonical_encoding() == p2.to_canonical_encoding()
# → True
```

### Deduplication

```python
problems = [...]  # List of ExactCoverProblems
unique_encodings = set(p.to_canonical_encoding() for p in problems)
print(f"Found {len(unique_encodings)} unique structures")
```

### Comparison

```python
p1, p2 = ExactCoverProblem(...), ExactCoverProblem(...)

idx1, idx2 = p1.canonical_order_index(), p2.canonical_order_index()

if idx1 < idx2:
    print("p1 comes before p2 in global order")
elif idx1 == idx2:
    print("p1 and p2 are isomorphic")
else:
    print("p2 comes before p1 in global order")
```

## Encoding Details

### Unary Encoding

For all k ≥ 0:

| k | un(k) | Description |
|---|-------|-------------|
| 0 | "0" | k ones + "0" |
| 1 | "10" | k ones + "0" |
| 2 | "110" | k ones + "0" |
| 3 | "1110" | k ones + "0" |
| k | "1"*k + "0" | k ones + "0" |

### Shortlex Order

(Non-empty strings only, since canonical encodings are never empty)

| Index | String | Description |
|-------|--------|-------------|
| 1 | "0" | First length-1 |
| 2 | "1" | Second length-1 |
| 3 | "00" | First length-2 |
| 4 | "01" | ... |
| 5 | "10" | ... |
| 6 | "11" | Last length-2 |
| 7 | "000" | First length-3 |

### Example: 2×2 Identity

```
Original matrix: [[1, 0], [0, 1]]
After canonicalization: [[0, 1], [1, 0]]  (columns sorted)

n=2, m=2
un(2) = "110"
un(2) = "110"
bits = "0110" (row-major: 0,1,1,0)

Encoding: "1101100110"
Length: 10 bits
Index: (2^10 - 1) + int("1101100110", 2)
     = 1023 + 870
     = 1893

Bounds: For length ℓ=10, indices range from 2^10 - 1 = 1023 to 2^11 - 2 = 2046
```

## Complexity

### Time Complexity
- `to_canonical_matrix()`: O(m·n·log(m))
- `to_canonical_encoding()`: O(n·m)
- `canonical_order_index()`: O(|encoding|)

### Space Complexity
- Matrix: O(n·m) memory
- Encoding: O(n + m + n·m) bits
- Index: O(1) (single int, may be huge)

### Encoding Length
Total bits: `n + m + n·m + 2`
- Unary overhead: `n + m + 2`
- Matrix bits: `n·m`

### Index Magnitude
For encoding of length ℓ:
- Minimum (string "0...0"): 2^ℓ - 1
- Maximum (string "1...1"): 2^(ℓ+1) - 2
- Approximate range: 2^ℓ ≲ index < 2^(ℓ+1)

Examples (with encoding length ℓ):
- 2×2 matrix (ℓ≈10): index ~10³
- 10×10 matrix (ℓ≈122): index ~10³⁷
- 100×100 matrix (ℓ≈10202): index ~10³⁰⁷¹

## Mathematical Properties

### Uniqueness Theorem
Two exact cover instances have the same canonical encoding ⟺ they are isomorphic

### Order Properties
The induced order ≺ on exact cover instances is:
- **Total**: Every pair is comparable
- **Well-founded**: Every non-empty set has minimum
- **Computable**: Can compare any two instances

### Bijection
Canonical encoding defines bijection:
```
ExactCover Instances / ≅  ←→  {0,1}*  ←→  ℕ
   (isomorphism classes)    (strings)  (indices)
```

## Practical Limits

### Feasible Operations
- ✅ Encoding: Up to ~100×100 matrices
- ✅ Comparison: Any two instances
- ✅ Deduplication: Large datasets

### Infeasible Operations
- ❌ Enumerate all instances to large index (doubly exponential)
- ❌ Materialize huge indices (>10^1000)
- ❌ Reverse lookup: index → instance (computationally hard)

## See Also

- **Full Documentation**: `docs/guide/exact_cover.md`
- **Examples**: `examples/canonical_encoding_demo.py`
- **Tests**: `test_canonical_encoding.py`
- **Summary**: `CANONICAL_ENCODING_SUMMARY.md`

## References

```python
from sudoku_nisq import ExactCoverProblem

# API documentation
help(ExactCoverProblem.to_canonical_matrix)
help(ExactCoverProblem.to_canonical_encoding)
help(ExactCoverProblem.canonical_order_index)
```
