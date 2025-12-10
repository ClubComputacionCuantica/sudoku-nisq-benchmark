# Canonical Encoding Reference

Canonical encoding provides unique binary representations for exact cover problem instances, enabling isomorphism detection, problem comparison, and systematic benchmarking.

## Overview

Every exact cover problem can be represented as a unique binary string that identifies its isomorphism class. Two problems that differ only in element labels (e.g., using numbers vs. letters) produce the same canonical encoding.

This framework enables:
- **Isomorphism detection**: Determine if two problems have the same structure
- **Deduplication**: Identify and remove structurally identical instances
- **Global ordering**: Position any instance in a universal shortlex ordering
- **Systematic benchmarking**: Navigate the space of all exact cover problems

## Quick Reference

The canonical encoding API is centered on three methods of
`ExactCoverProblem`:

- `to_canonical_matrix()` – canonical incidence matrix up to relabeling
- `to_canonical_encoding()` – binary string representing the isomorphism class
- `canonical_order_index()` – position of the encoding in shortlex order

```python
from sudoku_nisq import ExactCoverProblem

universe = [0, 1, 2, 3]
subsets = {
    'S_0': [0, 3],
    'S_1': [1, 2],
    'S_2': [0, 1, 2],
}
problem = ExactCoverProblem(universe, subsets, num_solutions=1)

canonical_matrix, ordered_universe = problem.to_canonical_matrix()
encoding = problem.to_canonical_encoding()
index = problem.canonical_order_index()
```

## Common Usage Patterns

### Isomorphism Detection

Check if two problems have the same structure:

```python
from sudoku_nisq import ExactCoverProblem

# Same structure, different labels
p1 = ExactCoverProblem([0, 1, 2], {'S_0': [0, 1], 'S_1': [1, 2]})
p2 = ExactCoverProblem(['a', 'b', 'c'], {'A': ['a', 'b'], 'B': ['b', 'c']})
p3 = ExactCoverProblem(['x', 'y', 'z'], {'X': ['x', 'y'], 'Y': ['y', 'z']})

# All produce identical encodings
enc1 = p1.to_canonical_encoding()
enc2 = p2.to_canonical_encoding()
enc3 = p3.to_canonical_encoding()

print(f"Are isomorphic: {enc1 == enc2 == enc3}")  # True
print(f"Same index: {p1.canonical_order_index() == p2.canonical_order_index()}")  # True
```

### Deduplication

Remove structurally identical problems from a dataset:

```python
problems = [...]  # List of ExactCoverProblem instances

# Get unique encodings
unique_encodings = {}
for problem in problems:
    encoding = problem.to_canonical_encoding()
    if encoding not in unique_encodings:
        unique_encodings[encoding] = problem

print(f"Original: {len(problems)} problems")
print(f"Unique: {len(unique_encodings)} structural classes")
```

### Problem Comparison

Order problems by their canonical indices:

```python
p1 = ExactCoverProblem([0, 1], {'S1': [0]})
p2 = ExactCoverProblem([0, 1, 2], {'S1': [0, 1]})

idx1 = p1.canonical_order_index()
idx2 = p2.canonical_order_index()

if idx1 < idx2:
    print("p1 comes before p2 in universal ordering")
elif idx1 == idx2:
    print("p1 and p2 are isomorphic")
else:
    print("p2 comes before p1 in universal ordering")
```

### Sudoku Integration

Get canonical encodings for different Sudoku encoding strategies:

```python
from sudoku_nisq import QSudoku
from sudoku_nisq.encodings import ExactCoverEncoding

# Create puzzle
puzzle = QSudoku.from_board([[1, 0], [0, 0]])
encoder = ExactCoverEncoding(puzzle)

# Different encoding strategies produce different exact cover instances
enc_simple = encoder.to_canonical_encoding('simple')
enc_pattern = encoder.to_canonical_encoding('pattern')

print(f"Simple encoding: {len(enc_simple)} bits")
print(f"Pattern encoding: {len(enc_pattern)} bits")
print(f"Are different: {enc_simple != enc_pattern}")  # True
```

## Encoding Format Details

### Unary Encoding and Shortlex

We use unary encoding $\operatorname{un}(k)$ (a run of $k$ ones followed by
`0`) and the standard **shortlex** order on binary strings (by length,
then lexicographically). These are described in more detail in the
"Binary String Encoding" and "Global Total Order (Shortlex)" sections
below.

### Example Walkthrough: $2\times 2$ Identity Matrix

```python
# Problem with 2 elements, 2 subsets
# After canonicalization: [[0,1], [1,0]] (columns sorted)

# Parameters:
n = 2     # universe size
m_star = 2    # unique subset count (m*)

"""
Encoding construction (mathematical form):

un(2) = 110 \quad \text{(universe size)}\\
un(2) = 110 \quad \text{(subset count)}\\
\operatorname{bits}(A^*) = 0110 \quad \text{(matrix, row-major)}\\
	ext{encoding} = 110\,110\,0110
"""

"""
Index computation:

\ell = 10\\
	ext{index} = (2^{10} - 1) + \operatorname{int}_2(1101100110) = 1023 + 870 = 1893
"""

"""
Bounds check:

	ext{For }\ell=10:\quad 2^{10} - 1 = 1023 \le 1893 \le 2^{11} - 2 = 2046\;\checkmark
"""
```

## Complexity Analysis

### Time Complexity

- `to_canonical_matrix()`: **O(m·n·log(m))**
  - Sorting m columns of length n
- `to_canonical_encoding()`: **O(n·m*)**
  - Linear scan of canonical matrix
- `canonical_order_index()`: **O(|encoding|)**
  - Linear in encoding length

### Space Complexity

- Canonical matrix: **O(n·m*)** (stored explicitly)
- Encoding string: **O(n + m* + n·m*)** bits
- Index: **O(1)** (single integer, potentially very large)

### Encoding Length

Total bits: **n + m* + n·m* + 2**

- Unary overhead: n + m* + 2
- Matrix bits: n·m*
- Dominated by matrix term when n ≈ m*

### Index Magnitude

For encoding of length $\ell$:
- **Minimum** (string $0\dots 0$): $2^{\ell} - 1$
- **Maximum** (string $1\dots 1$): $2^{\ell+1} - 2$
- **Approximate range**: $2^{\ell} \lesssim \text{index} < 2^{\ell+1}$

**Examples** (with approximate encoding length $\ell$):
- $2\times 2$ matrix ($\ell\approx 10$): index $\sim 10^{3}$
- $10\times 10$ matrix ($\ell\approx 122$): index $\sim 10^{37}$
- $100\times 100$ matrix ($\ell\approx 10{,}202$): index $\sim 10^{3071}$

## Mathematical Properties

### Uniqueness Theorem

**Theorem**: Two exact cover instances have the same canonical encoding if and only if they are isomorphic.

**Proof sketch**: The canonical form removes all degrees of freedom in labeling and ordering, leaving only the structural information.

### Order Properties

The shortlex ordering on canonical encodings induces a total order on exact cover instances:

- **Total**: Every pair of instances is comparable
- **Well-founded**: Every non-empty set has a minimum element
- **Computable**: Can determine I₁ ≺ I₂ by comparing encodings

### Bijection

Canonical encoding defines a bijection between:

$$
\frac{\text{ExactCover Instances}}{\cong}\;\longleftrightarrow\; \{0,1\}^*\;\longleftrightarrow\; \mathbb{N}_+\\
	ext{(isomorphism classes)}\qquad\text{(binary strings)}\qquad\text{(indices)}
$$

## Practical Limits

### Feasible Operations

✅ **Works well for**:
- Encoding instances up to ~100×100 matrices
- Comparing any two instances
- Deduplicating large datasets
- Detecting isomorphism

❌ **Not feasible for**:
- Enumerating instances to very large index (doubly exponential)
- Materializing indices exceeding 10¹⁰⁰⁰
- Reverse lookup: index → instance (computationally hard)
- Dense enumeration of high-dimensional problem spaces

## Use Cases in Practice

### Benchmarking

Systematically explore problem space:

```python
from sudoku_nisq import ExactCoverProblem

# Generate all small instances for comprehensive testing
for problem in ExactCoverProblem.enumerate_instances(max_n=3, max_m=3):
    encoding = problem.to_canonical_encoding()
    index = problem.canonical_order_index()
    
    # Run quantum solver
    result = solve_quantum(problem)
    
    # Store results indexed by canonical encoding
    results_db[encoding] = {
        'index': index,
        'problem_size': (len(problem.universe), len(problem.subsets)),
        'success_rate': result.success_rate,
        'circuit_depth': result.depth
    }
```

### Cache Management

Use encoding as cache key:

```python
import hashlib

def get_cached_circuit(problem):
    encoding = problem.to_canonical_encoding()
    cache_key = hashlib.sha256(encoding.encode()).hexdigest()
    cache_path = f".cache/{cache_key}.pkl"
    
    if os.path.exists(cache_path):
        return load_circuit(cache_path)
    else:
        circuit = build_circuit(problem)
        save_circuit(circuit, cache_path)
        return circuit
```

### Research Analysis

Study hardness as function of structure:

```python
import matplotlib.pyplot as plt

# Collect problem characteristics
problems = [...]
indices = [p.canonical_order_index() for p in problems]
depths = [compute_circuit_depth(p) for p in problems]

# Plot complexity vs. canonical index
plt.scatter(indices, depths)
plt.xlabel('Canonical Index')
plt.ylabel('Circuit Depth')
plt.xscale('log')
plt.title('Problem Hardness vs. Canonical Position')
plt.show()
```