# Generic Exact Cover Problems

This guide explains how to use `sudoku-nisq-benchmark` to solve arbitrary exact cover problems beyond Sudoku puzzles.

## Overview

An **exact cover problem** consists of:
- A universe $U$ of elements that must be covered
- A collection $\mathcal{S}$ of subsets of $U$

The goal is to find a subcollection of $\mathcal{S}$ that covers each element of $U$ **exactly once**.

### Why Generic Exact Cover?

While Sudoku puzzles are interesting exact cover instances, they have limitations for quantum algorithm research:

1. **Minimum size constraint**: Even the smallest Sudoku (2×2) requires substantial quantum resources
2. **Structured constraints**: Sudoku has specific row/column/subgrid structure
3. **Limited diversity**: The space of all exact cover problems is much larger than Sudoku alone

By supporting generic exact cover instances, you can:
- Test algorithms on smaller problems that fit current quantum hardware
- Explore the full space of exact cover problems
- Validate implementations before scaling to Sudoku
- Benchmark across diverse problem structures

## Mathematical Framework

### Canonical Representation

Every exact cover instance can be mapped to a canonical form:

```python
U_n = {0, 1, ..., n-1}  # Indexed universe
S = [{subset_0}, {subset_1}, ...]  # Subsets as index sets
```

Two exact cover problems that differ only in element labels are **isomorphic** and produce the same canonical representation.

### Incidence Matrix

Any exact cover instance $(U, \mathcal{S})$ with $|U| = n$ and $|\mathcal{S}| = m$ can be represented as an $n \times m$ binary matrix $A$ where:

$$A_{ij} = \begin{cases} 1 & \text{if element } i \in S_j \\ 0 & \text{otherwise} \end{cases}$$

### Relationship to Sudoku

Every Sudoku puzzle maps to an exact cover instance:
- **Universe**: Cell positions + row/digit + column/digit + subgrid/digit constraints
- **Subsets**: Each possible cell assignment covers exactly 4 constraints

However, most exact cover instances are **not** Sudoku puzzles. The Sudoku embedding is a structured subfamily of the general exact cover space.

## Quick Start

### Creating a Problem

```python
from sudoku_nisq import ExactCoverProblem, QExactCover

# Define universe and subsets
universe = [0, 1, 2, 3]
subsets = {
    'S_0': [0, 3],      # Covers elements 0 and 3
    'S_1': [1, 2],      # Covers elements 1 and 2  
    'S_2': [0, 1, 2],   # Alternative covering
}

# Create problem
problem = ExactCoverProblem(
    universe=universe,
    subsets=subsets,
    num_solutions=1
)

# Validate automatically on construction
# Raises ValueError if invalid
```

### Solving with Quantum Algorithm

```python
# Create quantum solver interface
qec = QExactCover(problem)

# Build quantum circuit
circuit = qec.build_circuit(sdk="qiskit")

# Run on Aer simulator
result = qec.run_aer(shots=1024)

# Analyze results
print(result['counts'])
```

#### Inspecting Selected Subsets

Decode a top bitstring to see which subsets were chosen:

```python
counts = result['counts']
bitstring, _ = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0]
bs = bitstring if isinstance(bitstring, str) else ''.join(str(b) for b in bitstring)
decoded = qec.solver.decode_bitstring(bs)
print("Selected indices:", decoded['selected_indices'])
print("Selected subsets (sample):", list(decoded['selected_subsets'].items())[:3])
```

Note: Sudoku board filling applies when using `QSudoku` (Sudoku mode). In generic exact-cover usage, focus on `selected_subsets`.

### Resource Estimation

```python
resources = qec.report_resources()

print(f"Qubits required: {resources['estimated']['n_qubits']}")
print(f"Estimated gates: {resources['estimated']['n_gates']}")
```

## Canonical Encoding (overview)

`ExactCoverProblem` supports a canonical encoding that maps any instance to a
unique binary string, up to relabeling of universe elements. This is mainly
useful for:

- Detecting isomorphic instances
- Deduplicating benchmark datasets
- Defining a global ordering over all finite exact cover problems

For typical usage you only need two methods:

```python
encoding = problem.to_canonical_encoding()
index = problem.canonical_order_index()
```

See `{doc}`canonical_encoding` for the full mathematical definition,
complexity analysis, and additional usage patterns.

## Examples

### Complete Workflow

```python
from sudoku_nisq import ExactCoverProblem, QExactCover

# 1. Define problem
universe = [0, 1, 2, 3]
subsets = {
    'S_0': [0, 3],
    'S_1': [1, 2],
    'S_2': [0, 1, 2],
}
problem = ExactCoverProblem(universe, subsets, num_solutions=1)

# 2. Create quantum solver
qec = QExactCover(problem)

# 3. Check resources
resources = qec.report_resources()
print(f"Requires {resources['estimated']['n_qubits']} qubits")

# 4. Build and run
circuit = qec.build_circuit()
result = qec.run_aer(shots=1024)

# 5. Analyze results
for bitstring, count in sorted(result['counts'].items(), 
                               key=lambda x: x[1], reverse=True)[:3]:
    selected = [f"S_{i}" for i, bit in enumerate(bitstring) if bit == '1']
    print(f"{bitstring}: {count} shots → subsets {selected}")
```

### Enumerating Small Instances

```python
# Generate all 2×2 exact cover instances
count = 0
for problem in ExactCoverProblem.enumerate_instances(max_n=2, max_m=2):
    count += 1
    print(f"Instance {count}: {problem.universe}, {problem.subsets}")

print(f"Total instances: {count}")  # 16 instances
```

### Using Incidence Matrices

```python
# Create from matrix
matrix = [
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0]
]
problem = ExactCoverProblem.from_incidence_matrix(matrix, num_solutions=2)

# Convert back to matrix
recovered = problem.to_incidence_matrix()
assert recovered == matrix
```

### Benchmark Comparison

See `examples/exact_cover_benchmark.py` for a complete comparison between generic exact cover and Sudoku instances.

```bash
python examples/exact_cover_benchmark.py
```

## Design Philosophy

### QExactCover vs QSudoku

| Feature | QExactCover | QSudoku |
|---------|------------|---------|
| **Scope** | Generic exact cover | Sudoku puzzles only |
| **API** | Minimal, focused | Comprehensive, feature-rich |
| **Backends** | Aer simulator only | IBM, Quantinuum, Aer, Braket |
| **Visualization** | Basic | Puzzle plots, counts plots |
| **Use case** | Algorithm testing, small benchmarks | Production solving, research |

**When to use QExactCover:**
- Testing quantum algorithms on small instances
- Benchmarking beyond Sudoku structure
- Validating implementations
- Exploring general exact cover space

**When to use QSudoku:**
- Solving actual Sudoku puzzles
- Production quantum execution
- Multi-backend workflows
- Comprehensive result analysis

### Encoding Strategy

For generic exact cover problems, only **simple encoding** is supported:
- Each subset becomes one possibility in the quantum state
- No pattern-based optimizations (those are Sudoku-specific)
- Direct mapping from classical subsets to quantum superposition

Sudoku's `pattern` encoding leverages Sudoku structure (full-row digit placements) and doesn't generalize to arbitrary exact cover instances.

## Advanced Topics

### Canonicalization and Isomorphism

Two exact cover problems are **isomorphic** if they differ only in element labels:

```python
p1 = ExactCoverProblem([1, 2, 3], {'S_0': [1, 2]})
p2 = ExactCoverProblem(['a', 'b', 'c'], {'S_0': ['a', 'b']})

# Same canonical form
assert p1.get_hash() == p2.get_hash()
```

This is useful for:
- Avoiding redundant circuit compilations
- Recognizing equivalent problems
- Caching optimization

### Solution Validation

The quantum solver validates measurement results:

```python
# Bitstring "101" means subsets S_0 and S_2 are selected
# Valid if S_0 ∪ S_2 covers all universe elements exactly once
```

Access validation through the solver:

```python
qec.solver._is_valid_solution("101")  # Returns bool
```

### Integration with ExactCoverQuantumSolver

QExactCover wraps `ExactCoverQuantumSolver` with direct problem input:

```python
from sudoku_nisq import ExactCoverQuantumSolver, ExactCoverProblem

problem = ExactCoverProblem(universe, subsets)

# Direct solver instantiation
solver = ExactCoverQuantumSolver(
    exact_cover_problem=problem,
    metadata_manager=None,
    encoding="simple"
)
```

This bypasses the Sudoku-specific `ExactCoverEncoding` layer.

## Canonical Encoding Framework

### Theoretical Foundation

Every exact cover instance can be embedded into a **global total order** through canonical encoding. This framework enables:
- Rigorous comparison between problem instances
- Enumeration of the complete problem space
- Formal complexity analysis
- Systematic benchmark generation

### Canonical Incidence Matrix

Given an exact cover instance $(U, \mathcal{S})$, the **canonical incidence matrix** $A^*$ is constructed as:

1. **Universe Ordering**: Order elements $U = \{u_0, u_1, \ldots, u_{n-1}\}$ canonically (e.g., sorted)
2. **Column Vectors**: For each subset $S_j \in \mathcal{S}$, form column vector $b_j$ where $(b_j)_i = 1$ iff $u_i \in S_j$
3. **Deduplication**: Remove duplicate column vectors
4. **Lexicographic Sort**: Order remaining columns by lexicographic comparison (as bitstrings)

**Result:** A unique $n \times m^*$ binary matrix $A^* \in \{0,1\}^{n \times m^*}$ that is independent of:
- Original element labels
- Original subset ordering
- Duplicate subsets in $\mathcal{S}$

**Theorem (Uniqueness):** Two exact cover instances have the same canonical matrix if and only if they are **isomorphic** (differ only in element relabeling).

### Binary String Encoding

The canonical matrix is encoded as a binary string:

$$\text{enc}(A^*) = \text{un}(n) \; ; \; \text{un}(m^*) \; ; \; \text{bits}(A^*)$$

where:
- $\text{un}(k) = \underbrace{1 \cdots 1}_k \, 0$ is the unary encoding of non-negative integer $k$ (for all $k \geq 0$)
  - $\text{un}(0) = 0$
  - $\text{un}(1) = 10$
  - $\text{un}(2) = 110$
  - $\text{un}(3) = 1110$, etc.
- $\text{bits}(A^*)$ is the concatenation of all matrix entries in **row-major order**
  - For $A^* = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, we have $\text{bits}(A^*) = 1001$

**Example:**
```python
# Instance: 2×2 identity matrix
A = [[1, 0], [0, 1]]
# After canonicalization (columns sorted): [[0, 1], [1, 0]]
n, m = 2, 2
# Encoding: un(2) ; un(2) ; bits
#         = "110" ; "110" ; "0110"
#         = "1101100110"
encoding = "1101100110"
```

### Global Total Order (Shortlex)

The **shortlex order** on $\{0,1\}^*$ orders binary strings by:
1. **Length first**: $|s_1| < |s_2| \Rightarrow s_1 \prec s_2$
2. **Lexicographically for equal length**: $0 < 1$

**Indexing Non-Empty Strings**:
Since canonical encodings are always non-empty (they include at least $\text{un}(n)$ and $\text{un}(m)$),
we enumerate non-empty binary strings in shortlex order:

| Index | Encoding | Description |
|-------|----------|-------------|
| 1 | "0" | First length-1 string |
| 2 | "1" | Second length-1 string |
| 3 | "00" | First length-2 string |
| 4 | "01" | ... |
| 5 | "10" | ... |
| 6 | "11" | Last length-2 string |
| 7 | "000" | First length-3 string |
| ... | ... | ... |

**Index Formula** (for non-empty string $s$ of length $\ell \geq 1$):

$$\text{index}(s) = (2^\ell - 1) + \text{int}(s, 2)$$

where $\text{int}(s, 2)$ interprets $s$ as a binary number.

**Index Bounds**: For strings of length $\ell$:
- Minimum (for "$0\ldots 0$"): $2^\ell - 1$
- Maximum (for "$1\ldots 1$"): $2^{\ell+1} - 2$
- Approximate range: $2^\ell \lesssim \text{index}(s) < 2^{\ell+1}$

### Global Order on Exact Cover Instances

Through canonical encoding, we obtain a **total order** on all finite exact cover instances:

$$I_1 \prec I_2 \prec I_3 \prec \cdots$$

where $I_k$ is the unique exact cover instance (up to isomorphism) whose canonical encoding
has shortlex index $k$.

**Properties:**
1. **Totality**: Every finite exact cover instance has a unique position in this order
2. **Computability**: Given an instance, we can compute its index (though values may be astronomically large)
3. **Enumeration**: We can theoretically enumerate all instances (though doubly exponential growth makes this infeasible in practice)

**Note**: We treat exact cover instances as pairs $(U, \mathcal{S})$ where $\mathcal{S}$ is a **set** of subsets
(no multiplicity). Duplicate subsets are removed during canonicalization.

### Using Canonical Encoding

#### Computing Canonical Encoding

```python
from sudoku_nisq import ExactCoverProblem

# Define problem
universe = [2, 0, 1]  # Non-canonical order
subsets = {'S_0': [0, 1], 'S_1': [1, 2], 'S_2': [0, 1]}  # Has duplicate
problem = ExactCoverProblem(universe, subsets)

# Get canonical representation
canonical_matrix, ordered_universe = problem.to_canonical_matrix()
print(f"Ordered universe: {ordered_universe}")  # [0, 1, 2]
print(f"Canonical matrix (deduplicated, sorted):")
for row in canonical_matrix:
    print(f"  {row}")

# Get binary encoding
encoding = problem.to_canonical_encoding()
print(f"Binary encoding: {encoding}")
print(f"Encoding length: {len(encoding)}")

# Get global index
index = problem.canonical_order_index()
print(f"Shortlex index: {index}")
```

#### Comparing Instances

```python
# Two representations of the same problem
p1 = ExactCoverProblem([0, 1, 2], {'S_0': [0, 1], 'S_1': [1, 2]})
p2 = ExactCoverProblem([2, 0, 1], {'S_a': [1, 0], 'S_b': [2, 1]})

# Same canonical encoding
assert p1.to_canonical_encoding() == p2.to_canonical_encoding()

# Same global index
assert p1.canonical_order_index() == p2.canonical_order_index()
```

#### Sudoku Canonical Encoding

For Sudoku puzzles, you can compute canonical encodings for each encoding type:

```python
from sudoku_nisq.sudoku_puzzle import SudokuPuzzle
from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding

# Create Sudoku puzzle
puzzle = SudokuPuzzle.from_board([[1, 0], [0, 0]])

# Get exact cover encoding
encoder = ExactCoverEncoding(puzzle)

# Compute canonical encodings
enc_simple = encoder.to_canonical_encoding('simple')
enc_pattern = encoder.to_canonical_encoding('pattern')

print(f"Simple encoding length: {len(enc_simple)}")
print(f"Pattern encoding length: {len(enc_pattern)}")

# Different encodings (different exact cover instances)
assert enc_simple != enc_pattern
```

**Note:** The same Sudoku puzzle produces **two different** canonical encodings (simple vs pattern) because they represent different exact cover problem instances with different constraint sets.

### Complexity Analysis

The canonical encoding framework enables precise complexity measurement:

**Encoding Length:**
- Unary part: $n + 1 + m^* + 1$ bits
- Matrix part: $n \cdot m^*$ bits
- **Total**: $n + m^* + n \cdot m^* + 2$ bits
  - Linear in $n$ (universe size) and $m^*$ (unique subset count)
  - Dominated by matrix size $n \cdot m^*$ (quadratic when $n \approx m^*$)

**Index Magnitude:**
For encoding of length $\ell$, the shortlex index satisfies:
$$2^\ell - 1 \leq \text{index} < 2^{\ell+1} - 1$$

More simply: $2^\ell \lesssim \text{index} < 2^{\ell+1}$

**Growth Rate:**
- Small problems (e.g., 2×2 matrix, $\ell \approx 10$): index $\sim 10^3$
- Medium problems (e.g., 10×10 matrix, $\ell \approx 122$): index $\sim 10^{37}$
- Large problems (e.g., 100×100 matrix, $\ell \approx 10202$): index $\sim 10^{3071}$

The doubly exponential growth makes exhaustive enumeration infeasible beyond tiny instances.

### Mathematical Properties

**Theorem (Isomorphism Invariance):**  
$I_1 \cong I_2 \iff \text{enc}(I_1) = \text{enc}(I_2)$

**Corollary (Unique Representative):**  
Each isomorphism class of exact cover instances has exactly one canonical encoding.

**Theorem (Order Completeness):**  
The induced order $\prec$ on exact cover instances is:
- **Total**: Any two distinct isomorphism classes are comparable
- **Well-founded**: Every non-empty set has a minimal element
- **Computable**: Given instances $I_1, I_2$, we can determine $I_1 \prec I_2$ by comparing encodings

## See Also

- [Examples](examples.md) - Code examples and notebooks
- [Getting Started](getting-started.md) - Installation and basic usage
