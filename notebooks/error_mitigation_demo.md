# Zero Noise Extrapolation (ZNE) and Probabilistic Error Cancellation (PEC) Integration

This notebook demonstrates how to use error mitigation techniques (ZNE and PEC) with the exact cover quantum solver for Sudoku.

## Overview

Traditional quantum error mitigation techniques like ZNE and PEC are designed for **expectation values**, but our Grover-based exact cover solver produces **bitstrings**. We've bridged this gap by wrapping bitstring measurements into scalar expectation values:

$$\langle f \rangle = \Pr[\text{valid exact cover solution}]$$

This expectation value represents the success probability of the quantum algorithm, which can be improved using error mitigation.

## Setup

```python
from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.backends import BackendManager

# For demonstration purposes, we'll use the Aer simulator
# In practice, you'd use a noisy backend for mitigation to be effective
```

## Example 1: Basic Usage Without Mitigation

```python
# Generate a simple 2x2 Sudoku puzzle
qs = QSudoku.generate(size=2, num_missing=2, seed=42)
print(f"Puzzle:\\n{qs.puzzle}")

# Set the exact cover solver
qs.set_solver(ExactCoverQuantumSolver, encoding='simple')

# Run without mitigation (standard execution)
result = qs.run_aer(shots=1024)
counts = result.get_counts()

print(f"\\nMeasurement counts: {counts}")
```

## Example 2: Using Zero Noise Extrapolation (ZNE)

```python
# To use ZNE, we need to enable it in the run method
# Note: ZNE is most effective on noisy hardware; on simulators it may not show improvement

# Get the backend
backend_manager = BackendManager()
backend = backend_manager.get_backend('aer')

# Run with ZNE enabled
# The solver automatically computes success probability as an expectation value
result_zne = qs.solver.run(
    backend=backend,
    backend_alias='aer',
    shots=2048,
    use_zne=True,
    # Optional: customize ZNE parameters
    # zne_scale_noise=...,  # Custom noise scaling function
    # zne_factory=...,      # Custom extrapolation factory
)

# Access the mitigated success probability
if hasattr(result_zne, '_mitigated_success_prob'):
    mitigated_prob = result_zne._mitigated_success_prob
    print(f"Mitigated success probability: {mitigated_prob:.4f}")

# Standard measurement counts are still available
counts_zne = result_zne.get_counts()
print(f"Measurement counts: {counts_zne}")
```

## Example 3: Comparing Raw vs Mitigated Results

```python
from sudoku_nisq.mitigation.expectation_wrapper import compute_success_expectation

# Run standard execution
result_raw = qs.solver.run(
    backend=backend,
    backend_alias='aer',
    shots=2048,
    use_zne=False
)

# Compute raw success probability
counts_raw = result_raw.get_counts()
raw_success_prob = compute_success_expectation(
    counts_raw, 
    qs.solver._is_valid_solution
)

# Run with ZNE
result_zne = qs.solver.run(
    backend=backend,
    backend_alias='aer',
    shots=2048,
    use_zne=True
)

# Get mitigated probability
mitigated_prob = result_zne._mitigated_success_prob

print(f"Raw success probability: {raw_success_prob:.4f}")
print(f"Mitigated success probability: {mitigated_prob:.4f}")
print(f"Improvement: {(mitigated_prob - raw_success_prob):.4f}")
```

## Example 4: Solution Validation

```python
# The solver's _is_valid_solution method checks if a bitstring is correct
def check_solutions(counts, solver):
    \"\"\"Analyze measurement results for valid solutions.\"\"\"
    total = sum(counts.values())
    valid_count = 0
    
    print("\\nBitstring Analysis:")
    for bitstring, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        is_valid = solver._is_valid_solution(bitstring)
        prob = count / total
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"  {bitstring}: {count:4d} shots ({prob:6.2%}) {status}")
        if is_valid:
            valid_count += count
    
    success_rate = valid_count / total
    print(f"\\nOverall success rate: {success_rate:.2%}")
    return success_rate

# Check raw results
print("=== Raw Results ===")
raw_rate = check_solutions(counts_raw, qs.solver)

# ZNE provides the mitigated success probability directly
print(f"\\n=== Mitigated Success Probability (ZNE) ===")
print(f"Mitigated: {mitigated_prob:.2%}")
```

## How It Works

### 1. Bitstring → Expectation Value

The key insight is defining an indicator function:

$$f(b) = \\begin{cases} 1 & \\text{if bitstring } b \\text{ is valid} \\\\ 0 & \\text{otherwise} \\end{cases}$$

The expectation value is then:

$$\\langle f \\rangle = \\sum_b P(b) \\cdot f(b) = \\Pr[\\text{valid solution}]$$

### 2. Error Mitigation

**Zero Noise Extrapolation (ZNE):**
- Runs the circuit at multiple noise levels (via gate folding)
- Extrapolates to zero noise
- Default: fold factors [1, 3, 5] with Richardson extrapolation

**Probabilistic Error Cancellation (PEC):**
- Decomposes gates into noisy basis operations
- Samples circuits with quasiprobabilities
- Requires noise model characterization
- More expensive but can be more accurate

### 3. Validation Logic

The `ExactCoverQuantumSolver._is_valid_solution` method checks:
1. Selected subsets cover all universe elements
2. Each element is covered exactly once (no duplicates)

```python
# Example validation
def _is_valid_solution(self, bitstring: str) -> bool:
    selected_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
    covered_elements = []
    for idx in selected_indices:
        subset_key = f'S_{idx}'
        if subset_key in self.subsets:
            covered_elements.extend(self.subsets[subset_key])
    
    return (len(covered_elements) == len(set(covered_elements)) and 
            set(covered_elements) == set(self.universe))
```

## Configuration Options (TODOs)

The current implementation uses default configurations. Future enhancements:

### ZNE Configuration
- **Noise scaling**: Customize fold factors and scaling strategy
- **Extrapolation factory**: Choose between Linear, Richardson, exponential fits
- **Backend-specific tuning**: Optimize for specific noise characteristics

### PEC Configuration  
- **Representation generation**: Auto-generate from backend calibration
- **Gate decomposition**: Custom noisy gate models
- **Sampling strategy**: Optimize circuit sampling for efficiency

### Combined Mitigation
- Support applying ZNE and PEC sequentially
- Hybrid strategies for different circuit regions

## References

- Mitiq Documentation: https://mitiq.readthedocs.io/
- ZNE Paper: https://arxiv.org/abs/1612.02058
- PEC Paper: https://arxiv.org/abs/1612.02058
- Exact Cover Algorithm: J. -R. Jiang and Y. -J. Wang, "Quantum Circuit Based on Grover's Algorithm to Solve Exact Cover Problem," 2023

## Notes

- Error mitigation is most effective on **noisy hardware**
- Simulators like Aer (without noise models) won't show significant improvement
- ZNE requires ~3-5x more circuit executions (one per fold factor)
- PEC can be very expensive (many sampled circuits)
- Always validate results using the success probability metric
