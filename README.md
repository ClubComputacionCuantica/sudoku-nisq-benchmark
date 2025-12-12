# Sudoku NISQ Evaluation Framework

[![Docs Status](https://img.shields.io/badge/docs-dev-green.svg)](https://ClubComputacionCuantica.github.io/sudoku-nisq-benchmark/)


## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Basic Usage](#basic-usage)
- [Installation](#installation)
- [Benchmarking Objectives](#benchmarking-objectives)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [References](#references)

---
**Work in progress**


## Introduction

Why Sudoku?

Sudoku is a natural example of a **constraint satisfaction problem (CSP)**, and **generalized Sudoku** is known to be **NP-complete**. It offers:

- Familiarity to a general audience
- Clean problem encoding with known structure
- Natural translation into **exact cover** and other CSP formulations

This makes Sudoku accessible to study current limitations of quantum devices.

---

## Key Features

- **Exact-cover Sudoku to circuits**: Encodes Sudoku (and generic exact cover problems) into Grover-style quantum circuits with simple or pattern encodings.
- **Multi-SDK circuit builds**: Construct circuits with PyTKET, Qiskit, or Braket; caching keeps builds reproducible and fast.
- **Resource introspection**: Automatic gate counting, qubit/depth metadata, and optional memory tracking during circuit construction.
- **Backend plumbing**: IBM backends are tested; Aer simulation is first-class. Hooks for Quantinuum and Braket exist but are still maturing.
- **Benchmarking scaffolding**: Metrics/data models for success probability, gate/volume efficiency, and hardware/transpile metadata collection.

> Note: Non-IBM hardware paths are experimental; expect partial coverage or `NotImplementedError` while native flows are finalized.

---

## Basic Usage

### Sudoku Workflow

#### 1. Create a Sudoku Puzzle

```python
# Generate a 4x4 Sudoku puzzle with 2 missing cells
from sudoku_nisq import QSudoku
sudoku = QSudoku.generate(size=4, num_missing_cells=2)  # or size=2 for 2x2 (no real subgrids)

# Visualize the puzzle
sudoku.plot_puzzle()
```

#### 2. Set a Quantum Solver

```python
# Import the solver class
from sudoku_nisq import ExactCoverQuantumSolver

# Configure the solver with an encoding strategy
sudoku.set_solver(ExactCoverQuantumSolver, encoding="simple") # or "pattern"
```

#### 3. Build the Quantum Circuit

```python
# Build the quantum circuit
circuit = sudoku.build_circuit()

# Visualize the circuit
sudoku.draw_circuit()
```

#### 4. Run on Quantum Hardware or Simulator

```python
# Run on Aer simulator
result = sudoku.run_aer(shots=1024)

# Or, connect to IBM hardware
ibm_alias = sudoku.init_ibm(api_token="your_token", instance="your_instance", 
                          device="ibm_brisbane")
result = sudoku.run(ibm_alias, opt_level=1, shots=1000)
```

#### 5. Analyze Results

```python
# Visualize measurement results
sudoku.counts_plot(result, backend_alias="Aer", shots=1024)

# Show only valid solutions
sudoku.counts_plot(result, show_valid_only=True)

# Get resource utilization summary
resources = sudoku.report_resources()
```

### Generic Exact Cover Workflow

For problems smaller than minimal Sudoku or general exact cover instances:

```python
from sudoku_nisq import ExactCoverProblem, QExactCover

# 1. Define an exact cover problem
universe = [0, 1, 2, 3]
subsets = {
    'S_0': [0, 3],
    'S_1': [1, 2],
    'S_2': [0, 1, 2]
}
problem = ExactCoverProblem(universe, subsets, num_solutions=1)

# 2. Create quantum solver interface
qec = QExactCover(problem)

# 3. Build and run circuit
circuit = qec.build_circuit()
result = qec.run_aer(shots=1024)

# 4. Analyze resources
resources = qec.report_resources()
print(f"Requires {resources['estimated']['n_qubits']} qubits")

# 5. (Optional) Compute canonical encoding
encoding = problem.to_canonical_encoding()
index = problem.canonical_order_index()
print(f"Canonical encoding: {encoding[:50]}... (length {len(encoding)})")
print(f"Global shortlex index: {index}")
```

See `examples/exact_cover_benchmark.py` for a complete comparison and `examples/canonical_encoding_demo.py` for canonical encoding examples.

---

## Installation

### Prerequisites

- Python 3.10+
- Poetry (for dependency management and virtual environment setup)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/ClubComputacionCuantica/sudoku-nisq-benchmark.git
   cd sudoku-nisq-benchmark
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Run commands within the virtual environment:
   - To execute scripts or tools directly, use `poetry run`:
     ```bash
     poetry run python your_script.py
     ```

---

## Benchmarking Objectives

The framework aims to:

- Evaluate the performance of quantum algorithms for solving NP-complete problems.
- Benchmark quantum hardware capabilities using structured CSPs like Sudoku.
- Explore the impact of different encodings and optimization levels on circuit performance.

---

## Current Limitations

- Only the **ExactCoverQuantumSolver** is production-ready; other solver classes are placeholders.
- Real hardware remains challenged beyond toy (2x2) or small 4x4 instances.
- Circuit optimization and transpilation strategies are still evolving.
- IBM paths are validated; Quantinuum/Braket provider flows are in-progress and may be incomplete.

---

## Future Work

- Optimize circuit designs for better scalability.
- Extend support for additional quantum hardware providers.
- Develop new encoding strategies for improved efficiency.

---

## Maintainer

- Roberto Navarro A (Quantum Algorithms Group, ESFM - IPN)

---

## Citation

If using this benchmarking system in research, please cite:
- The sudoku-nisq-benchmark project
- Relevant quantum benchmarking standards papers
- Provider-specific calibration papers (if using hardware data)

---

## References

- Jiang & Wang, "Quantum Circuit Based on Grover’s Algorithm to Solve Exact Cover Problem," 2023, IEEE APWCS. https://doi.org/10.1109/APWCS60142.2023.10234054
