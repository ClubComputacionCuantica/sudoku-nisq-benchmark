# Sudoku NISQ Evaluation Framework

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Basic Usage](#basic-usage)
- [Installation](#installation)
- [Quantum Algorithms Implemented](#quantum-algorithms-implemented)
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

- **Multiple Solver Algorithms and Encodings**: Includes Exact Cover and Graph Coloring solvers with flexible encoding strategies.
- **Hardware Integration**: Currently supports IBM Quantum backends, with plans to integrate Quantinuum and other providers in the future.
- **Circuit Caching and Memory Management**: Intelligent caching to avoid redundant computation.
- **Automated Benchmarking**: ExperimentRunner for systematic evaluation across solvers, encodings, backends, and optimization levels.

---

## Basic Usage

### 1. Create a Sudoku Puzzle

```python
# Generate a 4x4 Sudoku puzzle with 2 missing cells
from sudoku_nisq import QSudoku
sudoku = QSudoku.generate(subgrid_size=2, num_missing_cells=2)

# Visualize the puzzle
sudoku.plot_puzzle()
```

### 2. Set a Quantum Solver

```python
# Import the solver class
from sudoku_nisq.exact_cover_solver import ExactCoverQuantumSolver

# Configure the solver with an encoding strategy
sudoku.set_solver(ExactCoverQuantumSolver, encoding="simple") # or "pattern"
```

### 3. Build the Quantum Circuit

```python
# Build the quantum circuit
circuit = sudoku.build_circuit()

# Visualize the circuit
sudoku.draw_circuit()
```

### 4. Run on Quantum Hardware or Simulator

```python
# Run on Aer simulator
result = sudoku.run_aer(shots=1024)

# Or, connect to IBM hardware
ibm_alias = sudoku.init_ibm(api_token="your_token", instance="your_instance", 
                          device="ibm_brisbane")
result = sudoku.run(ibm_alias, opt_level=1, shots=1000)
```

### 5. Analyze Results

```python
# Visualize measurement results
sudoku.counts_plot(result, backend_alias="Aer", shots=1024)

# Show only valid solutions
sudoku.counts_plot(result, show_valid_only=True)

# Get resource utilization summary
resources = sudoku.report_resources()
```

---

## Installation

### Prerequisites
- Python 3.10+
- Poetry (for dependency management and virtual environment setup)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/RobBEN93/sudoku-nisq-benchmark.git
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

## Quantum Algorithms Implemented

- **Grover-based Sudoku Solver**

---

## Benchmarking Objectives

The framework aims to:

- Evaluate the performance of quantum algorithms for solving NP-complete problems.
- Benchmark quantum hardware capabilities using structured CSPs like Sudoku.
- Explore the impact of different encodings and optimization levels on circuit performance.

---

## Current Limitations

- Only the **ExactCoverQuantumSolver** is fully implemented; other solvers are placeholders.
- Real hardware cannot yet reliably solve even small 4x4 puzzles.
- Circuits are not yet fully optimized.
- Only IBM Quantum backends are currently supported; Quantinuum and others are not yet functional.

---

## Future Work

- Optimize circuit designs for better scalability.
- Extend support for additional quantum hardware providers.
- Develop new encoding strategies for improved efficiency.

---

## References

- Jiang & Wang, "Quantum Circuit Based on Grover’s Algorithm to Solve Exact Cover Problem," 2023, IEEE APWCS. https://doi.org/10.1109/APWCS60142.2023.10234054
