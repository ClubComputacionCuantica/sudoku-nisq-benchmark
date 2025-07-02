# Sudoku NISQ Benchmark Framework

## Table of Contents
- [Introduction](#introduction)
- [Goals](#goals)
- [Key Features](#key-features)
- [Why Sudoku?](#why-sudoku)
- [Basic Usage](#basic-usage)
- [Installation](#installation)
- [Quantum Algorithms Implemented](#quantum-algorithms-implemented)
- [Benchmarking Objectives](#benchmarking-objectives)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction
** AI generated draft
* Also mention that it offers an accessible view to real quantum computing applied to a familiar problem

This repository contains a designed to demonstrate how **Sudoku puzzles can serve as accessible benchmarking problems for near-term quantum devices** (NISQ era). It integrates **puzzle generation**, **quantum circuit construction using multiple algorithms**, **resource analysis**, and **hardware transpilation**.

Our framework offers a transparent, pedagogically meaningful, and technically rigorous approach to understanding the limitations and variability of quantum processors when applied to small but non-trivial constraint satisfaction problems (CSPs).

---

## Goals
- Provide a **publicly understandable benchmarking method** for NISQ hardware.
- Use well-defined, **NP-complete problems (generalized Sudoku)** as a testbed.
- Explore and compare **different quantum algorithms** on the same problem.
- Enable reproducible studies on **hardware resource constraints**, **transpilation effects**, and **solution fidelity**.

---

## Key Features
- 📄 Random or user-defined Sudoku puzzles $n^2 \times n^2$
- ⚛️ Three quantum algorithms: Grover-based exact cover and graph coloring approaches, and quantum backtracking
- 🛠️ Transpilation and execution on real hardware
- 📊 Data generation + analysis for comparing hardware and algorithm performance

---

## Why Sudoku?

Sudoku is a natural example of a **constraint satisfaction problem (CSP)**, and **generalized Sudoku** is known to be **NP-complete**. It offers:

- Familiarity to a general audience
- Clean problem encoding with known structure
- Natural translation into **exact cover** and other CSP formulations

This makes Sudoku both accessible and rigorous enough to expose current limitations of quantum devices.

---

## Basic Usage

Refer to the example Jupyter notebooks for generating puzzles, constructing quantum circuits, benchmarking, and visualizing results.

```python
from sudoku_nisq.q_sudoku import Sudoku

sudoku = Sudoku(subgrid_size=2, missing_cells=5)
sudoku.plot()
sudoku.quantum.get_circuit()
sudoku.quantum.find_resources()
```

For benchmarking on IBM backends:

```python
from pytket.extensions.qiskit import IBMQBackend
backend = IBMQBackend("ibm_oslo")
circuit = sudoku.quantum.get_circuit()
compiled = backend.get_compiled_circuit(circuit)
result = backend.get_result(backend.process_circuit(compiled, n_shots=100))
```

---

## Installation

### Prerequisites
- Python 3.10+
- Poetry

### Steps

---

## Quantum Algorithms Implemented

- **Grover-based Sudoku Solvers**: Transforms Sudoku to an exact cover problem or a graph coloring problem, then applies Grover search.
- **Quantum Backtracking**: Uses A. Montanaro's quantum backtracking algorithm.

Each method has different trade-offs in circuit depth and hardware compatibility.

---

## Benchmarking Objectives

We use this framework to study:
- Logical vs transpiled resource counts (qubits, CX gates, depth)
- Performance degradation across hardware backends
- Algorithm scalability with puzzle difficulty

Metrics collected include:
- Gate counts
- Qubit usage
- Execution time
- Success probability (solution state detection)

---

## Current Limitations

- Real hardware cannot yet reliably solve even small 4x4 puzzles without heavy error.
- Circuits are not yet fully optimized.
- Hardware variability makes reproducibility difficult across platforms.
- Limited testing beyond IBMQ.

---

## Future Work

- Benchmark across multiple QPUs
- Explore quantum counting to estimate number of solutions
- Include error mitigation techniques (zero-noise extrapolation, purification)
- Create an interactive web interface or dashboard

---

## References

- Jiang & Wang, "Quantum Circuit Based on Grover’s Algorithm to Solve Exact Cover Problem," 2023, IEEE APWCS. https://doi.org/10.1109/APWCS60142.2023.10234054
- Farhi et al., "A Quantum Approximate Optimization Algorithm," 2014. arXiv:1411.4028
- D-Wave Quantum Sudoku: https://www.dwavesys.com/media/
- Quantum CSP methods overview: Montanaro, 2015. "Quantum walk speedup of backtracking algorithms". arXiv:1509.02374

---

**Status:** Work in progress 🚧