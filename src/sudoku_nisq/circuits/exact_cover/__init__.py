"""Exact cover circuit implementations for different quantum SDKs.

This module contains provider-specific circuit implementations for the exact cover
Sudoku solver using Grover's algorithm. Each implementation is tailored to its
respective quantum SDK:

- **pytket_impl.py**: PyTKET implementation for Quantinuum and generic backends
- **qiskit_impl.py**: Qiskit implementation for IBM Quantum backends
- **braket_impl.py**: (Future) AWS Braket implementation

These implementations are NOT part of the public API. They are accessed internally
by ExactCoverQuantumSolver through SDK abstraction based on the provider:

- IBM backends → qiskit_impl.py
- Quantinuum backends → pytket_impl.py  
- AWS backends → braket_impl.py (when implemented)

Users should use the high-level solver API rather than importing these directly:

    from sudoku_nisq import QSudoku, ExactCoverQuantumSolver
    
    # Example 1: Automatic SDK selection (no backend)
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    circuit = puzzle.build_circuit()  # Uses pytket_impl (default)
    
    # Example 2: Explicit SDK selection (no backend needed)
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    qiskit_circuit = puzzle.build_circuit(sdk="qiskit")   # Force qiskit_impl
    pytket_circuit = puzzle.build_circuit(sdk="pytket")   # Force pytket_impl
    braket_circuit = puzzle.build_circuit(sdk="braket")   # Force braket_impl
    
    # Example 3: Provider-driven SDK selection (with backend)
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    puzzle.initialize_ibm(api_token, instance, "ibm_brisbane")
    circuit = puzzle.build_circuit()  # Uses qiskit_impl (IBM provider)
    
    # Example 4: Override provider's SDK for comparison
    puzzle.initialize_ibm(api_token, instance, "ibm_brisbane")
    qiskit_circ = puzzle.build_circuit(sdk="qiskit")   # Use IBM's default
    pytket_circ = puzzle.build_circuit(sdk="pytket")   # Compare with PyTKET

The solver automatically selects the appropriate implementation based on:
1. Explicit sdk parameter (highest priority)
2. Backend's provider SDK mapping
3. PyTKET default (no backend, no explicit SDK)

This ensures optimal compatibility while enabling SDK comparison and research workflows.

For implementation details, see:
- REFACTORING_SUMMARY.md: SDK abstraction architecture
- SDK_ABSTRACTION.md: Provider-to-SDK mapping design
"""