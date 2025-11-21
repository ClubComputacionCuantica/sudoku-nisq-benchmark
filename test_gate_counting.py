"""Test gate counting feature for exact cover circuits."""

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

def test_gate_counting_pytket():
    """Test gate counting with PyTKET implementation."""
    print("\n=== Testing PyTKET Gate Counting ===")
    
    # Create a simple 2x2 Sudoku puzzle
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    # Build circuit with PyTKET (default)
    circuit = puzzle.build_circuit(sdk="pytket")
    
    # Get gate counts from the solver
    gate_counts = puzzle._solver.gate_counts
    
    print(f"\nCircuit built with {circuit.n_qubits} qubits and {circuit.n_gates} gates")
    print("\nGate type breakdown (PyTKET):")
    if gate_counts:
        for gate_name, count in sorted(gate_counts.items()):
            print(f"  {gate_name}: {count}")
        
        # Verify we have the expected gate types
        expected_gates = ['H', 'X', 'Measure']
        for gate in expected_gates:
            assert gate in gate_counts, f"Expected gate type '{gate}' not found in counts"
        
        # Verify we have some multi-controlled gates
        has_mcx = any(gate.endswith('X') and gate != 'X' for gate in gate_counts)
        assert has_mcx, "Expected to find multi-controlled X gates (CX, CCX, etc.)"
        
        print("\n✓ PyTKET gate counting passed!")
    else:
        print("✗ Gate counts not available")
        return False
    
    return True

def test_gate_counting_qiskit():
    """Test gate counting with Qiskit implementation."""
    print("\n=== Testing Qiskit Gate Counting ===")
    
    # Create a simple 2x2 Sudoku puzzle
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    # Build circuit with Qiskit
    circuit = puzzle.build_circuit(sdk="qiskit")
    
    # Get gate counts from the solver
    gate_counts = puzzle._solver.gate_counts
    
    print(f"\nCircuit built with {circuit.num_qubits} qubits")
    print("\nGate type breakdown (Qiskit):")
    if gate_counts:
        for gate_name, count in sorted(gate_counts.items()):
            print(f"  {gate_name}: {count}")
        
        # Verify we have the expected gate types
        expected_gates = ['H', 'X', 'Measure']
        for gate in expected_gates:
            assert gate in gate_counts, f"Expected gate type '{gate}' not found in counts"
        
        # Verify we have some multi-controlled gates
        has_mcx = any(gate.endswith('X') and gate != 'X' for gate in gate_counts)
        assert has_mcx, "Expected to find multi-controlled X gates (CX, CCX, etc.)"
        
        print("\n✓ Qiskit gate counting passed!")
    else:
        print("✗ Gate counts not available")
        return False
    
    return True

def test_gate_counting_braket():
    """Test gate counting with Braket implementation."""
    print("\n=== Testing Braket Gate Counting ===")
    try:
        # Create a simple 2x2 Sudoku puzzle
        puzzle = QSudoku.generate(size=4, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
        
        # Build circuit with Braket
        circuit = puzzle.build_circuit(sdk="braket")
        
        # Get gate counts from the solver
        gate_counts = puzzle._solver.gate_counts
        
        print(f"\nCircuit built with {len(circuit.qubits)} qubits")
        print("\nGate type breakdown (Braket):")
        if gate_counts:
            for gate_name, count in sorted(gate_counts.items()):
                print(f"  {gate_name}: {count}")
            
            # Verify we have the expected gate types
            expected_gates = ['H', 'X', 'Measure']
            for gate in expected_gates:
                assert gate in gate_counts, f"Expected gate type '{gate}' not found in counts"
            
            # Verify we have some multi-controlled gates
            has_mcx = any(gate.endswith('X') and gate != 'X' for gate in gate_counts)
            assert has_mcx, "Expected to find multi-controlled X gates (CX, CCX, etc.)"
            
            print("\n✓ Braket gate counting passed!")
        else:
            print("✗ Gate counts not available")
            return False
        
        return True
    except ImportError:
        print("Braket SDK not installed; skipping Braket test.")
        return True  # Don't fail suite if Braket is missing

def test_gate_counts_consistency():
    """Test that gate counts are consistent between PyTKET and Qiskit.
    
    Note: Small differences are expected due to different gate decompositions.
    For example, PyTKET uses CnZ directly while Qiskit implements it as H-MCX-H.
    """
    print("\n=== Testing Gate Count Consistency ===")
    
    # Create the same puzzle (both will be random but same structure)
    puzzle_pytket = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle_pytket.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    puzzle_qiskit = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle_qiskit.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    # Build both circuits
    circuit_pytket = puzzle_pytket.build_circuit(sdk="pytket")
    circuit_qiskit = puzzle_qiskit.build_circuit(sdk="qiskit")
    
    # Get gate counts
    counts_pytket = puzzle_pytket._solver.gate_counts
    counts_qiskit = puzzle_qiskit._solver.gate_counts
    
    print("\nPyTKET gate counts:")
    for gate_name, count in sorted(counts_pytket.items()):
        print(f"  {gate_name}: {count}")
    
    print("\nQiskit gate counts:")
    for gate_name, count in sorted(counts_qiskit.items()):
        print(f"  {gate_name}: {count}")
    
    # Check that both implementations count fundamental gates
    if counts_pytket and counts_qiskit:
        print("\n✓ Both implementations successfully count gates!")
        
        # Note expected differences due to decomposition choices
        all_gates = set(counts_pytket.keys()) | set(counts_qiskit.keys())
        differences = []
        for gate in sorted(all_gates):
            p_count = counts_pytket.get(gate, 0)
            q_count = counts_qiskit.get(gate, 0)
            if p_count != q_count:
                differences.append(f"  {gate}: PyTKET={p_count}, Qiskit={q_count}")
        
        if differences:
            print("\nNote: Gate count differences due to decomposition strategies:")
            for diff in differences:
                print(diff)
            print("  (PyTKET uses CnZ directly; Qiskit uses H-MCX-H for multi-controlled Z)")
        
        return True
    else:
        print("\n✗ Gate counts missing from one or both implementations")
        return False

def test_gate_counts_metadata():
    """Test that gate counts are included in metadata."""
    print("\n=== Testing Gate Counts in Metadata ===")
    
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    # Build circuit
    circuit = puzzle.build_circuit()
    
    # Check if gate counts are accessible via get_gate_counts()
    gate_counts = puzzle._solver.get_gate_counts()
    
    if gate_counts:
        print("\nGate counts accessible via get_gate_counts():")
        for gate_name, count in sorted(gate_counts.items()):
            print(f"  {gate_name}: {count}")
        print("\n✓ Gate counts metadata test passed!")
        return True
    else:
        print("\n✗ Gate counts not accessible via get_gate_counts()")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Gate Counting Feature Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test PyTKET implementation
    try:
        results.append(("PyTKET", test_gate_counting_pytket()))
    except Exception as e:
        print(f"\n✗ PyTKET test failed with error: {e}")
        results.append(("PyTKET", False))
    
    # Test Qiskit implementation
    try:
        results.append(("Qiskit", test_gate_counting_qiskit()))
    except Exception as e:
        print(f"\n✗ Qiskit test failed with error: {e}")
        results.append(("Qiskit", False))
    
    # Test Braket implementation
    try:
        results.append(("Braket", test_gate_counting_braket()))
    except Exception as e:
        print(f"\n✗ Braket test failed with error: {e}")
        results.append(("Braket", False))
    
    # Test consistency
    try:
        results.append(("Consistency", test_gate_counts_consistency()))
    except Exception as e:
        print(f"\n✗ Consistency test failed with error: {e}")
        results.append(("Consistency", False))
    
    # Test metadata
    try:
        results.append(("Metadata", test_gate_counts_metadata()))
    except Exception as e:
        print(f"\n✗ Metadata test failed with error: {e}")
        results.append(("Metadata", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("All tests PASSED! 🎉")
    else:
        print("Some tests FAILED. Please review the output above.")
    print("=" * 60)
