"""Comprehensive tests for explicit SDK selection feature."""

import pytest
from sudoku_nisq import QSudoku
from sudoku_nisq.solvers import ExactCoverQuantumSolver


class TestSDKSelection:
    """Test suite for explicit SDK selection in circuit building."""
    
    @pytest.fixture
    def puzzle(self):
        """Create a test puzzle with solver configured."""
        p = QSudoku.generate(size=4, num_missing_cells=2)
        p.set_solver(ExactCoverQuantumSolver, encoding="simple")
        return p
    
    def test_default_sdk_is_pytket(self, puzzle):
        """Test that default SDK is PyTKET when no backend and no explicit SDK."""
        circuit = puzzle.build_circuit()
        assert type(circuit).__name__ == "Circuit"
        assert "pytket" in type(circuit).__module__
    
    def test_explicit_pytket_selection(self, puzzle):
        """Test explicit PyTKET SDK selection."""
        circuit = puzzle.build_circuit(sdk="pytket")
        assert type(circuit).__name__ == "Circuit"
        assert "pytket" in type(circuit).__module__
    
    def test_explicit_qiskit_selection(self, puzzle):
        """Test explicit Qiskit SDK selection."""
        circuit = puzzle.build_circuit(sdk="qiskit")
        assert type(circuit).__name__ == "QuantumCircuit"
        assert "qiskit" in type(circuit).__module__
    
    def test_explicit_sdk_returns_native_format(self, puzzle):
        """Test that explicit SDK selection returns native circuit format, not PyTKET."""
        pytket_circ = puzzle.build_circuit(sdk="pytket")
        qiskit_circ = puzzle.build_circuit(sdk="qiskit")
        
        # They should be different types
        assert type(pytket_circ) is not type(qiskit_circ)
        
        # PyTKET should have specific attributes
        assert hasattr(pytket_circ, 'n_qubits')
        assert hasattr(pytket_circ, 'n_gates')
        
        # Qiskit should have specific attributes
        assert hasattr(qiskit_circ, 'num_qubits')
        assert hasattr(qiskit_circ, 'size')
    
    def test_invalid_sdk_raises_error(self, puzzle):
        """Test that invalid SDK name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SDK 'invalid'"):
            puzzle.build_circuit(sdk="invalid")
    
    def test_sdk_case_sensitive(self, puzzle):
        """Test that SDK parameter is case-sensitive."""
        with pytest.raises(ValueError, match="Invalid SDK 'QISKIT'"):
            puzzle.build_circuit(sdk="QISKIT")
    
    def test_explicit_sdk_rebuilds_circuit(self, puzzle):
        """Test that explicit SDK forces rebuild even if cache exists."""
        # Build with default (PyTKET) - this creates cache
        circuit1 = puzzle.build_circuit()
        
        # Build with explicit Qiskit - should rebuild, not load cache
        circuit2 = puzzle.build_circuit(sdk="qiskit")
        
        # Should be different types
        assert type(circuit1) is not type(circuit2)
        assert "pytket" in type(circuit1).__module__
        assert "qiskit" in type(circuit2).__module__
    
    def test_multiple_sdk_builds(self, puzzle):
        """Test building with multiple SDKs in sequence."""
        pytket_1 = puzzle.build_circuit(sdk="pytket")
        qiskit_1 = puzzle.build_circuit(sdk="qiskit")
        pytket_2 = puzzle.build_circuit(sdk="pytket")
        qiskit_2 = puzzle.build_circuit(sdk="qiskit")
        
        # Each should be correct type
        assert type(pytket_1).__name__ == "Circuit"
        assert type(qiskit_1).__name__ == "QuantumCircuit"
        assert type(pytket_2).__name__ == "Circuit"
        assert type(qiskit_2).__name__ == "QuantumCircuit"
    
    def test_sdk_parameter_is_optional(self, puzzle):
        """Test that sdk parameter is truly optional."""
        # Should work without sdk parameter
        circuit = puzzle.build_circuit()
        assert circuit is not None
    
    def test_circuit_equivalence_across_sdks(self, puzzle):
        """Test that circuits from different SDKs solve the same problem."""
        pytket_circ = puzzle.build_circuit(sdk="pytket")
        qiskit_circ = puzzle.build_circuit(sdk="qiskit")
        
        # Both should have the same logical qubit count (may differ in ancillas)
        # This is a basic sanity check
        assert pytket_circ.n_qubits >= puzzle._solver.s_size
        assert qiskit_circ.num_qubits >= puzzle._solver.s_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
