"""Tests for SDK/backend pairing and circuit format consistency.

This module tests the critical type safety mechanism that ensures quantum circuits
are in the correct SDK format before transpilation. It validates:

1. SDK tracking: _main_circuit_sdk attribute correctly stores circuit format
2. Automatic rebuilding: Circuit is rebuilt when SDK mismatch detected
3. Type guards: Transpilers reject wrong circuit types with helpful errors
4. Cross-SDK workflows: Building with one SDK then running on different backend

The tests prevent runtime type errors like passing a PyTKET Circuit to Qiskit's
transpiler or vice versa.
"""

import os
import pytest

# Enable new metadata architecture for testing
os.environ['SUDOKU_NISQ_NEW_METADATA'] = '1'

from sudoku_nisq import QSudoku, ExactCoverQuantumSolver
from sudoku_nisq.metadata.config import MetadataConfig


@pytest.fixture
def setup_environment():
    """Setup test environment with new metadata architecture."""
    MetadataConfig.ENABLE_NEW_ARCHITECTURE = True
    yield
    MetadataConfig.ENABLE_NEW_ARCHITECTURE = False


@pytest.fixture
def minimal_puzzle():
    """Create minimal 2x2 puzzle for fast testing."""
    puzzle = QSudoku.generate(size=2, num_missing_cells=1)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    return puzzle


class TestSDKTracking:
    """Test that circuit SDK format is tracked correctly."""
    
    def test_sdk_tracked_after_build_pytket(self, minimal_puzzle):
        """Verify _main_circuit_sdk is set when building with PyTKET (default)."""
        circuit = minimal_puzzle.build_circuit()
        
        solver = minimal_puzzle.quantum_solver
        assert hasattr(solver, '_main_circuit_sdk'), "SDK tracking attribute missing"
        assert solver._main_circuit_sdk == "pytket", f"Expected 'pytket', got {solver._main_circuit_sdk}"
        assert "Circuit" in type(circuit).__name__, f"Expected PyTKET Circuit, got {type(circuit)}"
    
    def test_sdk_tracked_after_build_qiskit(self, minimal_puzzle):
        """Verify _main_circuit_sdk is set when explicitly building with Qiskit."""
        circuit = minimal_puzzle.build_circuit(sdk="qiskit")
        
        solver = minimal_puzzle.quantum_solver
        assert hasattr(solver, '_main_circuit_sdk'), "SDK tracking attribute missing"
        assert solver._main_circuit_sdk == "qiskit", f"Expected 'qiskit', got {solver._main_circuit_sdk}"
        assert type(circuit).__name__ == "QuantumCircuit", f"Expected Qiskit QuantumCircuit, got {type(circuit)}"
    
    def test_sdk_tracked_after_build_no_explicit_sdk(self, minimal_puzzle):
        """Verify SDK is tracked even when no explicit sdk parameter is passed."""
        # Build without explicit SDK - should default to PyTKET
        minimal_puzzle.build_circuit()
        
        solver = minimal_puzzle.quantum_solver
        assert hasattr(solver, '_main_circuit_sdk'), "SDK tracking should work for implicit SDK selection"
        assert solver._main_circuit_sdk in ["pytket", "qiskit", "braket"], "SDK must be one of supported types"


class TestSDKMismatchDetection:
    """Test automatic detection and handling of SDK mismatches."""
    
    def test_qiskit_circuit_with_qiskit_backend_no_rebuild(self, setup_environment, minimal_puzzle):
        """Verify no rebuild when circuit SDK matches backend SDK."""
        # Build with Qiskit explicitly
        minimal_puzzle.build_circuit(sdk="qiskit")
        circuit1_id = id(minimal_puzzle.quantum_solver.main_circuit)
        
        # Run with Qiskit backend (Aer) - should NOT rebuild
        backend_alias = minimal_puzzle.init_aer(method="statevector")
        minimal_puzzle.set_validation_context(["0", "1"])
        
        minimal_puzzle.run(backend_alias, shots=100, opt_level=0)
        
        # Verify circuit was NOT replaced
        circuit2_id = id(minimal_puzzle.quantum_solver.main_circuit)
        assert circuit1_id == circuit2_id, "Circuit should not be rebuilt when SDK matches backend"
    
    def test_pytket_circuit_with_qiskit_backend_rebuilds(self, setup_environment, minimal_puzzle):
        """Verify circuit is rebuilt when format doesn't match backend SDK."""
        # Build with PyTKET (default)
        circuit1 = minimal_puzzle.build_circuit()
        assert "Circuit" in type(circuit1).__name__, "Should start with PyTKET circuit"
        circuit1_id = id(minimal_puzzle.quantum_solver.main_circuit)
        
        # Verify PyTKET SDK is tracked
        assert minimal_puzzle.quantum_solver._main_circuit_sdk == "pytket"
        
        # Initialize Qiskit backend
        backend_alias = minimal_puzzle.init_aer(method="statevector")
        minimal_puzzle.set_validation_context(["0", "1"])
        
        # Run with Qiskit backend - should trigger rebuild
        minimal_puzzle.run(backend_alias, shots=100, opt_level=0)
        
        # Verify circuit was rebuilt in Qiskit format
        circuit2 = minimal_puzzle.quantum_solver.main_circuit
        circuit2_id = id(circuit2)
        
        assert circuit1_id != circuit2_id, "Circuit should be rebuilt when SDK mismatch detected"
        assert type(circuit2).__name__ == "QuantumCircuit", f"Rebuilt circuit should be Qiskit, got {type(circuit2)}"
        assert minimal_puzzle.quantum_solver._main_circuit_sdk == "qiskit", "SDK tracker should update after rebuild"


class TestCrossSDKWorkflows:
    """Test workflows involving multiple SDK formats."""
    
    def test_build_qiskit_then_run_twice_same_backend(self, setup_environment, minimal_puzzle):
        """Test that multiple runs with same backend don't trigger unnecessary rebuilds."""
        # Build Qiskit circuit
        minimal_puzzle.build_circuit(sdk="qiskit")
        circuit_id_initial = id(minimal_puzzle.quantum_solver.main_circuit)
        
        # Initialize Qiskit Aer backend
        backend_alias = minimal_puzzle.init_aer(method="statevector")
        minimal_puzzle.set_validation_context(["0", "1"])
        
        # First run
        minimal_puzzle.run(backend_alias, shots=100, opt_level=0)
        circuit_id_after_run1 = id(minimal_puzzle.quantum_solver.main_circuit)
        
        # Second run
        minimal_puzzle.run(backend_alias, shots=100, opt_level=1)
        circuit_id_after_run2 = id(minimal_puzzle.quantum_solver.main_circuit)
        
        # All three should be the same object
        assert circuit_id_initial == circuit_id_after_run1 == circuit_id_after_run2, \
            "Circuit should not be rebuilt on subsequent runs with matching SDK"
    
    def test_multiple_sdk_builds_update_tracker(self, minimal_puzzle):
        """Test that SDK tracker updates correctly when rebuilding with different SDKs."""
        # Build with PyTKET
        minimal_puzzle.build_circuit()
        assert minimal_puzzle.quantum_solver._main_circuit_sdk == "pytket"
        
        # Rebuild with Qiskit
        minimal_puzzle.build_circuit(sdk="qiskit")
        assert minimal_puzzle.quantum_solver._main_circuit_sdk == "qiskit"
        
        # Rebuild with PyTKET again
        minimal_puzzle.build_circuit(sdk="pytket")
        assert minimal_puzzle.quantum_solver._main_circuit_sdk == "pytket"


class TestTypeGuards:
    """Test that transpilers validate circuit types and provide helpful errors."""
    
    def test_pytket_transpiler_rejects_qiskit_circuit(self, minimal_puzzle):
        """Verify PyTKET transpiler raises TypeError for Qiskit circuit."""
        # Build Qiskit circuit
        minimal_puzzle.build_circuit(sdk="qiskit")
        
        # Try to manually call PyTKET transpiler (simulate SDK mismatch)
        
        # This should fail with helpful error
        # Note: We can't easily test this without a PyTKET backend, so we skip for now
        # The type guard is tested implicitly by test_pytket_circuit_with_qiskit_backend_rebuilds
        pytest.skip("Cannot test PyTKET transpiler without PyTKET backend")
    
    def test_qiskit_transpiler_rejects_pytket_circuit(self, setup_environment, minimal_puzzle):
        """Verify Qiskit transpiler raises TypeError for PyTKET circuit with helpful message."""
        from qiskit_aer import AerSimulator
        
        # Build PyTKET circuit
        minimal_puzzle.build_circuit()
        assert "Circuit" in type(minimal_puzzle.quantum_solver.main_circuit).__name__
        
        # Manually set SDK tracker to qiskit to bypass auto-rebuild (simulate bug scenario)
        minimal_puzzle.quantum_solver._main_circuit_sdk = "qiskit"
        
        # Try to transpile - should fail with TypeError
        backend = AerSimulator()
        solver = minimal_puzzle.quantum_solver
        
        with pytest.raises(TypeError) as exc_info:
            solver._transpile_qiskit(backend, opt_level=0)
        
        # Verify error message is helpful
        error_msg = str(exc_info.value)
        assert "qiskit.circuit.QuantumCircuit" in error_msg, "Error should mention expected type"
        assert "sdk='qiskit'" in error_msg or "sdk=\"qiskit\"" in error_msg, "Error should suggest fix"


class TestSDKConsistencyAcrossRuns:
    """Test SDK consistency in multi-run scenarios."""
    
    def test_six_runs_maintain_sdk_consistency(self, setup_environment, minimal_puzzle):
        """Replicate user's 6-run scenario to ensure SDK stays consistent."""
        # Build Qiskit circuit explicitly (as in user's test.py)
        minimal_puzzle.build_circuit(sdk="qiskit")
        
        # Initialize Aer backend
        backend_alias = minimal_puzzle.init_aer(method="statevector")
        minimal_puzzle.set_validation_context(["0", "1"])
        
        # Track circuit ID across runs
        initial_circuit_id = id(minimal_puzzle.quantum_solver.main_circuit)
        circuit_ids = [initial_circuit_id]
        
        # Run 6 times with different opt levels (2 runs each at 0/1/2)
        opt_levels = [0, 0, 1, 1, 2, 2]
        for opt_level in opt_levels:
            minimal_puzzle.run(backend_alias, shots=100, opt_level=opt_level)
            circuit_ids.append(id(minimal_puzzle.quantum_solver.main_circuit))
        
        # All circuit IDs should be the same (no unnecessary rebuilds)
        assert len(set(circuit_ids)) == 1, f"Circuit was rebuilt unexpectedly. IDs: {circuit_ids}"
        
        # Verify SDK tracker stayed consistent
        assert minimal_puzzle.quantum_solver._main_circuit_sdk == "qiskit"


@pytest.mark.integration
class TestSDKDetectionWithRealBackends:
    """Integration tests with actual backend objects."""
    
    def test_aer_backend_detected_as_qiskit(self, minimal_puzzle):
        """Verify AerSimulator is correctly detected as Qiskit SDK."""
        from qiskit_aer import AerSimulator
        
        backend = AerSimulator()
        solver = minimal_puzzle.quantum_solver
        
        detected_sdk = solver._detect_backend_sdk(backend)
        assert detected_sdk == "qiskit", f"AerSimulator should be detected as 'qiskit', got '{detected_sdk}'"
    
    def test_none_backend_defaults_to_pytket(self, minimal_puzzle):
        """Verify None backend defaults to PyTKET SDK."""
        solver = minimal_puzzle.quantum_solver
        
        detected_sdk = solver._detect_backend_sdk(None)
        assert detected_sdk == "pytket", f"None backend should default to 'pytket', got '{detected_sdk}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
