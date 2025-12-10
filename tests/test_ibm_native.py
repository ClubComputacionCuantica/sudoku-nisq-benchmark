"""Tests for native Qiskit runtime backend integration.

Tests the full stack of IBM backend usage without PyTKET wrappers:
- Provider authentication and device listing
- Backend SDK detection for qiskit_ibm_runtime
- Circuit transpilation with generate_preset_pass_manager
- Native execution with backend.run()
- Error mitigation executor compatibility
"""

import pytest
from unittest.mock import Mock, patch
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2


class TestIBMProviderNative:
    """Test native Qiskit runtime service integration in IBMProvider."""
    
    def test_provider_returns_native_backend(self):
        """Test that IBMProvider returns native QiskitRuntimeService backend."""
        from sudoku_nisq.providers.ibm import IBMProvider
        
        # Mock QiskitRuntimeService
        with patch('sudoku_nisq.providers.ibm.QiskitRuntimeService') as mock_service_class:
            mock_service = Mock()
            mock_backend = Mock(spec=BackendV2)
            mock_backend.name = 'ibm_brisbane'
            mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
            
            mock_service.backend.return_value = mock_backend
            # backends() should return a list for authentication
            mock_service.backends.return_value = [mock_backend]
            mock_service_class.return_value = mock_service
            
            # Save account should work
            mock_service_class.save_account = Mock()
            
            provider = IBMProvider()
            provider.authenticate(api_token='fake_token', instance='test/instance')
            
            # Add device should return native backend
            backend = provider.add_device('ibm_brisbane', alias='brisbane')
            
            assert backend is mock_backend
            assert 'qiskit_ibm_runtime' in backend.__module__
    
    def test_provider_authentication_flow(self):
        """Test authentication saves account and initializes service."""
        from sudoku_nisq.providers.ibm import IBMProvider
        
        with patch('sudoku_nisq.providers.ibm.QiskitRuntimeService') as mock_service_class:
            mock_service = Mock()
            backend1 = Mock()
            backend1.name = 'ibm_brisbane'
            backend2 = Mock()
            backend2.name = 'ibm_kyoto'
            mock_service.backends.return_value = [backend1, backend2]
            mock_service_class.return_value = mock_service
            mock_service_class.save_account = Mock()
            
            provider = IBMProvider()
            devices = provider.authenticate(
                api_token='test_token',
                instance='test/instance',
                overwrite=False
            )
            
            # Should call save_account with correct parameters
            mock_service_class.save_account.assert_called_once_with(
                channel='ibm_quantum',
                token='test_token',
                instance='test/instance',
                overwrite=True
            )
            
            # Should list devices
            assert len(devices) == 2
            assert 'ibm_brisbane' in devices
            assert 'ibm_kyoto' in devices
    
    def test_provider_list_devices_without_auth_fails(self):
        """Test that listing devices without auth raises error."""
        from sudoku_nisq.providers.ibm import IBMProvider
        
        provider = IBMProvider()
        
        with pytest.raises(RuntimeError, match="Call authenticate"):
            provider.list_available_devices()


class TestBackendSDKDetection:
    """Test enhanced SDK detection for native Qiskit backends."""
    
    def test_detect_qiskit_runtime_backend(self):
        """Test detection of native qiskit_ibm_runtime backends."""
        from sudoku_nisq import QSudoku
        from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
        
        # Create mock runtime backend
        mock_backend = Mock(spec=BackendV2)
        mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
        mock_backend.name = 'ibm_brisbane'
        
        puzzle = QSudoku.generate(size=2, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
        solver = puzzle._solver
        
        sdk_type = solver._detect_backend_sdk(mock_backend)
        
        assert sdk_type == "qiskit"
    
    def test_detect_pytket_ibmq_wrapper(self):
        """Test detection still identifies PyTKET IBMQBackend wrapper."""
        from sudoku_nisq import QSudoku
        from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
        
        # Mock PyTKET IBMQBackend
        mock_backend = Mock()
        mock_backend.__module__ = 'pytket.extensions.qiskit.backends'
        mock_backend.get_compiled_circuit = Mock()
        mock_backend.process_circuit = Mock()
        
        puzzle = QSudoku.generate(size=2, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
        solver = puzzle._solver
        
        sdk_type = solver._detect_backend_sdk(mock_backend)
        
        assert sdk_type == "pytket"
    
    def test_detect_qiskit_aer_simulator(self):
        """Test detection of Qiskit AerSimulator."""
        from qiskit_aer import AerSimulator
        from sudoku_nisq import QSudoku
        from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
        
        backend = AerSimulator()
        
        puzzle = QSudoku.generate(size=2, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
        solver = puzzle._solver
        
        sdk_type = solver._detect_backend_sdk(backend)
        
        assert sdk_type == "qiskit"


class TestNativeQiskitTranspilation:
    """Test transpilation with generate_preset_pass_manager."""
    
    def test_transpile_with_target_backend(self):
        """Test transpilation uses generate_preset_pass_manager for Target backends."""
        from sudoku_nisq import QSudoku
        from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
        from qiskit.transpiler import Target
        
        # Create mock backend with Target
        mock_backend = Mock(spec=BackendV2)
        mock_backend.target = Mock(spec=Target)
        mock_backend.num_qubits = 127
        
        puzzle = QSudoku.generate(size=2, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
        solver = puzzle._solver
        
        # Build Qiskit circuit
        solver.build_main_circuit(sdk='qiskit')
        
        with patch('qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager') as mock_pm:
            mock_pass_manager = Mock()
            mock_pass_manager.run.return_value = QuantumCircuit(2, 2)
            mock_pm.return_value = mock_pass_manager
            
            result = solver._transpile_qiskit(mock_backend, opt_level=2)
            
            # Should use generate_preset_pass_manager
            mock_pm.assert_called_once_with(
                optimization_level=2,
                backend=mock_backend
            )
            assert isinstance(result, QuantumCircuit)
    
    def test_transpile_fallback_without_target(self):
        """Test transpilation falls back to compiler.transpile without Target."""
        from sudoku_nisq import QSudoku
        from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
        
        # Mock backend without Target
        mock_backend = Mock()
        mock_backend.target = None
        
        puzzle = QSudoku.generate(size=2, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
        solver = puzzle._solver
        
        solver.build_main_circuit(sdk='qiskit')
        
        with patch('qiskit.compiler.transpile') as mock_transpile:
            mock_transpile.return_value = QuantumCircuit(2, 2)
            
            result = solver._transpile_qiskit(mock_backend, opt_level=1)
            
            # Should fall back to compiler.transpile
            mock_transpile.assert_called_once()
            assert isinstance(result, QuantumCircuit)


class TestNativeQiskitExecution:
    """Test execution path for native Qiskit backends."""
    
    def test_run_qiskit_native_uses_backend_run(self):
        """Test that _run_qiskit_native uses backend.run() interface."""
        from sudoku_nisq import QSudoku
        from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
        
        # Mock runtime backend
        mock_backend = Mock(spec=BackendV2)
        mock_job = Mock()
        mock_result = Mock()
        mock_result.get_counts.return_value = {'00': 512, '11': 512}
        mock_job.result.return_value = mock_result
        mock_backend.run.return_value = mock_job
        
        puzzle = QSudoku.generate(size=2, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
        solver = puzzle._solver
        
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = solver._run_qiskit_native(mock_backend, circuit, shots=1024)
        
        # Should call backend.run()
        mock_backend.run.assert_called_once_with(circuit, shots=1024)
        assert result is mock_result
    
    def test_run_method_routes_to_qiskit_native(self):
        """Test that run() routes to _run_qiskit_native for Qiskit backends."""
        from sudoku_nisq import QSudoku
        from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
        
        # Mock runtime backend
        mock_backend = Mock(spec=BackendV2)
        mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
        mock_backend.target = Mock()
        
        mock_job = Mock()
        mock_result = Mock()
        mock_result.get_counts.return_value = {'0000': 1024}
        mock_job.result.return_value = mock_result
        mock_backend.run.return_value = mock_job
        
        puzzle = QSudoku.generate(size=2, num_missing_cells=2)
        puzzle.set_solver(ExactCoverQuantumSolver, encoding='simple')
        solver = puzzle._solver
        solver.store_transpiled = False
        
        # Build and transpile
        solver.build_main_circuit(sdk='qiskit')
        
        with patch.object(solver, '_transpile_qiskit') as mock_transpile:
            circuit = QuantumCircuit(2, 2)
            circuit.measure_all()
            mock_transpile.return_value = circuit
            
            result = solver.run(
                backend=mock_backend,
                backend_alias='ibm_test',
                shots=1024,
                optimisation_level=1
            )
            
            # Should use native execution
            mock_backend.run.assert_called_once()
            assert result is mock_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
