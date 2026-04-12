"""Tests for Qiskit Runtime primitive support (SamplerV2)."""

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")

pytestmark = pytest.mark.integration

from unittest.mock import Mock, MagicMock, patch  # noqa: E402
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister  # noqa: E402

from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver  # noqa: E402
from sudoku_nisq.sudoku_puzzle import SudokuPuzzle  # noqa: E402


def create_test_puzzle():
    """Helper to create a test puzzle."""
    return SudokuPuzzle.generate(subgrid_size=2, num_missing_cells=2)


class TestRuntimePrimitiveDetection:
    """Test backend detection for Runtime primitives vs legacy path."""
    
    def test_requires_runtime_primitives_ibm_backend(self):
        """IBM backends from qiskit_ibm_runtime should require primitives."""
        # Create mock IBM backend
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
        
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        assert solver._requires_runtime_primitives(mock_backend) is True
    
    def test_requires_runtime_primitives_aer_simulator(self):
        """Aer simulators should use legacy path."""
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_aer.backends.aer_simulator'
        
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        assert solver._requires_runtime_primitives(mock_backend) is False
    
    def test_requires_runtime_primitives_none_backend(self):
        """None backend should return False."""
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        assert solver._requires_runtime_primitives(None) is False


class TestRuntimeSamplerExecution:
    """Test SamplerV2 execution path with mocked IBM backend."""
    
    def test_run_with_sampler_basic(self):
        """Test basic SamplerV2 execution with mocked job and result."""
        # Create test circuit
        qr = QuantumRegister(4, 'S')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(range(4))
        circuit.measure(range(4), range(4))
        
        # Mock IBM backend
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
        mock_backend.name = 'ibm_test'
        
        # Mock SamplerV2 and job
        with patch('qiskit_ibm_runtime.SamplerV2') as MockSampler:
            # Create mock job
            mock_job = Mock()
            mock_job.job_id.return_value = 'test_job_123'
            mock_job.status.return_value = 'DONE'
            mock_job.metrics.return_value = {
                'timestamps': {'created': 0, 'started': 1, 'finished': 11}
            }
            
            # Create mock result structure
            mock_bit_array = Mock()
            mock_bit_array.get_counts.return_value = {
                '0000': 256,
                '0001': 256,
                '0010': 256,
                '0011': 256
            }
            
            mock_pub_result = Mock()
            mock_pub_result.data.c = mock_bit_array  # Register name 'c'
            
            mock_primitive_result = MagicMock()  # Use MagicMock for __getitem__
            mock_primitive_result.__getitem__.return_value = mock_pub_result
            
            mock_job.result.return_value = mock_primitive_result
            
            # Mock sampler instance
            mock_sampler_instance = Mock()
            mock_sampler_instance.run.return_value = mock_job
            MockSampler.return_value = mock_sampler_instance
            
            # Create solver and run
            puzzle = create_test_puzzle()
            solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
            
            result = solver._run_with_sampler(mock_backend, circuit, shots=1024)
            
            # Verify sampler was initialized with backend
            MockSampler.assert_called_once_with(mode=mock_backend)
            
            # Verify run was called with circuit and shots
            mock_sampler_instance.run.assert_called_once_with([circuit], shots=1024)
            
            # Verify result structure
            assert hasattr(result, 'get_counts')
            counts = result.get_counts()
            assert isinstance(counts, dict)
            assert sum(counts.values()) == 1024
            assert result.job_id == 'test_job_123'
            assert result.success is True
            assert result.execution_time == 10  # 11 - 1
    
    def test_run_with_sampler_missing_runtime_package(self):
        """Test error when qiskit-ibm-runtime is not installed."""
        circuit = QuantumCircuit(2, 2)
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
        
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'qiskit_ibm_runtime': None}):
            # Force re-import by removing from locals if present
            with pytest.raises(ImportError, match="qiskit-ibm-runtime is required"):
                solver._run_with_sampler(mock_backend, circuit, shots=512)
    
    def test_run_with_sampler_alternative_register_names(self):
        """Test counts extraction with different classical register names."""
        circuit = QuantumCircuit(4, 4)
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
        
        test_cases = [
            ('c', {'0000': 100, '1111': 100}),      # Our standard
            ('meas', {'0000': 200, '1111': 200}),    # IBM examples
            ('cr', {'0000': 300, '1111': 300}),      # Common alternative
        ]
        
        for reg_name, expected_counts in test_cases:
            with patch('qiskit_ibm_runtime.SamplerV2') as MockSampler:
                # Setup mock bit array that returns counts
                mock_bit_array = Mock()
                mock_bit_array.get_counts.return_value = expected_counts
                
                # Create a simple object with dynamic attributes
                class MockData:
                    def __getattr__(self, name):
                        if name == reg_name:
                            return mock_bit_array
                        raise AttributeError(f"No attribute {name}")
                
                mock_pub_result = Mock()
                mock_pub_result.data = MockData()
                
                mock_primitive_result = MagicMock()
                mock_primitive_result.__getitem__.return_value = mock_pub_result
                
                mock_job = Mock()
                mock_job.job_id.return_value = f'job_{reg_name}'
                mock_job.status.return_value = 'DONE'
                mock_job.metrics.return_value = {'timestamps': {'created': 0, 'started': 0, 'finished': 1}}
                mock_job.result.return_value = mock_primitive_result
                
                mock_sampler = Mock()
                mock_sampler.run.return_value = mock_job
                MockSampler.return_value = mock_sampler
                
                # Execute
                puzzle = create_test_puzzle()
                solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
                result = solver._run_with_sampler(mock_backend, circuit, shots=sum(expected_counts.values()))
                
                # Verify counts extracted correctly
                assert result.get_counts() == expected_counts


class TestLegacyExecution:
    """Test legacy backend.run() path for simulators."""
    
    def test_run_legacy_aer_simulator(self):
        """Test legacy execution path with Aer-like backend."""
        # Create test circuit
        qr = QuantumRegister(2, 'S')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(range(2))
        circuit.measure(range(2), range(2))
        
        # Mock Aer simulator
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_aer.backends.aer_simulator'
        
        # Mock legacy result
        mock_counts = {'00': 256, '01': 256, '10': 256, '11': 256}
        mock_result = Mock()
        mock_result.get_counts.return_value = mock_counts
        mock_result.job_id = 'aer_job_456'
        mock_result.success = True
        
        mock_job = Mock()
        mock_job.result.return_value = mock_result
        
        mock_backend.run.return_value = mock_job
        
        # Execute
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        result = solver._run_legacy(mock_backend, circuit, shots=1024)
        
        # Verify backend.run was called
        mock_backend.run.assert_called_once_with(circuit, shots=1024)
        
        # Verify result
        assert result.get_counts() == mock_counts
        assert result.job_id == 'aer_job_456'
        assert hasattr(result, 'compiled_circuit')


class TestDualPathIntegration:
    """Test that run() correctly routes to Runtime or legacy path."""
    
    def test_run_routes_to_sampler_for_ibm_backend(self):
        """Verify run() uses Sampler for IBM backends."""
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        # Create a simple circuit to use as main_circuit
        solver.main_circuit = QuantumCircuit(4, 4)
        
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
        mock_backend.target = Mock()  # BackendV2 indicator
        
        with patch.object(solver, '_run_with_sampler') as mock_sampler:
            with patch.object(solver, '_transpile_qiskit', return_value=solver.main_circuit):
                with patch.object(solver, 'save_circuit'):  # Skip circuit saving
                    with patch.object(solver, '_detect_backend_sdk', return_value='qiskit'):
                        mock_sampler.return_value = Mock(get_counts=lambda: {'0000': 1024})
                        
                        solver.run(mock_backend, 'ibm_test', shots=1024, optimisation_level=1)
                        
                        # Verify Sampler path was used
                        mock_sampler.assert_called_once()
    
    def test_run_routes_to_legacy_for_aer_backend(self):
        """Verify run() uses legacy path for Aer."""
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        # Create a simple circuit to use as main_circuit
        solver.main_circuit = QuantumCircuit(4, 4)
        
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_aer.backends.aer_simulator'
        mock_backend.configuration = Mock()
        
        with patch.object(solver, '_run_legacy') as mock_legacy:
            with patch.object(solver, '_transpile_qiskit', return_value=solver.main_circuit):
                with patch.object(solver, 'save_circuit'):  # Skip circuit saving
                    with patch.object(solver, '_detect_backend_sdk', return_value='qiskit'):
                        mock_legacy.return_value = Mock(get_counts=lambda: {'0000': 1024})
                        
                        solver.run(mock_backend, 'aer_sim', shots=1024, optimisation_level=0)
                        
                        # Verify legacy path was used
                        mock_legacy.assert_called_once()


class TestResultCompatibility:
    """Test that both paths produce compatible result objects."""
    
    def test_both_paths_support_get_counts(self):
        """Both Runtime and legacy results should support get_counts()."""
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        
        circuit = QuantumCircuit(4, 4)
        expected_counts = {'0000': 512, '1111': 512}
        
        # Test Runtime path
        with patch('qiskit_ibm_runtime.SamplerV2') as MockSampler:
            mock_bit_array = Mock()
            mock_bit_array.get_counts.return_value = expected_counts
            mock_pub_result = Mock()
            mock_pub_result.data.c = mock_bit_array
            mock_primitive_result = MagicMock()
            mock_primitive_result.__getitem__.return_value = mock_pub_result
            
            mock_job = Mock()
            mock_job.job_id.return_value = 'runtime_job'
            mock_job.status.return_value = 'DONE'
            mock_job.metrics.return_value = {'timestamps': {'created': 0, 'started': 0, 'finished': 1}}
            mock_job.result.return_value = mock_primitive_result
            
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            MockSampler.return_value = mock_sampler
            
            mock_backend = Mock()
            mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
            
            runtime_result = solver._run_with_sampler(mock_backend, circuit, shots=1024)
            assert runtime_result.get_counts() == expected_counts
        
        # Test legacy path
        mock_result = Mock()
        mock_result.get_counts.return_value = expected_counts
        mock_job = Mock()
        mock_job.result.return_value = mock_result
        
        mock_backend = Mock()
        mock_backend.__module__ = 'qiskit_aer.backends'
        mock_backend.run.return_value = mock_job
        
        legacy_result = solver._run_legacy(mock_backend, circuit, shots=1024)
        assert legacy_result.get_counts() == expected_counts
    
    def test_both_paths_have_job_metadata(self):
        """Both paths should provide job_id and success indicators."""
        puzzle = create_test_puzzle()
        solver = ExactCoverQuantumSolver(puzzle=puzzle, encoding="pattern")
        circuit = QuantumCircuit(2, 2)
        
        # Runtime path
        with patch('qiskit_ibm_runtime.SamplerV2') as MockSampler:
            mock_bit_array = Mock()
            mock_bit_array.get_counts.return_value = {'00': 100}
            mock_pub_result = Mock()
            mock_pub_result.data.c = mock_bit_array
            mock_primitive_result = MagicMock()
            mock_primitive_result.__getitem__.return_value = mock_pub_result
            
            mock_job = Mock()
            mock_job.job_id.return_value = 'runtime_123'
            mock_job.status.return_value = 'DONE'
            mock_job.metrics.return_value = {'timestamps': {'created': 0, 'started': 0, 'finished': 1}}
            mock_job.result.return_value = mock_primitive_result
            
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            MockSampler.return_value = mock_sampler
            
            mock_backend = Mock()
            mock_backend.__module__ = 'qiskit_ibm_runtime.ibm_backend'
            
            runtime_result = solver._run_with_sampler(mock_backend, circuit, shots=100)
            assert runtime_result.job_id == 'runtime_123'
            assert runtime_result.success is True
            assert hasattr(runtime_result, 'execution_time')
        
        # Legacy path
        mock_result = Mock()
        mock_result.get_counts.return_value = {'00': 100}
        mock_result.job_id = 'legacy_456'
        mock_result.success = True
        mock_job = Mock()
        mock_job.result.return_value = mock_result
        
        mock_backend = Mock()
        mock_backend.run.return_value = mock_job
        
        legacy_result = solver._run_legacy(mock_backend, circuit, shots=100)
        assert legacy_result.job_id == 'legacy_456'
        assert legacy_result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
