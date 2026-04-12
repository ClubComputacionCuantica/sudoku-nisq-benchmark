"""Tests for QiskitMetadataCollector."""

import pytest
from unittest.mock import Mock, patch

from sudoku_nisq.metrics.collectors.qiskit_collector import QiskitMetadataCollector
from sudoku_nisq.metrics.data_models import (
    HardwareMetadata,
    CompilationMetadata,
    ExecutionResult,
)


class TestQiskitMetadataCollector:
    """Test suite for QiskitMetadataCollector."""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance."""
        return QiskitMetadataCollector()
    
    @pytest.fixture
    def mock_backend(self):
        """Mock Qiskit backend."""
        backend = Mock()
        backend.name = "fake_backend"
        return backend
    
    @pytest.fixture
    def mock_circuit(self):
        """Mock Qiskit circuit."""
        circuit = Mock()
        circuit.num_qubits = 5
        circuit.depth = Mock(return_value=10)
        circuit.count_ops = Mock(return_value={"h": 5, "cx": 4, "measure": 5})
        return circuit
    
    @pytest.fixture
    def mock_compiled_circuit(self):
        """Mock transpiled Qiskit circuit."""
        circuit = Mock()
        circuit.num_qubits = 5
        circuit.depth = Mock(return_value=15)
        circuit.count_ops = Mock(return_value={"h": 5, "cx": 6, "swap": 2, "measure": 5})
        circuit._layout = None
        return circuit
    
    def test_collect_hardware_metadata_delegates_to_stage5(
        self, collector, mock_backend
    ):
        """Test that hardware collection delegates to Stage 5 collector."""
        # Mock the Stage 5 collector
        mock_hw_dict = {
            "backend_name": "fake_backend",
            "provider": "IBM",
            "calibration_timestamp": "2025-12-30T12:00:00Z",
            "single_qubit_gate_errors": {0: 0.001, 1: 0.002},
            "two_qubit_gate_errors": {(0, 1): 0.01},
            "readout_errors": {0: 0.02, 1: 0.03},
            "t1_times": {0: 100.0, 1: 120.0},
            "t2_times": {0: 80.0, 1: 90.0},
            "extra_properties": {"version": "1.0"},
        }

        with patch(
            "sudoku_nisq.metrics.collectors.qiskit_collector.stage5_collect_hw",
            return_value=mock_hw_dict,
        ):
            hw_metadata = collector.collect_hardware_metadata(mock_backend)
        
        # Verify conversion to dataclass
        assert isinstance(hw_metadata, HardwareMetadata)
        assert hw_metadata.backend_name == "fake_backend"
        assert hw_metadata.provider == "IBM"
        assert hw_metadata.single_qubit_gate_error == {0: 0.001, 1: 0.002}
    
    def test_extract_compilation_metadata_basic(
        self, collector, mock_circuit, mock_compiled_circuit, mock_backend
    ):
        """Test basic compilation metadata extraction."""
        comp_metadata = collector.extract_compilation_metadata(
            mock_circuit, mock_compiled_circuit, mock_backend
        )
        
        assert isinstance(comp_metadata, CompilationMetadata)
        assert comp_metadata.pre_transpile_gates == {
            "h": 5,
            "cx": 4,
            "measure": 5,
        }
        assert comp_metadata.post_transpile_gates == {
            "h": 5,
            "cx": 6,
            "swap": 2,
            "measure": 5,
        }
        assert comp_metadata.pre_transpile_depth == 10
        assert comp_metadata.post_transpile_depth == 15
    
    def test_extract_compilation_metadata_with_layout(
        self, collector, mock_circuit, mock_compiled_circuit, mock_backend
    ):
        """Test compilation metadata with layout information."""
        # Create mock TranspileLayout with V2 helper methods
        mock_initial_layout = Mock()
        mock_initial_layout.get_virtual_bits.return_value = {0: 0, 1: 2, 2: 4}
        
        mock_layout = Mock()
        mock_layout.initial_virtual_layout.return_value = mock_initial_layout
        mock_layout.final_index_layout.return_value = {0: 0, 1: 2, 2: 4}
        mock_layout.initial_layout = {0: 0, 1: 2, 2: 4}  # Fallback
        mock_layout.final_layout = {0: 0, 1: 2, 2: 4}    # Fallback
        
        mock_compiled_circuit.layout = mock_layout
        mock_compiled_circuit._layout = mock_layout  # Fallback
        
        comp_metadata = collector.extract_compilation_metadata(
            mock_circuit, mock_compiled_circuit, mock_backend
        )
        
        # initial_layout is now a List[int] per data model
        assert comp_metadata.initial_layout == [0, 2, 4]
    
    def test_extract_execution_result_legacy_format(
        self, collector, mock_backend, mock_compiled_circuit
    ):
        """Test execution result extraction from legacy Result format."""
        # Mock legacy Result object
        mock_result = Mock()
        mock_result.get_counts = Mock(return_value={"00": 500, "11": 500})
        mock_result.job_id = "job_123"
        mock_result.results = [Mock(shots=1000)]
        
        # Mock job with job_id() method
        mock_job = Mock()
        mock_job.job_id.return_value = "job_123"
        
        exec_result = collector.extract_execution_result(
            mock_result, mock_backend, mock_compiled_circuit, job=mock_job
        )
        
        assert isinstance(exec_result, ExecutionResult)
        assert exec_result.counts == {"00": 500, "11": 500}
        assert exec_result.shots == 1000
        assert exec_result.backend_name == "fake_backend"
        assert exec_result.num_qubits == 5
        assert exec_result.circuit_depth == 15
        assert exec_result.job_id == "job_123"
        assert exec_result.two_qubit_gates == 8  # 6 cx + 2 swap
    
    def test_extract_execution_result_samplerv2_format(
        self, collector, mock_backend, mock_compiled_circuit
    ):
        """Test execution result extraction from SamplerV2 format."""
        # Mock BitArray returned by join_data()
        mock_bit_array = Mock()
        mock_bit_array.get_counts.return_value = {"00": 400, "11": 600}
        mock_bit_array.num_shots = 1000
        
        mock_pub_result = Mock()
        mock_pub_result.join_data.return_value = mock_bit_array
        mock_pub_result.metadata = {"shots": 1000}
        
        mock_result = Mock()
        mock_result.__getitem__ = Mock(return_value=mock_pub_result)
        mock_result.get_counts = None  # No legacy method
        
        exec_result = collector.extract_execution_result(
            mock_result, mock_backend, mock_compiled_circuit
        )
        
        assert exec_result.counts == {"00": 400, "11": 600}
        assert exec_result.shots == 1000
    
    def test_extract_execution_result_infers_shots(
        self, collector, mock_backend, mock_compiled_circuit
    ):
        """Test that shots are inferred from counts if not available."""
        mock_result = Mock()
        mock_result.get_counts = Mock(return_value={"00": 300, "11": 700})
        mock_result.job_id = None
        mock_result.results = []  # No shots metadata
        
        exec_result = collector.extract_execution_result(
            mock_result, mock_backend, mock_compiled_circuit
        )
        
        assert exec_result.shots == 1000  # 300 + 700
    
    def test_extract_execution_result_unrecognized_format_raises(
        self, collector, mock_backend, mock_compiled_circuit
    ):
        """Test that unrecognized result format raises ValueError."""
        mock_result = Mock(spec=[])  # No get_counts, no __getitem__
        
        with pytest.raises(ValueError, match="Unrecognized result format"):
            collector.extract_execution_result(
                mock_result, mock_backend, mock_compiled_circuit
            )
    
    def test_calculate_circuit_volume_simple_circuit(self, collector):
        """Test circuit volume calculation for simple circuit."""
        # We need a real QuantumCircuit for this test
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit
        
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        
        volume = collector.calculate_circuit_volume(qc)
        
        # Volume should be 3 gates (h, cx, cx) - excludes measurements
        assert volume == 3
    
    def test_calculate_circuit_volume_with_barriers(self, collector):
        """Test that barriers are excluded from volume."""
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        
        volume = collector.calculate_circuit_volume(qc)
        
        assert volume == 2  # h and cx, no barrier
    
    def test_calculate_circuit_volume_no_qiskit_returns_none(self, collector):
        """Test that missing qiskit returns None gracefully."""
        mock_circuit = Mock()

        with patch(
            "sudoku_nisq.metrics.collectors.qiskit_collector.circuit_to_dag",
            None,
        ):
            volume = collector.calculate_circuit_volume(mock_circuit)
            assert volume is None
    
    def test_calculate_circuit_volume_error_returns_none(self, collector):
        """Test that calculation errors return None gracefully."""
        mock_circuit = Mock()
        
        with patch(
            "sudoku_nisq.metrics.collectors.qiskit_collector.circuit_to_dag",
            side_effect=RuntimeError("DAG error"),
        ):
            volume = collector.calculate_circuit_volume(mock_circuit)
            assert volume is None
