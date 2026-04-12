"""Tests for AerMetadataCollector."""

import pytest
from unittest.mock import Mock, patch

from sudoku_nisq.metrics.collectors.aer_collector import AerMetadataCollector
from sudoku_nisq.metrics.data_models import HardwareMetadata


class TestAerMetadataCollector:
    """Test suite for AerMetadataCollector."""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance."""
        return AerMetadataCollector()
    
    @pytest.fixture
    def mock_aer_backend(self):
        """Mock Aer backend."""
        backend = Mock()
        backend.name = "aer_simulator"
        return backend
    
    def test_collect_hardware_metadata_delegates_to_stage5(
        self, collector, mock_aer_backend
    ):
        """Test that Aer hardware collection delegates to Stage 5."""
        mock_hw_dict = {
            "backend_name": "aer_simulator",
            "provider": "Aer",
            "calibration_timestamp": "2025-12-30T12:00:00Z",
            "single_qubit_gate_errors": None,
            "two_qubit_gate_errors": None,
            "readout_errors": None,
            "t1_times": None,
            "t2_times": None,
            "extra_properties": {
                "method": "statevector",
                "device": "CPU",
                "precision": "double",
            },
        }
        
        with patch(
            "sudoku_nisq.metrics.collectors.aer_collector.stage5_collect_hw",
            return_value=mock_hw_dict,
        ):
            hw_metadata = collector.collect_hardware_metadata(mock_aer_backend)
        
        assert isinstance(hw_metadata, HardwareMetadata)
        assert hw_metadata.backend_name == "aer_simulator"
        assert hw_metadata.provider == "Aer"
        assert hw_metadata.extra_properties["method"] == "statevector"
    
    def test_inherits_compilation_extraction(self, collector):
        """Test that AerMetadataCollector inherits compilation extraction."""
        # Verify method exists
        assert hasattr(collector, "extract_compilation_metadata")
        assert callable(collector.extract_compilation_metadata)
    
    def test_inherits_execution_extraction(self, collector):
        """Test that AerMetadataCollector inherits execution extraction."""
        assert hasattr(collector, "extract_execution_result")
        assert callable(collector.extract_execution_result)
    
    def test_inherits_volume_calculation(self, collector):
        """Test that AerMetadataCollector inherits volume calculation."""
        assert hasattr(collector, "calculate_circuit_volume")
        assert callable(collector.calculate_circuit_volume)
    
    def test_can_use_with_real_aer(self, collector):
        """Integration test with real Aer simulator (if available)."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer import AerSimulator
        from qiskit import QuantumCircuit
        
        # Create real Aer backend
        backend = AerSimulator(method="statevector")
        
        # Create simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Test volume calculation (uses parent class method)
        volume = collector.calculate_circuit_volume(qc)
        assert volume == 2  # h + cx (excludes measurements)
        
        # Test hardware metadata collection
        try:
            hw_metadata = collector.collect_hardware_metadata(backend)
            # Backend name may vary depending on method (aer_simulator, aer_simulator_statevector, etc.)
            assert "aer_simulator" in hw_metadata.backend_name
            assert hw_metadata.provider.lower() == "aer"  # Case-insensitive
        except ImportError:
            # Stage 5 collector may not be available in test environment
            pytest.skip("Stage 5 collector not available")
