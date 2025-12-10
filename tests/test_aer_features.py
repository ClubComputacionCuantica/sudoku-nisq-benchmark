"""
Comprehensive tests for Qiskit Aer integration.

Tests cover:
- AerProvider registration and configuration
- Multiple simulation methods (statevector, density_matrix, MPS, etc.)
- Noise model integration (custom and device-based)
- BackendManager integration
- QSudoku convenience methods
- GPU detection (when available)
- Integration with existing error mitigation
"""

import pytest
from sudoku_nisq import QSudoku
from sudoku_nisq.backends import BackendManager
from sudoku_nisq.providers import AerProvider
from sudoku_nisq.solvers import ExactCoverQuantumSolver


class TestAerProvider:
    """Test AerProvider class functionality."""
    
    def test_provider_initialization(self):
        """Test AerProvider can be instantiated and configured."""
        provider = AerProvider()
        assert provider.provider_name == "aer"
        assert provider.sdk_type == "qiskit"
        assert provider.is_configured  # No auth needed
    
    def test_list_available_methods(self):
        """Test listing available simulation methods."""
        provider = AerProvider()
        methods = provider.list_available_devices()
        
        # Should include at least basic methods
        assert isinstance(methods, list)
        assert len(methods) > 0
        assert "automatic" in methods or "statevector" in methods
    
    def test_add_ideal_statevector_backend(self):
        """Test adding ideal statevector simulator."""
        provider = AerProvider()
        alias = provider.add_device("statevector", alias="test_statevector")
        
        assert alias == "test_statevector"
        backend = provider.get_device(alias)
        assert backend is not None
        # Verify it's an AerSimulator
        assert backend.__class__.__name__ == "AerSimulator"
    
    def test_add_density_matrix_backend(self):
        """Test adding density matrix simulator."""
        provider = AerProvider()
        alias = provider.add_device("density_matrix", alias="test_dm")
        
        backend = provider.get_device(alias)
        assert backend is not None
    
    def test_add_backend_with_noise_model(self):
        """Test adding backend with custom noise model."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        
        provider = AerProvider()
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(
            depolarizing_error(0.01, 2), ['cx']
        )
        
        alias = provider.add_device(
            "density_matrix",
            noise_model=noise,
            alias="noisy_dm"
        )
        
        backend = provider.get_device(alias)
        assert backend is not None
    
    def test_query_capabilities(self):
        """Test querying Aer capabilities."""
        provider = AerProvider()
        info = provider.query_available_devices()
        
        assert "methods" in info
        assert "devices" in info
        assert "has_gpu" in info
        assert "version" in info
        
        assert isinstance(info["methods"], list)
        assert isinstance(info["devices"], list)
        assert isinstance(info["has_gpu"], bool)
        assert "CPU" in info["devices"]


class TestBackendManagerAerIntegration:
    """Test BackendManager integration with AerProvider."""
    
    def setup_method(self):
        """Create fresh BackendManager for each test."""
        BackendManager._singleton = None
        self.manager = BackendManager.inst()
    
    def test_aer_provider_registered(self):
        """Test AerProvider is registered by default."""
        providers = self.manager.list_providers()
        assert "aer" in providers
    
    def test_init_aer_basic(self):
        """Test basic init_aer with defaults."""
        alias = self.manager.init_aer()
        
        assert alias.startswith("aer_")
        assert self.manager.is_registered(alias)
        backend = self.manager.get(alias)
        assert backend is not None
    
    def test_init_aer_with_method(self):
        """Test init_aer with specific simulation method."""
        alias = self.manager.init_aer(
            device="statevector",
            alias="my_statevector"
        )
        
        assert alias == "my_statevector"
        backend = self.manager.get(alias)
        assert backend is not None
    
    def test_init_aer_with_noise(self):
        """Test init_aer with noise model."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(
            depolarizing_error(0.005, 1), ['u1', 'u2', 'u3']
        )
        
        alias = self.manager.init_aer(
            device="density_matrix",
            noise_model=noise,
            alias="noisy_sim"
        )
        
        assert alias == "noisy_sim"
        backend = self.manager.get(alias)
        assert backend is not None
    
    def test_get_backend_sdk_aer(self):
        """Test SDK detection for Aer backends."""
        alias = self.manager.init_aer(device="statevector")
        sdk = self.manager.get_backend_sdk(alias)
        assert sdk == "qiskit"


class TestQSudokuAerMethods:
    """Test QSudoku Aer convenience methods."""
    
    def setup_method(self):
        """Create puzzle for each test."""
        BackendManager._singleton = None
        # Use 2x2 puzzle (subgrid_size=1) to avoid memory issues
        self.puzzle = QSudoku.generate(subgrid_size=1, num_missing_cells=1)
        self.puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    def test_init_aer_basic(self):
        """Test basic init_aer from QSudoku."""
        alias = self.puzzle.init_aer(method="statevector")
        
        assert alias.startswith("aer_")
        # Verify backend is attached
        manager = BackendManager.inst()
        backend = manager.get(alias)
        assert backend is not None
    
    def test_init_aer_with_custom_alias(self):
        """Test init_aer with custom alias."""
        alias = self.puzzle.init_aer(
            method="density_matrix",
            alias="my_aer"
        )
        
        assert alias == "my_aer"
    
    @pytest.mark.heavy
    def test_run_aer_enhanced(self):
        """Test enhanced run_aer with parameters."""
        result = self.puzzle.run_aer(
            shots=64,
            method="matrix_product_state",  # Use MPS to avoid memory issues
            optimization_level=0
        )
        
        assert result is not None
        # Should return Qiskit Result object
        assert hasattr(result, 'get_counts')
        counts = result.get_counts()
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 64
    
    @pytest.mark.heavy
    def test_run_aer_with_different_methods(self):
        """Test run_aer with different simulation methods."""
        # Use memory-efficient methods only
        methods = ["matrix_product_state", "automatic"]
        
        for method in methods:
            result = self.puzzle.run_aer(
                shots=64,
                method=method
            )
            assert result is not None
            counts = result.get_counts()
            assert sum(counts.values()) == 64
    
    @pytest.mark.heavy
    def test_run_aer_with_noise_custom(self):
        """Test run_aer_with_noise with custom noise model."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(
            depolarizing_error(0.01, 2), ['cx']
        )
        
        result = self.puzzle.run_aer_with_noise(
            shots=128,
            noise_model=noise,
            method="density_matrix"
        )
        
        assert result is not None
        counts = result.get_counts()
        assert sum(counts.values()) == 128
    
    @pytest.mark.skip(reason="Requires fake provider updates for recent Qiskit versions")
    def test_run_aer_with_noise_from_device(self):
        """Test run_aer_with_noise with device name."""
        pytest.importorskip("qiskit_aer")
        
        # This test may fail if fake backends are not available
        try:
            result = self.puzzle.run_aer_with_noise(
                shots=256,
                device_name="ibm_brisbane",
                method="density_matrix"
            )
            assert result is not None
        except (ValueError, ImportError) as e:
            pytest.skip(f"Fake backend not available: {e}")


class TestAerSimulationMethods:
    """Test different Aer simulation methods."""
    
    def setup_method(self):
        """Create puzzle for each test."""
        BackendManager._singleton = None
        # Use 4x4 puzzle (subgrid_size=2) with pattern encoding (3 missing cells)
        self.puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=3)
        self.puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")
    
    @pytest.mark.heavy
    def test_automatic_method(self):
        """Test automatic method selection."""
        result = self.puzzle.run_aer(
            shots=64,
            method="automatic"
        )
        assert result is not None
    
    @pytest.mark.heavy
    def test_statevector_method(self):
        """Test matrix_product_state simulation (avoiding large statevector)."""
        result = self.puzzle.run_aer(
            shots=64,
            method="matrix_product_state"
        )
        assert result is not None
        counts = result.get_counts()
        assert len(counts) > 0
    
    @pytest.mark.heavy
    def test_density_matrix_method(self):
        """Test density matrix simulation."""
        result = self.puzzle.run_aer(
            shots=64,
            method="density_matrix",
            optimization_level=0  # Simpler transpilation
        )
        assert result is not None
        counts = result.get_counts()
        assert len(counts) > 0
    
    @pytest.mark.skip(reason="Stabilizer requires Clifford-only circuits")
    def test_stabilizer_method(self):
        """Test stabilizer simulation (requires Clifford circuit)."""
        # Current Sudoku circuits are not Clifford-only
        result = self.puzzle.run_aer(
            shots=256,
            method="stabilizer"
        )
        assert result is not None


class TestAerPrecisionAndDevices:
    """Test Aer precision and device selection."""
    
    def setup_method(self):
        """Create puzzle for each test."""
        BackendManager._singleton = None
        # Use 4x4 puzzle (subgrid_size=2) with pattern encoding (3 missing cells)
        self.puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=3)
        self.puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")
    
    @pytest.mark.heavy
    def test_double_precision(self):
        """Test double precision (default)."""
        result = self.puzzle.run_aer(
            shots=64,
            method="matrix_product_state",
            precision="double"
        )
        assert result is not None
    
    @pytest.mark.heavy
    def test_single_precision(self):
        """Test single precision."""
        result = self.puzzle.run_aer(
            shots=64,
            method="matrix_product_state",
            precision="single"
        )
        assert result is not None
    
    @pytest.mark.heavy
    def test_cpu_device(self):
        """Test CPU device (default)."""
        result = self.puzzle.run_aer(
            shots=64,
            device="CPU"
        )
        assert result is not None
    
    @pytest.mark.skip(reason="GPU requires qiskit-aer-gpu and CUDA")
    def test_gpu_device(self):
        """Test GPU device (requires GPU support)."""
        # Only run if GPU is available
        provider = AerProvider()
        info = provider.query_available_devices()
        
        if not info["has_gpu"]:
            pytest.skip("GPU not available")
        
        result = self.puzzle.run_aer(
            shots=256,
            method="statevector",
            device="GPU",
            precision="single"
        )
        assert result is not None


class TestAerBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def setup_method(self):
        """Create puzzle for each test."""
        BackendManager._singleton = None
        # Use 4x4 puzzle (subgrid_size=2) with pattern encoding (3 missing cells)
        self.puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=3)
        self.puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")
    
    @pytest.mark.heavy
    def test_old_run_aer_still_works(self):
        """Test that old run_aer(shots=N) calls still work."""
        # Old usage pattern
        result = self.puzzle.run_aer(shots=128)
        
        assert result is not None
        counts = result.get_counts()
        assert sum(counts.values()) == 128
    
    @pytest.mark.heavy
    def test_default_parameters(self):
        """Test run_aer with all defaults."""
        result = self.puzzle.run_aer()
        
        assert result is not None
        counts = result.get_counts()
        assert sum(counts.values()) == 1024  # Default shots (unchanged in code)


class TestAerWithTranspilation:
    """Test Aer integration with transpilation and caching."""
    
    def setup_method(self):
        """Create puzzle for each test."""
        BackendManager._singleton = None
        # Use 4x4 puzzle (subgrid_size=2) with pattern encoding (3 missing cells)
        self.puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=3)
        self.puzzle.set_solver(
            ExactCoverQuantumSolver,
            encoding="pattern",
            store_transpiled=True
        )
    
    @pytest.mark.heavy
    def test_transpilation_levels(self):
        """Test different optimization levels."""
        # Only test opt_level 0 and 2 to reduce resource usage
        for opt_level in [0, 2]:
            result = self.puzzle.run_aer(
                shots=64,
                method="matrix_product_state",
                optimization_level=opt_level
            )
            assert result is not None
            counts = result.get_counts()
            assert sum(counts.values()) == 64
    
    def test_aer_backend_detection(self):
        """Test that Aer backend is properly detected as Qiskit SDK."""
        from qiskit_aer import AerSimulator
        
        backend = AerSimulator()
        solver = self.puzzle._solver
        
        # Should detect as qiskit
        sdk = solver._detect_backend_sdk(backend)
        assert sdk == "qiskit"


class TestAerNoiseModels:
    """Test various noise model configurations."""
    
    def setup_method(self):
        """Create puzzle for each test."""
        BackendManager._singleton = None
        # Use 4x4 puzzle (subgrid_size=2) with pattern encoding (3 missing cells)
        self.puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=3)
        self.puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")
    
    @pytest.mark.heavy
    def test_depolarizing_noise(self):
        """Test depolarizing error noise model."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(
            depolarizing_error(0.01, 1), ['u1', 'u2', 'u3']
        )
        noise.add_all_qubit_quantum_error(
            depolarizing_error(0.02, 2), ['cx']
        )
        
        result = self.puzzle.run_aer(
            shots=128,
            method="density_matrix",
            noise_model=noise
        )
        
        assert result is not None
        counts = result.get_counts()
        assert sum(counts.values()) == 128
    
    @pytest.mark.heavy
    def test_readout_error(self):
        """Test readout error noise model."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer.noise import NoiseModel, ReadoutError
        
        noise = NoiseModel()
        # Simple readout error: 5% chance of bit flip
        readout_error = ReadoutError([[0.95, 0.05], [0.05, 0.95]])
        noise.add_all_qubit_readout_error(readout_error)
        
        result = self.puzzle.run_aer(
            shots=128,
            method="matrix_product_state",  # Use MPS instead of statevector
            noise_model=noise
        )
        
        assert result is not None


class TestAerPerformanceOptions:
    """Test Aer performance and parallelization options."""
    
    def setup_method(self):
        """Create puzzle for each test."""
        BackendManager._singleton = None
        # Use 4x4 puzzle (subgrid_size=2) with pattern encoding (3 missing cells)
        self.puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=3)
        self.puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")
    
    @pytest.mark.heavy
    def test_blocking_options(self):
        """Test qubit blocking options."""
        result = self.puzzle.run_aer(
            shots=64,
            method="matrix_product_state",
            blocking_enable=True,
            blocking_qubits=5
        )
        assert result is not None
    
    @pytest.mark.heavy
    def test_seed_simulator(self):
        """Test reproducible simulation with seed."""
        result1 = self.puzzle.run_aer(
            shots=64,
            method="matrix_product_state",
            seed_simulator=42
        )
        
        result2 = self.puzzle.run_aer(
            shots=64,
            method="matrix_product_state",
            seed_simulator=42
        )
        
        # Same seed should give same results
        counts1 = result1.get_counts()
        counts2 = result2.get_counts()
        
        # Results should be identical with same seed
        assert counts1 == counts2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
