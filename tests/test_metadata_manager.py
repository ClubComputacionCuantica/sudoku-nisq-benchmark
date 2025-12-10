"""
Comprehensive tests for MetadataManager class.

Tests cover:
- Initialization and lazy loading
- Atomic file operations with error recovery
- Puzzle metadata management
- Circuit resource tracking
- Backend resource management
- Data persistence and caching
- Edge cases and error conditions
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from sudoku_nisq.metadata_manager import MetadataManager


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary directory for cache storage."""
    cache_dir = tmp_path / ".quantum_cache"
    return cache_dir


@pytest.fixture
def puzzle_hash():
    """Provide a sample puzzle hash."""
    return "d6d5713893ca50df092c48b099d9eeb4e1a22fbd"


@pytest.fixture
def metadata_manager(temp_cache_dir, puzzle_hash):
    """Provide a fresh MetadataManager instance for each test."""
    return MetadataManager(temp_cache_dir, puzzle_hash)


@pytest.fixture
def sample_board():
    """Provide a sample 4x4 Sudoku board."""
    return [
        [1, 0, 3, 0],
        [0, 0, 0, 4],
        [0, 3, 0, 0],
        [4, 0, 0, 2]
    ]


@pytest.fixture
def sample_resources():
    """Provide sample circuit resources."""
    return {
        "n_qubits": 16,
        "n_gates": 256,
        "n_mcx_gates": 12,
        "depth": 64
    }


class TestInitialization:
    """Test MetadataManager initialization and basic setup."""
    
    def test_init_sets_attributes(self, temp_cache_dir, puzzle_hash):
        """Test that initialization sets all required attributes."""
        manager = MetadataManager(temp_cache_dir, puzzle_hash, sort_keys=False)
        
        assert manager.cache_base == temp_cache_dir
        assert manager.puzzle_hash == puzzle_hash
        assert manager.sort_keys is False
        assert manager.metadata_path == temp_cache_dir / puzzle_hash / "metadata.json"
        assert manager._data is None
        assert manager._dirty is False
    
    def test_init_default_sort_keys(self, temp_cache_dir, puzzle_hash):
        """Test that sort_keys defaults to True."""
        manager = MetadataManager(temp_cache_dir, puzzle_hash)
        assert manager.sort_keys is True
    
    def test_metadata_path_construction(self, temp_cache_dir, puzzle_hash):
        """Test that metadata path is correctly constructed."""
        manager = MetadataManager(temp_cache_dir, puzzle_hash)
        expected_path = temp_cache_dir / puzzle_hash / "metadata.json"
        assert manager.metadata_path == expected_path


class TestLazyLoading:
    """Test lazy loading behavior of metadata."""
    
    def test_load_creates_empty_dict_when_file_missing(self, metadata_manager):
        """Test that load() creates empty dict when file doesn't exist."""
        data = metadata_manager.load()
        
        assert data == {}
        assert metadata_manager._data is not None
        assert metadata_manager._dirty is False
    
    def test_load_returns_cached_data_on_subsequent_calls(self, metadata_manager):
        """Test that load() returns same object reference on multiple calls."""
        first_load = metadata_manager.load()
        second_load = metadata_manager.load()
        
        assert first_load is second_load
    
    def test_load_reads_existing_file(self, metadata_manager):
        """Test that load() reads data from existing metadata file."""
        # Create metadata file with data
        test_data = {
            "puzzle_hash": "test123",
            "size": 9,
            "solvers": {"TestSolver": {}}
        }
        metadata_manager.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_manager.metadata_path.write_text(json.dumps(test_data))
        
        loaded_data = metadata_manager.load()
        
        assert loaded_data == test_data
        assert metadata_manager._dirty is False
    
    def test_load_handles_corrupt_json(self, metadata_manager):
        """Test that load() handles corrupt JSON with warning."""
        # Create corrupt metadata file
        metadata_manager.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_manager.metadata_path.write_text("{ invalid json }")
        
        with pytest.warns(UserWarning, match="corrupt; resetting metadata"):
            data = metadata_manager.load()
        
        assert data == {}
        assert metadata_manager._dirty is False


class TestAtomicSave:
    """Test atomic save operations and persistence."""
    
    def test_save_does_nothing_when_not_dirty(self, metadata_manager):
        """Test that save() skips I/O when data hasn't changed."""
        metadata_manager.load()
        
        # Ensure directory doesn't exist yet
        assert not metadata_manager.metadata_path.exists()
        
        metadata_manager.save()
        
        # Should not create file if not dirty
        assert not metadata_manager.metadata_path.exists()
    
    def test_save_creates_directory_structure(self, metadata_manager):
        """Test that save() creates necessary directory structure."""
        metadata_manager.load()
        metadata_manager._dirty = True
        
        metadata_manager.save()
        
        assert metadata_manager.metadata_path.parent.exists()
        assert metadata_manager.metadata_path.exists()
    
    def test_save_persists_data_to_disk(self, metadata_manager, sample_board):
        """Test that save() correctly writes data to disk."""
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        
        metadata_manager.save()
        
        # Read file directly and verify
        with open(metadata_manager.metadata_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["size"] == 4
        assert saved_data["num_missing_cells"] == 10
        assert json.loads(saved_data["board"]) == sample_board
    
    def test_save_clears_dirty_flag(self, metadata_manager):
        """Test that save() clears the dirty flag."""
        metadata_manager.load()
        metadata_manager._dirty = True
        
        metadata_manager.save()
        
        assert metadata_manager._dirty is False
    
    def test_save_uses_sort_keys_setting(self, temp_cache_dir, puzzle_hash, sample_board):
        """Test that save() respects sort_keys setting."""
        # Test with sort_keys=True
        manager_sorted = MetadataManager(temp_cache_dir, puzzle_hash, sort_keys=True)
        manager_sorted.ensure_puzzle_fields(size=4, num_missing_cells=10, board=sample_board)
        manager_sorted.save()
        
        content_sorted = manager_sorted.metadata_path.read_text()
        
        # Test with sort_keys=False
        hash2 = "different_hash"
        manager_unsorted = MetadataManager(temp_cache_dir, hash2, sort_keys=False)
        manager_unsorted.ensure_puzzle_fields(size=4, num_missing_cells=10, board=sample_board)
        manager_unsorted.save()
        
        content_unsorted = manager_unsorted.metadata_path.read_text()
        
        # Both should be valid JSON
        json.loads(content_sorted)
        json.loads(content_unsorted)


class TestUnload:
    """Test memory management with unload()."""
    
    def test_unload_clears_cached_data(self, metadata_manager):
        """Test that unload() releases cached data."""
        metadata_manager.load()
        assert metadata_manager._data is not None
        
        metadata_manager.unload()
        
        assert metadata_manager._data is None
        assert metadata_manager._dirty is False
    
    def test_unload_discards_unsaved_changes(self, metadata_manager, sample_board):
        """Test that unload() discards unsaved modifications."""
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        assert metadata_manager._dirty is True
        
        metadata_manager.unload()
        
        # Load again - should not have the changes
        data = metadata_manager.load()
        assert "size" not in data
    
    def test_reload_after_unload(self, metadata_manager, sample_board):
        """Test that data can be reloaded after unload()."""
        # Save some data
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        metadata_manager.save()
        
        # Unload and reload
        metadata_manager.unload()
        data = metadata_manager.load()
        
        assert data["size"] == 4
        assert data["num_missing_cells"] == 10


class TestPuzzleFields:
    """Test puzzle-level metadata management."""
    
    def test_ensure_puzzle_fields_sets_all_fields(self, metadata_manager, puzzle_hash, sample_board):
        """Test that ensure_puzzle_fields sets all required fields."""
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        
        data = metadata_manager.load()
        
        assert data["puzzle_hash"] == puzzle_hash
        assert data["size"] == 4
        assert data["num_missing_cells"] == 10
        assert json.loads(data["board"]) == sample_board
    
    def test_ensure_puzzle_fields_marks_dirty(self, metadata_manager, sample_board):
        """Test that ensure_puzzle_fields sets dirty flag."""
        assert metadata_manager._dirty is False
        
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        
        assert metadata_manager._dirty is True
    
    def test_ensure_puzzle_fields_unchanged_not_dirty(self, metadata_manager, sample_board):
        """Test that setting same values doesn't mark as dirty."""
        # Set initial data
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        metadata_manager.save()
        
        # Set same data again
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        
        assert metadata_manager._dirty is False
    
    def test_ensure_puzzle_fields_board_serialization(self, metadata_manager):
        """Test that board is correctly serialized to JSON string."""
        board = [[1, 2], [3, 4]]
        
        metadata_manager.ensure_puzzle_fields(
            size=2,
            num_missing_cells=0,
            board=board
        )
        
        data = metadata_manager.load()
        stored_board = data["board"]
        
        # Should be stored as JSON string
        assert isinstance(stored_board, str)
        assert json.loads(stored_board) == board


class TestCircuitResources:
    """Test main circuit resource tracking."""
    
    def test_set_main_circuit_resources_creates_structure(self, metadata_manager, sample_resources):
        """Test that set_main_circuit_resources creates proper nested structure."""
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        data = metadata_manager.load()
        
        assert "solvers" in data
        assert "ExactCoverSolver" in data["solvers"]
        assert "encodings" in data["solvers"]["ExactCoverSolver"]
        assert "pattern" in data["solvers"]["ExactCoverSolver"]["encodings"]
    
    def test_set_main_circuit_resources_stores_data(self, metadata_manager, sample_resources):
        """Test that circuit resources are correctly stored."""
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        data = metadata_manager.load()
        stored = data["solvers"]["ExactCoverSolver"]["encodings"]["pattern"]["main_circuit_resources"]
        
        assert stored == sample_resources
    
    def test_set_main_circuit_resources_marks_dirty(self, metadata_manager, sample_resources):
        """Test that setting resources marks data as dirty."""
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        assert metadata_manager._dirty is True
    
    def test_set_main_circuit_resources_unchanged_not_dirty(self, metadata_manager, sample_resources):
        """Test that setting identical resources doesn't mark as dirty."""
        # Set initial resources
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.save()
        
        # Set same resources again
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        assert metadata_manager._dirty is False
    
    def test_set_main_circuit_resources_multiple_encodings(self, metadata_manager, sample_resources):
        """Test storing resources for multiple encodings."""
        resources_pattern = sample_resources
        resources_simple = {**sample_resources, "n_qubits": 20}
        
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=resources_pattern
        )
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="simple",
            resources=resources_simple
        )
        
        data = metadata_manager.load()
        encodings = data["solvers"]["ExactCoverSolver"]["encodings"]
        
        assert "pattern" in encodings
        assert "simple" in encodings
        assert encodings["pattern"]["main_circuit_resources"]["n_qubits"] == 16
        assert encodings["simple"]["main_circuit_resources"]["n_qubits"] == 20
    
    def test_set_main_circuit_resources_sdk_type_default(self, metadata_manager, sample_resources):
        """Test that sdk_type defaults to 'pytket'."""
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        data = metadata_manager.load()
        sdk_type = data["solvers"]["ExactCoverSolver"]["encodings"]["pattern"]["sdk_type"]
        
        assert sdk_type == "pytket"
    
    def test_set_main_circuit_resources_custom_sdk_type(self, metadata_manager, sample_resources):
        """Test setting custom sdk_type."""
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources,
            sdk_type="qiskit"
        )
        
        data = metadata_manager.load()
        sdk_type = data["solvers"]["ExactCoverSolver"]["encodings"]["pattern"]["sdk_type"]
        
        assert sdk_type == "qiskit"


class TestBackendResources:
    """Test backend-specific resource tracking."""
    
    def test_set_backend_resources_creates_structure(self, metadata_manager):
        """Test that backend resources create proper nested structure."""
        resources = {"n_qubits": 127, "n_gates": 5000, "depth": 1200}
        
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=resources
        )
        
        data = metadata_manager.load()
        
        assert "solvers" in data
        assert "ExactCoverSolver" in data["solvers"]
        assert "encodings" in data["solvers"]["ExactCoverSolver"]
        assert "pattern" in data["solvers"]["ExactCoverSolver"]["encodings"]
        assert "backends" in data["solvers"]["ExactCoverSolver"]["encodings"]["pattern"]
        assert "ibm_brisbane" in data["solvers"]["ExactCoverSolver"]["encodings"]["pattern"]["backends"]
    
    def test_set_backend_resources_stores_by_opt_level(self, metadata_manager):
        """Test that backend resources are keyed by optimization level."""
        resources_opt2 = {"n_qubits": 127, "n_gates": 5000, "depth": 1200}
        resources_opt3 = {"n_qubits": 127, "n_gates": 4500, "depth": 1000}
        
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=resources_opt2
        )
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=3,
            resources=resources_opt3
        )
        
        data = metadata_manager.load()
        backend_data = data["solvers"]["ExactCoverSolver"]["encodings"]["pattern"]["backends"]["ibm_brisbane"]
        
        assert "2" in backend_data
        assert "3" in backend_data
        assert backend_data["2"]["n_gates"] == 5000
        assert backend_data["3"]["n_gates"] == 4500
    
    def test_set_backend_resources_marks_dirty(self, metadata_manager):
        """Test that setting backend resources marks data as dirty."""
        resources = {"n_qubits": 127, "n_gates": 5000, "depth": 1200}
        
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=resources
        )
        
        assert metadata_manager._dirty is True
    
    def test_set_backend_resources_unchanged_not_dirty(self, metadata_manager):
        """Test that setting identical backend resources doesn't mark as dirty."""
        resources = {"n_qubits": 127, "n_gates": 5000, "depth": 1200}
        
        # Set initial resources
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=resources
        )
        metadata_manager.save()
        
        # Set same resources again
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=resources
        )
        
        assert metadata_manager._dirty is False
    
    def test_set_backend_resources_multiple_backends(self, metadata_manager):
        """Test storing resources for multiple backends."""
        resources_brisbane = {"n_qubits": 127, "n_gates": 5000, "depth": 1200}
        resources_kyoto = {"n_qubits": 127, "n_gates": 4800, "depth": 1100}
        
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=resources_brisbane
        )
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_kyoto",
            opt_level=2,
            resources=resources_kyoto
        )
        
        data = metadata_manager.load()
        backends = data["solvers"]["ExactCoverSolver"]["encodings"]["pattern"]["backends"]
        
        assert "ibm_brisbane" in backends
        assert "ibm_kyoto" in backends
        assert backends["ibm_brisbane"]["2"]["n_gates"] == 5000
        assert backends["ibm_kyoto"]["2"]["n_gates"] == 4800


class TestSolverManagement:
    """Test solver data management operations."""
    
    def test_remove_solver_deletes_data(self, metadata_manager, sample_resources):
        """Test that remove_solver deletes all solver data."""
        # Add solver data
        metadata_manager.set_main_circuit_resources(
            solver_name="OldSolver",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.save()
        
        # Remove solver
        metadata_manager.remove_solver("OldSolver")
        
        data = metadata_manager.load()
        assert "OldSolver" not in data.get("solvers", {})
    
    def test_remove_solver_marks_dirty(self, metadata_manager, sample_resources):
        """Test that removing solver marks data as dirty."""
        # Add solver data
        metadata_manager.set_main_circuit_resources(
            solver_name="OldSolver",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.save()
        
        # Remove solver
        metadata_manager.remove_solver("OldSolver")
        
        assert metadata_manager._dirty is True
    
    def test_remove_solver_nonexistent_not_dirty(self, metadata_manager):
        """Test that removing nonexistent solver doesn't mark as dirty."""
        metadata_manager.load()
        
        metadata_manager.remove_solver("NonexistentSolver")
        
        assert metadata_manager._dirty is False
    
    def test_remove_solver_preserves_other_solvers(self, metadata_manager, sample_resources):
        """Test that removing one solver preserves others."""
        # Add multiple solvers
        metadata_manager.set_main_circuit_resources(
            solver_name="Solver1",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.set_main_circuit_resources(
            solver_name="Solver2",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.save()
        
        # Remove one solver
        metadata_manager.remove_solver("Solver1")
        
        data = metadata_manager.load()
        assert "Solver1" not in data["solvers"]
        assert "Solver2" in data["solvers"]
    
    def test_get_solver_data_returns_data(self, metadata_manager, sample_resources):
        """Test that get_solver_data returns correct solver data."""
        metadata_manager.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        solver_data = metadata_manager.get_solver_data("TestSolver")
        
        assert solver_data is not None
        assert "encodings" in solver_data
        assert "pattern" in solver_data["encodings"]
    
    def test_get_solver_data_returns_none_for_missing(self, metadata_manager):
        """Test that get_solver_data returns None for nonexistent solver."""
        metadata_manager.load()
        
        solver_data = metadata_manager.get_solver_data("NonexistentSolver")
        
        assert solver_data is None


class TestResourceSummary:
    """Test resource summary generation."""
    
    def test_get_resource_summary_structure(self, metadata_manager, sample_board, sample_resources):
        """Test that resource summary has correct structure."""
        # Set up puzzle and solver data
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        metadata_manager.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        summary = metadata_manager.get_resource_summary()
        
        assert "puzzle_info" in summary
        assert "solvers" in summary
        assert "hash" in summary["puzzle_info"]
        assert "size" in summary["puzzle_info"]
        assert "num_missing_cells" in summary["puzzle_info"]
    
    def test_get_resource_summary_puzzle_info(self, metadata_manager, puzzle_hash, sample_board):
        """Test that puzzle info is correctly included in summary."""
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        
        summary = metadata_manager.get_resource_summary()
        puzzle_info = summary["puzzle_info"]
        
        assert puzzle_info["hash"] == puzzle_hash
        assert puzzle_info["size"] == 4
        assert puzzle_info["num_missing_cells"] == 10
    
    def test_get_resource_summary_solver_data(self, metadata_manager, sample_resources):
        """Test that solver data is correctly structured in summary."""
        metadata_manager.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        summary = metadata_manager.get_resource_summary()
        
        assert "TestSolver" in summary["solvers"]
        assert "pattern" in summary["solvers"]["TestSolver"]
        assert "main_circuit" in summary["solvers"]["TestSolver"]["pattern"]
        assert "backends" in summary["solvers"]["TestSolver"]["pattern"]
    
    def test_get_resource_summary_main_circuit_resources(self, metadata_manager, sample_resources):
        """Test that main circuit resources are included in summary."""
        metadata_manager.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="pattern",
            resources=sample_resources
        )
        
        summary = metadata_manager.get_resource_summary()
        main_circuit = summary["solvers"]["TestSolver"]["pattern"]["main_circuit"]
        
        assert main_circuit == sample_resources
    
    def test_get_resource_summary_backend_data(self, metadata_manager, sample_resources):
        """Test that backend data is included in summary."""
        backend_resources = {"n_qubits": 127, "n_gates": 5000, "depth": 1200}
        
        metadata_manager.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.set_backend_resources(
            solver_name="TestSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=backend_resources
        )
        
        summary = metadata_manager.get_resource_summary()
        backends = summary["solvers"]["TestSolver"]["pattern"]["backends"]
        
        assert "ibm_brisbane" in backends
        assert backends["ibm_brisbane"]["2"] == backend_resources
    
    def test_get_resource_summary_empty_metadata(self, metadata_manager):
        """Test resource summary with empty metadata."""
        summary = metadata_manager.get_resource_summary()
        
        assert summary["puzzle_info"]["hash"] is None
        assert summary["puzzle_info"]["size"] is None
        assert summary["puzzle_info"]["num_missing_cells"] is None
        assert summary["solvers"] == {}
    
    def test_get_resource_summary_multiple_solvers(self, metadata_manager, sample_resources):
        """Test resource summary with multiple solvers."""
        metadata_manager.set_main_circuit_resources(
            solver_name="Solver1",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.set_main_circuit_resources(
            solver_name="Solver2",
            encoding="simple",
            resources={**sample_resources, "n_qubits": 20}
        )
        
        summary = metadata_manager.get_resource_summary()
        
        assert "Solver1" in summary["solvers"]
        assert "Solver2" in summary["solvers"]
        assert summary["solvers"]["Solver1"]["pattern"]["main_circuit"]["n_qubits"] == 16
        assert summary["solvers"]["Solver2"]["simple"]["main_circuit"]["n_qubits"] == 20


class TestIntegrationScenarios:
    """Test complete workflows and integration scenarios."""
    
    def test_complete_workflow(self, metadata_manager, puzzle_hash, sample_board, sample_resources):
        """Test a complete workflow from puzzle setup to backend resources."""
        # 1. Set puzzle fields
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        
        # 2. Add main circuit resources
        metadata_manager.set_main_circuit_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            resources=sample_resources,
            sdk_type="qiskit"
        )
        
        # 3. Add backend resources
        backend_resources = {"n_qubits": 127, "n_gates": 5000, "depth": 1200}
        metadata_manager.set_backend_resources(
            solver_name="ExactCoverSolver",
            encoding="pattern",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources=backend_resources
        )
        
        # 4. Save to disk
        metadata_manager.save()
        
        # 5. Verify by creating new instance and loading
        new_manager = MetadataManager(metadata_manager.cache_base, puzzle_hash)
        summary = new_manager.get_resource_summary()
        
        assert summary["puzzle_info"]["size"] == 4
        assert summary["puzzle_info"]["num_missing_cells"] == 10
        assert "ExactCoverSolver" in summary["solvers"]
        assert "pattern" in summary["solvers"]["ExactCoverSolver"]
        assert summary["solvers"]["ExactCoverSolver"]["pattern"]["main_circuit"] == sample_resources
        assert "ibm_brisbane" in summary["solvers"]["ExactCoverSolver"]["pattern"]["backends"]
    
    def test_multi_solver_comparison(self, metadata_manager, sample_board):
        """Test workflow for comparing multiple solvers."""
        # Set puzzle
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        
        # Add multiple solvers with different characteristics
        solvers = {
            "ExactCover": {"n_qubits": 16, "n_gates": 256, "depth": 64},
            "Backtracking": {"n_qubits": 20, "n_gates": 512, "depth": 128},
            "GroversSearch": {"n_qubits": 24, "n_gates": 1024, "depth": 256}
        }
        
        for solver_name, resources in solvers.items():
            metadata_manager.set_main_circuit_resources(
                solver_name=solver_name,
                encoding="pattern",
                resources=resources
            )
        
        metadata_manager.save()
        
        # Verify all solvers are present
        summary = metadata_manager.get_resource_summary()
        for solver_name in solvers:
            assert solver_name in summary["solvers"]
    
    def test_persistence_across_instances(self, temp_cache_dir, puzzle_hash, sample_board):
        """Test that data persists correctly across different instances."""
        # First instance - write data
        manager1 = MetadataManager(temp_cache_dir, puzzle_hash)
        manager1.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        manager1.save()
        
        # Second instance - modify data
        manager2 = MetadataManager(temp_cache_dir, puzzle_hash)
        manager2.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="pattern",
            resources={"n_qubits": 16, "n_gates": 256}
        )
        manager2.save()
        
        # Third instance - verify both changes persist
        manager3 = MetadataManager(temp_cache_dir, puzzle_hash)
        data = manager3.load()
        
        assert data["size"] == 4
        assert "TestSolver" in data["solvers"]
    
    def test_memory_management_with_unload(self, metadata_manager, sample_board, sample_resources):
        """Test proper memory management using load/unload cycle."""
        # Initial load and modification
        metadata_manager.ensure_puzzle_fields(
            size=4,
            num_missing_cells=10,
            board=sample_board
        )
        metadata_manager.save()
        
        # Unload to free memory
        metadata_manager.unload()
        assert metadata_manager._data is None
        
        # Reload and add more data
        metadata_manager.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="pattern",
            resources=sample_resources
        )
        metadata_manager.save()
        
        # Verify everything is still there
        summary = metadata_manager.get_resource_summary()
        assert summary["puzzle_info"]["size"] == 4
        assert "TestSolver" in summary["solvers"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_board(self, metadata_manager):
        """Test handling of empty board."""
        empty_board = []
        
        metadata_manager.ensure_puzzle_fields(
            size=0,
            num_missing_cells=0,
            board=empty_board
        )
        
        data = metadata_manager.load()
        assert json.loads(data["board"]) == []
    
    def test_large_board(self, metadata_manager):
        """Test handling of large board (16x16)."""
        large_board = [[i * 16 + j for j in range(16)] for i in range(16)]
        
        metadata_manager.ensure_puzzle_fields(
            size=16,
            num_missing_cells=200,
            board=large_board
        )
        metadata_manager.save()
        
        data = metadata_manager.load()
        assert json.loads(data["board"]) == large_board
    
    def test_special_characters_in_solver_name(self, metadata_manager):
        """Test handling of special characters in solver names."""
        solver_name = "My-Solver_v2.0"
        
        metadata_manager.set_main_circuit_resources(
            solver_name=solver_name,
            encoding="pattern",
            resources={"n_qubits": 10}
        )
        
        data = metadata_manager.load()
        assert solver_name in data["solvers"]
    
    def test_zero_resources(self, metadata_manager):
        """Test handling of zero resource values."""
        zero_resources = {"n_qubits": 0, "n_gates": 0, "depth": 0}
        
        metadata_manager.set_main_circuit_resources(
            solver_name="MinimalSolver",
            encoding="minimal",
            resources=zero_resources
        )
        
        data = metadata_manager.load()
        stored = data["solvers"]["MinimalSolver"]["encodings"]["minimal"]["main_circuit_resources"]
        assert stored == zero_resources
    
    def test_very_long_puzzle_hash(self, temp_cache_dir):
        """Test handling of very long puzzle hash."""
        # Use a reasonably long hash that works on Windows (max path is ~260 chars)
        long_hash = "a" * 64  # SHA-256 hex is 64 chars, so this is realistic
        manager = MetadataManager(temp_cache_dir, long_hash)
        
        manager.load()
        manager._dirty = True
        manager.save()
        
        assert manager.metadata_path.exists()
    
    def test_unicode_in_metadata(self, metadata_manager):
        """Test handling of unicode characters in metadata."""
        board = [[1, 2], [3, 4]]
        metadata_manager.ensure_puzzle_fields(
            size=2,
            num_missing_cells=0,
            board=board
        )
        
        # Try to add unicode through solver name
        metadata_manager.set_main_circuit_resources(
            solver_name="测试求解器",  # Chinese characters
            encoding="pattern",
            resources={"n_qubits": 4}
        )
        
        metadata_manager.save()
        
        # Reload and verify
        new_manager = MetadataManager(
            metadata_manager.cache_base,
            metadata_manager.puzzle_hash
        )
        data = new_manager.load()
        assert "测试求解器" in data["solvers"]
