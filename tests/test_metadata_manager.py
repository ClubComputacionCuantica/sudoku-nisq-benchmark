import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json
import csv

from sudoku_nisq.metadata_manager import MetadataManager

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture to create a temporary cache directory for tests."""
    return tmp_path

@pytest.fixture
def metadata_manager(temp_cache_dir):
    """Fixture for a MetadataManager instance with a temporary cache."""
    puzzle_hash = "test_puzzle_hash_123"
    return MetadataManager(cache_base=temp_cache_dir, puzzle_hash=puzzle_hash)

@pytest.fixture
def metadata_manager_with_csv(temp_cache_dir):
    """Fixture for a MetadataManager instance with CSV logging enabled."""
    puzzle_hash = "test_puzzle_hash_456"
    csv_path = temp_cache_dir / "log.csv"
    return MetadataManager(cache_base=temp_cache_dir, puzzle_hash=puzzle_hash, log_csv_path=csv_path)

def test_initialization(metadata_manager, temp_cache_dir):
    """Test MetadataManager initialization."""
    assert metadata_manager.cache_base == temp_cache_dir
    assert metadata_manager.puzzle_hash == "test_puzzle_hash_123"
    assert metadata_manager.metadata_path == temp_cache_dir / "test_puzzle_hash_123" / "metadata.json"
    assert metadata_manager._data is None
    assert metadata_manager._dirty is False

def test_load_non_existent_metadata(metadata_manager):
    """Test loading metadata when the file doesn't exist."""
    data = metadata_manager.load()
    assert data == {}
    assert metadata_manager._data == {}

def test_load_existing_metadata(metadata_manager, temp_cache_dir):
    """Test loading metadata from an existing file."""
    metadata_path = metadata_manager.metadata_path
    metadata_path.parent.mkdir(parents=True)
    test_data = {"key": "value"}
    metadata_path.write_text(json.dumps(test_data))

    data = metadata_manager.load()
    assert data == test_data
    assert metadata_manager._data == test_data

def test_save_metadata(metadata_manager):
    """Test saving metadata to a file."""
    mock_file = MagicMock()
    mock_file.name = "tempfile.tmp"
    mock_file.__enter__.return_value.name = "tempfile.tmp"

    with patch('sudoku_nisq.metadata_manager.tempfile.NamedTemporaryFile', return_value=mock_file) as mock_tempfile, \
         patch('sudoku_nisq.metadata_manager.os.fsync') as mock_fsync, \
         patch('sudoku_nisq.metadata_manager.os.replace') as mock_replace:

        md = metadata_manager.load()
        md["new_key"] = "new_value"
        metadata_manager._dirty = True

        metadata_manager.save()

        mock_tempfile.assert_called_once()
        handle = mock_file.__enter__()
        handle.write.assert_called()
        handle.flush.assert_called_once()
        mock_fsync.assert_called_once_with(handle.fileno())
        mock_replace.assert_called_once_with("tempfile.tmp", metadata_manager.metadata_path)
        assert not metadata_manager._dirty

def test_ensure_puzzle_fields(metadata_manager):
    """Test the ensure_puzzle_fields method."""
    metadata_manager.ensure_puzzle_fields(
        size=4,
        num_missing_cells=5,
        board=[[1]]
    )
    
    data = metadata_manager.load()
    assert data["size"] == 4
    assert data["num_missing_cells"] == 5
    assert data["puzzle_hash"] == "test_puzzle_hash_123"
    assert metadata_manager._dirty

def test_set_main_circuit_resources(metadata_manager):
    """Test setting main circuit resources."""
    resources = {"n_qubits": 10, "n_gates": 100}
    metadata_manager.set_main_circuit_resources(
        solver_name="TestSolver",
        encoding="simple",
        resources=resources
    )
    
    data = metadata_manager.load()
    solver_data = data["solvers"]["TestSolver"]["encodings"]["simple"]
    assert solver_data["main_circuit_resources"] == resources
    assert metadata_manager._dirty

def test_set_backend_resources(metadata_manager):
    """Test setting backend resources."""
    resources = {"n_qubits": 8, "depth": 20}
    metadata_manager.set_backend_resources(
        solver_name="TestSolver",
        encoding="simple",
        backend_alias="mock_be",
        opt_level=2,
        resources=resources
    )
    
    data = metadata_manager.load()
    backend_data = data["solvers"]["TestSolver"]["encodings"]["simple"]["backends"]["mock_be"]["2"]
    assert backend_data == resources
    assert metadata_manager._dirty

def test_csv_logging(metadata_manager_with_csv, temp_cache_dir):
    """Test that setting backend resources triggers CSV logging."""
    csv_path = metadata_manager_with_csv.log_csv_path
    
    # Set some initial data
    metadata_manager_with_csv.ensure_puzzle_fields(size=4, num_missing_cells=2, board=[])
    metadata_manager_with_csv.set_main_circuit_resources("SolverA", "enc1", {"n_qubits": 5})

    # This call should trigger a CSV write
    backend_res = {"n_qubits": 4, "error": "none"}
    metadata_manager_with_csv.set_backend_resources(
        solver_name="SolverA",
        encoding="enc1",
        backend_alias="be1",
        opt_level=1,
        resources=backend_res
    )

    assert csv_path.exists()
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == metadata_manager_with_csv._CSV_FIELDNAMES
        row = next(reader)
        # Create a dict from the row to easily check values
        row_dict = dict(zip(header, row))
        assert row_dict["puzzle_hash"] == "test_puzzle_hash_456"
        assert row_dict["solver_name"] == "SolverA"
        assert row_dict["backend_alias"] == "be1"
        assert row_dict["opt_level"] == "1"
        assert row_dict["main_n_qubits"] == "5"
        assert row_dict["backend_n_qubits"] == "4"

def test_export_full_metadata_csv(metadata_manager, temp_cache_dir):
    """Test exporting all metadata to a CSV file."""
    # Populate some data
    metadata_manager.ensure_puzzle_fields(size=4, num_missing_cells=2, board=[])
    metadata_manager.set_main_circuit_resources("SolverA", "enc1", {"n_qubits": 10})
    metadata_manager.set_backend_resources("SolverA", "enc1", "be1", 1, {"n_qubits": 8})
    metadata_manager.set_backend_resources("SolverA", "enc1", "be1", 2, {"n_qubits": 7})
    
    export_path = temp_cache_dir / "export.csv"
    metadata_manager.export_full_metadata_csv(export_path)

    assert export_path.exists()
    with open(export_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == metadata_manager._CSV_FIELDNAMES
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0][header.index("opt_level")] == "1"
        assert rows[1][header.index("opt_level")] == "2"
        assert rows[0][header.index("main_n_qubits")] == "10"