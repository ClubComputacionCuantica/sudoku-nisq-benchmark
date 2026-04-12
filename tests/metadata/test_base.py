"""Tests for StageMetadataManager base class utilities."""

import pytest
from pathlib import Path
from sudoku_nisq.metadata.base import StageMetadataManager


@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary cache directory for testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


class ConcreteManager(StageMetadataManager):
    """Concrete implementation for testing base class."""
    
    def __init__(self, storage_path: Path):
        self._storage_path = storage_path
    
    @property
    def stage_number(self) -> int:
        return 99
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def record(self, **kwargs):
        return "test_id"
    
    def query(self, **filters):
        return []


def test_load_json_empty(tmp_cache):
    """Test loading non-existent JSON returns empty dict."""
    manager = ConcreteManager(tmp_cache / "test.json")
    
    data = manager._load_json()
    assert data == {}


def test_save_and_load_json(tmp_cache):
    """Test atomic JSON save and load."""
    manager = ConcreteManager(tmp_cache / "test.json")
    
    test_data = {"key1": "value1", "nested": {"key2": 123}}
    manager._save_json(test_data)
    
    assert manager.storage_path.exists()
    
    loaded = manager._load_json()
    assert loaded == test_data


def test_load_json_corrupt(tmp_cache):
    """Test loading corrupt JSON returns empty dict with warning."""
    manager = ConcreteManager(tmp_cache / "corrupt.json")
    
    # Write invalid JSON
    manager.storage_path.write_text("{invalid json")
    
    with pytest.warns(RuntimeWarning, match="Failed to load"):
        data = manager._load_json()
        assert data == {}


def test_load_jsonl_empty(tmp_cache):
    """Test loading non-existent JSONL returns empty list."""
    manager = ConcreteManager(tmp_cache / "test.jsonl")
    
    data = manager._load_jsonl()
    assert data == []


def test_append_and_load_jsonl(tmp_cache):
    """Test JSONL append and load."""
    manager = ConcreteManager(tmp_cache / "test.jsonl")
    
    record1 = {"id": "1", "value": 100}
    record2 = {"id": "2", "value": 200}
    
    manager._append_jsonl(record1)
    manager._append_jsonl(record2)
    
    loaded = manager._load_jsonl()
    assert len(loaded) == 2
    assert loaded[0] == record1
    assert loaded[1] == record2


def test_save_jsonl(tmp_cache):
    """Test bulk JSONL save."""
    manager = ConcreteManager(tmp_cache / "test.jsonl")
    
    records = [{"id": str(i), "value": i * 10} for i in range(5)]
    manager._save_jsonl(records)
    
    loaded = manager._load_jsonl()
    assert loaded == records


def test_load_jsonl_with_empty_lines(tmp_cache):
    """Test JSONL loading skips empty lines."""
    manager = ConcreteManager(tmp_cache / "test.jsonl")
    
    # Write JSONL with empty lines
    manager.storage_path.write_text('{"id": "1"}\n\n{"id": "2"}\n')
    
    loaded = manager._load_jsonl()
    assert len(loaded) == 2


def test_load_jsonl_with_malformed_line(tmp_cache):
    """Test JSONL loading skips malformed lines with warning."""
    manager = ConcreteManager(tmp_cache / "test.jsonl")
    
    # Write JSONL with one malformed line
    manager.storage_path.write_text('{"id": "1"}\n{invalid json}\n{"id": "2"}\n')
    
    with pytest.warns(RuntimeWarning, match="Skipping malformed JSONL"):
        loaded = manager._load_jsonl()
        assert len(loaded) == 2
        assert loaded[0]["id"] == "1"
        assert loaded[1]["id"] == "2"


def test_get_nested(tmp_cache):
    """Test nested dict navigation."""
    manager = ConcreteManager(tmp_cache / "test.json")
    
    data = {
        "level1": {
            "level2": {
                "level3": "value"
            }
        }
    }
    
    result = manager._get_nested(["level1", "level2", "level3"], data=data)
    assert result == "value"
    
    # Non-existent path returns default
    result = manager._get_nested(["nonexistent", "path"], data=data, default="default_val")
    assert result == "default_val"


def test_set_nested(tmp_cache):
    """Test nested dict setting with intermediate dict creation."""
    manager = ConcreteManager(tmp_cache / "test.json")
    
    data = {}
    result = manager._set_nested(["level1", "level2", "level3"], "new_value", data=data)
    
    assert result["level1"]["level2"]["level3"] == "new_value"


def test_exists(tmp_cache):
    """Test exists() method."""
    manager = ConcreteManager(tmp_cache / "test.json")
    
    assert manager.exists() is False
    
    manager.storage_path.write_text("{}")
    assert manager.exists() is True


def test_clear(tmp_cache):
    """Test clear() method removes storage file."""
    manager = ConcreteManager(tmp_cache / "test.json")
    
    manager.storage_path.write_text("{}")
    assert manager.exists() is True
    
    manager.clear()
    assert manager.exists() is False


def test_repr(tmp_cache):
    """Test __repr__ output."""
    manager = ConcreteManager(tmp_cache / "test.json")
    repr_str = repr(manager)
    
    assert "ConcreteManager" in repr_str
    assert "stage=99" in repr_str
    assert "test.json" in repr_str
