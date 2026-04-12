"""Abstract base class for stage-specific metadata managers.

Provides shared utilities for JSON/JSONL storage, atomic writes, and query interfaces.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import tempfile
import os
import threading


_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class StageMetadataManager(ABC):
    """Abstract base for stage-specific metadata persistence.
    
    Each concrete manager:
    1. Owns a specific file (JSON or JSONL format)
    2. Auto-extracts metadata from domain objects (circuits, results, etc.)
    3. Provides query interface for its stage
    4. Handles atomic writes and file locking
    
    Subclasses must implement:
    - stage_number: Integer identifier (1-7)
    - storage_path: Path property returning file location
    - record(**kwargs): Store stage data, return unique ID
    - query(**filters): Retrieve records matching criteria
    """
    
    @property
    @abstractmethod
    def stage_number(self) -> int:
        """Stage identifier (1-7)."""
        pass
    
    @property
    @abstractmethod
    def storage_path(self) -> Path:
        """Path to this stage's storage file (JSON or JSONL)."""
        pass
    
    @abstractmethod
    def record(self, **kwargs: Any) -> Any:
        """Record stage-specific data.
        
        :param kwargs: Stage-specific parameters
        :returns: Unique identifier for this record (puzzle_hash, circuit_hash, run_id, etc.)
        """
        pass
    
    @abstractmethod
    def query(self, **filters: Any) -> Any:
        """Retrieve records matching filter criteria.
        
        :param filters: Stage-specific filter parameters
        :returns: List of matching records, single dict, or None
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Return JSON schema for validation (optional extension point).
        
        :returns: JSON schema dict (default: empty)
        """
        return {}
    
    # Shared utilities for concrete implementations

    def _get_path_lock(self) -> threading.Lock:
        """Get a process-local lock for this manager's storage path.

        This prevents interleaved writes from multiple threads in the same
        Python process. It intentionally does not attempt cross-process locking
        to avoid platform-specific instability.
        """
        key = str(self.storage_path)
        with _PATH_LOCKS_GUARD:
            if key not in _PATH_LOCKS:
                _PATH_LOCKS[key] = threading.Lock()
            return _PATH_LOCKS[key]
    
    def _load_json(self) -> Dict[str, Any]:
        """Load JSON file into dict.
        
        :returns: Parsed JSON dict, or empty dict if file doesn't exist
        """
        if not self.storage_path.exists():
            return {}
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            import warnings
            warnings.warn(
                f"Failed to load {self.storage_path}: {e}. Returning empty dict.",
                RuntimeWarning
            )
            return {}
    
    def _save_json(self, data: Dict[str, Any]) -> None:
        """Atomically save dict to JSON file.
        
        Uses tempfile + rename pattern for atomic writes.
        
        :param data: Dict to serialize as JSON
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self._get_path_lock()
        
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.storage_path.parent,
            prefix=f".tmp_{self.storage_path.name}_",
            text=True
        )
        
        try:
            with lock:
                with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass

                # Atomic rename
                os.replace(tmp_path, self.storage_path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    
    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """Load JSONL file into list of dicts.
        
        :returns: List of parsed JSON objects, or empty list if file doesn't exist
        """
        if not self.storage_path.exists():
            return []
        
        records = []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        import warnings
                        warnings.warn(
                            f"Skipping malformed JSONL at {self.storage_path}:{line_num}: {e}",
                            RuntimeWarning
                        )
        except OSError as e:
            import warnings
            warnings.warn(
                f"Failed to load {self.storage_path}: {e}. Returning empty list.",
                RuntimeWarning
            )
        
        return records
    
    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        """Append single record to JSONL file.
        
        Uses append mode (no need to load full file).
        
        :param record: Dict to append as JSON line
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        lock = self._get_path_lock()
        with lock:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
    
    def _save_jsonl(self, records: List[Dict[str, Any]]) -> None:
        """Atomically save list of dicts to JSONL file.
        
        Uses tempfile + rename for atomic writes. Use for bulk updates.
        For single appends, prefer _append_jsonl().
        
        :param records: List of dicts to serialize as JSONL
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.storage_path.parent,
            prefix=f".tmp_{self.storage_path.name}_",
            text=True
        )
        
        lock = self._get_path_lock()

        try:
            with lock:
                with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass

                os.replace(tmp_path, self.storage_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    
    def _get_nested(self, keys: List[str], data: Optional[Dict] = None, default: Any = None) -> Any:
        """Navigate nested dict with key path.
        
        :param keys: List of keys defining nested path (e.g., ["solvers", "exact_cover", "pattern"])
        :param data: Dict to navigate (default: load from storage)
        :param default: Value to return if path doesn't exist
        :returns: Value at nested path, or default
        """
        if data is None:
            data = self._load_json()
        
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        
        return current
    
    def _set_nested(self, keys: List[str], value: Any, data: Optional[Dict] = None) -> Dict:
        """Set value at nested dict path, creating intermediate dicts as needed.
        
        :param keys: List of keys defining nested path
        :param value: Value to set
        :param data: Dict to modify (default: load from storage)
        :returns: Modified dict (not automatically saved)
        """
        if data is None:
            data = self._load_json()
        
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        return data
    
    def exists(self) -> bool:
        """Check if storage file exists.
        
        :returns: True if storage file exists on disk
        """
        return self.storage_path.exists()
    
    def clear(self) -> None:
        """Delete storage file (use with caution).
        
        Removes the stage's storage file from disk. Cannot be undone.
        """
        if self.storage_path.exists():
            self.storage_path.unlink()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stage={self.stage_number}, path={self.storage_path})"
