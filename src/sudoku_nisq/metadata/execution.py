"""Stage 5: Execution metadata manager.

Tracks runtime data with hardware calibration snapshots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import uuid
from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig


class ExecutionMetadataManager(StageMetadataManager):
    """Manages Stage 5: Execution (Runtime) metadata.
    
    Append-only log tracking every execution:
    - Raw measurement counts
    - Shot count
    - Execution and queue times
    - Hardware calibration snapshot (T1/T2, error rates)
    - Link to Stage 3 (compilation_id)
    - Job ID
    
    Storage: {puzzle_hash}/stage_5_executions.jsonl (append-only)
    Returns: run_id (UUID) for linking to metrics (Stage 6-7)
    """
    
    @property
    def stage_number(self) -> int:
        return 5
    
    def __init__(self, cache_base: Path, puzzle_hash: str):
        """Initialize execution metadata manager.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        :param puzzle_hash: Puzzle identifier (determines storage subdirectory)
        """
        self.cache_base = Path(cache_base)
        self.puzzle_hash = puzzle_hash
        self._storage_path = self.cache_base / puzzle_hash / MetadataConfig.STAGE_5_EXECUTIONS
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def record(  # type: ignore[override]
        self,
        compilation_id: str,
        backend_name: str,
        counts: Dict[str, int],
        shots: int,
        execution_time_ms: float,
        hardware_snapshot: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        circuit_metrics: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        run_id: Optional[str] = None
    ) -> str:
        """Record execution data with hardware snapshot.
        
        :param compilation_id: Stage 3 compilation ID (required for provenance chain)
        :param backend_name: Name of backend that executed the circuit
        :param counts: Measurement outcome distribution {bitstring: count}
        :param shots: Total number of shots executed
        :param execution_time_ms: Execution time in milliseconds
        :param hardware_snapshot: Hardware calibration data collected at execution
        :param job_id: Provider-specific job identifier (optional)
        :param circuit_metrics: Circuit characteristics (n_qubits, depth, gates)
        :param timestamp: Execution timestamp (auto-generated if None)
        :param run_id: Execution run ID (auto-generated UUID if None)
        :returns: run_id (UUID for linking to metrics)
        :raises ValueError: If compilation_id is missing or empty
        """
        if not compilation_id:
            raise ValueError("compilation_id is required for Stage 5 recording")
        
        # Generate run_id if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        # Use current timestamp if not provided (UTC, timezone-aware)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Build metadata record
        metadata = {
            "run_id": run_id,
            "compilation_id": compilation_id,
            "backend_name": backend_name,
            "timestamp": timestamp.isoformat(),
            "shots": shots,
            "execution_time_ms": execution_time_ms,
            "counts": counts,
        }
        
        # Add optional fields if provided
        if hardware_snapshot is not None:
            metadata["hardware_snapshot"] = hardware_snapshot
        if job_id is not None:
            metadata["job_id"] = job_id
        if circuit_metrics is not None:
            metadata["circuit_metrics"] = circuit_metrics
        
        # Append to JSONL file
        self._append_jsonl(metadata)
        
        return run_id
    
    def query(  # type: ignore[override]
        self,
        run_id: Optional[str] = None,
        compilation_id: Optional[str] = None,
        backend_name: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query execution records by compilation/run/backend.
        
        :param run_id: Filter by specific run ID
        :param compilation_id: Filter by Stage 3 compilation ID
        :param backend_name: Filter by backend name
        :param date_range: Filter by timestamp range (start_datetime, end_datetime)
        :param limit: Maximum number of records to return (most recent first)
        :returns: List of matching execution records
        """
        records = self._load_jsonl()
        
        # Apply filters
        if run_id is not None:
            records = [r for r in records if r.get("run_id") == run_id]
        
        if compilation_id is not None:
            records = [r for r in records if r.get("compilation_id") == compilation_id]
        
        if backend_name is not None:
            records = [r for r in records if r.get("backend_name") == backend_name]
        
        if date_range is not None:
            start, end = date_range
            records = [
                r for r in records
                if start <= datetime.fromisoformat(r["timestamp"]) <= end
            ]
        
        # Apply limit (most recent first)
        if limit is not None:
            records = records[:limit]
        
        return records

