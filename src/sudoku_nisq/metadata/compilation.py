"""Stage 3: Compilation metadata manager.

Tracks transpilation provenance with routing metadata and gate transformations.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig


class CompilationMetadataManager(StageMetadataManager):
    """Manages Stage 3: Compilation (𝖢) provenance metadata.
    
    Append-only log tracking every transpilation:
    - Pre/post gate counts and depth
    - Routing metadata (SWAP counts, initial/final layouts)
    - Optimization level
    - Transpiler seed (for deterministic compilation)
    - Compilation timestamp
    - Link to Stage 2a (circuit_hash)
    
    Storage: {puzzle_hash}/stage_3_compilation.jsonl (append-only)
    Returns: compilation_id (UUID) for linking to Stage 5
    """
    
    @property
    def stage_number(self) -> int:
        return 3
    
    def __init__(self, cache_base: Path, puzzle_hash: str):
        """Initialize compilation metadata manager.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        :param puzzle_hash: Puzzle identifier (determines storage subdirectory)
        """
        self.cache_base = Path(cache_base)
        self.puzzle_hash = puzzle_hash
        self._storage_path = self.cache_base / puzzle_hash / MetadataConfig.STAGE_3_COMPILATION
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def record(  # type: ignore[override]
            self, 
               circuit_hash: str,
               backend_alias: str,
               opt_level: int,
               resources: Dict[str, Any],
               routing: Optional[Dict[str, Any]] = None,
               **kwargs) -> str:
        """Record compilation provenance with auto-extracted routing metadata.
        
        Args:
            circuit_hash: Hash from Stage 2a (logical IR circuit)
            backend_alias: Target backend identifier
            opt_level: Optimization level (0-3)
            resources: Dict with n_qubits, n_gates, depth, gate_counts, etc.
            routing: Optional routing metadata (initial_layout, final_layout, swap_count)
            **kwargs: Additional metadata (sdk_type, etc.)
            
        Returns:
            compilation_id: UUID for linking to Stage 5 executions
        """
        # Generate unique compilation ID
        compilation_id = str(uuid.uuid4())
        
        # Build compilation record
        record = {
            "compilation_id": compilation_id,
            "circuit_hash": circuit_hash,
            "backend_alias": backend_alias,
            "opt_level": opt_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resources": resources,
        }
        
        # Add routing metadata if available
        if routing:
            record["routing"] = routing
        
        # Add additional fields
        for key, value in kwargs.items():
            if key not in record:
                record[key] = value
        
        # Append to JSONL file (O(1) write)
        self._append_jsonl(record)
        
        return compilation_id
    
    def query(  # type: ignore[override]
              self, 
              backend_alias: Optional[str] = None,
              opt_level: Optional[int] = None,
              circuit_hash: Optional[str] = None,
              compilation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query compilation records by backend/opt_level/circuit.
        
        Args:
            backend_alias: Filter by backend alias
            opt_level: Filter by optimization level
            circuit_hash: Filter by Stage 2a circuit hash
            compilation_id: Filter by specific compilation ID
            
        Returns:
            List of matching compilation records (empty if no matches)
        """
        if not self.storage_path.exists():
            return []
        
        # Lazy load JSONL and filter
        results = []
        for record in self._load_jsonl():
            # Apply filters
            if backend_alias and record.get("backend_alias") != backend_alias:
                continue
            if opt_level is not None and record.get("opt_level") != opt_level:
                continue
            if circuit_hash and record.get("circuit_hash") != circuit_hash:
                continue
            if compilation_id and record.get("compilation_id") != compilation_id:
                continue
            
            results.append(record)
        
        return results
