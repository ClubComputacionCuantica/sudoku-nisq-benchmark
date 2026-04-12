"""Stage 4: Executable metadata manager.

Minimal tracking of job IDs linking compilations to provider-controlled pulse schedules.
"""

from pathlib import Path
from typing import Dict, List, Union
from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig


class ExecutableMetadataManager(StageMetadataManager):
    """Manages Stage 4: Executable (Pulse-level) metadata.
    
    Minimal implementation: Most providers (IBM, AWS, Quantinuum) handle pulse-level
    compilation server-side. This manager only tracks job_id links.
    
    Storage: {puzzle_hash}/stage_4_executable.json
    Structure: {compilation_id: job_id}
    
    Note: Pulse-level schedules are provider-controlled and typically not accessible.
    This stage primarily serves as a link between Stage 3 (compilation) and Stage 5 (execution).
    """
    
    @property
    def stage_number(self) -> int:
        return 4
    
    def __init__(self, cache_base: Path, puzzle_hash: str):
        """Initialize executable metadata manager.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        :param puzzle_hash: Puzzle identifier (determines storage subdirectory)
        """
        self.cache_base = Path(cache_base)
        self.puzzle_hash = puzzle_hash
        self._storage_path = self.cache_base / puzzle_hash / MetadataConfig.STAGE_4_EXECUTABLE
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def record(self, **kwargs) -> str:
        """Record job_id link for compilation.
        
        :param kwargs: Link parameters (compilation_id, job_id)
        :returns: job_id
        
        Placeholder: Minimal implementation in Phase 3 (alongside Stage 5)
        """
        raise NotImplementedError("ExecutableMetadataManager.record() - Phase 3 implementation pending")
    
    def query(self, **filters) -> Union[List[Dict], Dict, None]:
        """Query job_id by compilation_id.
        
        :param filters: Filter criteria (compilation_id)
        :returns: job_id string or None
        
        Placeholder: Minimal implementation in Phase 3
        """
        raise NotImplementedError("ExecutableMetadataManager.query() - Phase 3 implementation pending")
