"""Stage 1: Instance Selection metadata manager.

Tracks puzzle generation, sampling distribution, and puzzle characteristics.
"""

from pathlib import Path
from typing import Dict, List, Union
from datetime import datetime, timezone
from sudoku_nisq.metadata.base import StageMetadataManager
from sudoku_nisq.metadata.config import MetadataConfig


class InstanceMetadataManager(StageMetadataManager):
    """Manages Stage 1: Instance Selection (𝓘, μ) metadata.
    
    Global registry tracking all generated puzzles with:
    - Puzzle hash (deterministic identifier)
    - Size and difficulty parameters
    - PRNG seed (for reproducibility)
    - Generation timestamp
    - Solution count (if known)
    - Puzzle complexity metrics
    
    Storage: .quantum_solver_cache/instances/registry.json
    """
    
    @property
    def stage_number(self) -> int:
        return 1
    
    def __init__(self, cache_base: Path):
        """Initialize instance metadata manager with global registry.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        """
        self.cache_base = Path(cache_base)
        self._storage_path = self.cache_base / MetadataConfig.STAGE_1_GLOBAL_REGISTRY
    
    @property
    def storage_path(self) -> Path:
        return self._storage_path
    
    def record(self, **kwargs) -> str:
        """Record puzzle instance metadata to global registry.
        
        :param puzzle_hash: Deterministic puzzle identifier (required)
        :param size: Board size (e.g., 4 for 2×2, 9 for 3×3)
        :param subgrid_size: Subgrid dimension
        :param num_missing_cells: Number of empty cells
        :param board: Full board state (list of tuples)
        :param open_tuples: Empty cell positions
        :param pre_tuples: Given/clue positions
        :param prng_seed: PRNG seed for reproducibility (optional, deferred to Phase 5.5)
        :param generation_timestamp: ISO timestamp (auto-generated if not provided)
        :param solution_count: Number of valid solutions (optional)
        :returns: puzzle_hash identifier
        
        Example::
        
            manager = InstanceMetadataManager(cache_base)
            puzzle_hash = manager.record(
                puzzle_hash="abc123def456",
                size=4,
                subgrid_size=2,
                num_missing_cells=2,
                board=[(0,0,1), (0,1,2), ...],
                open_tuples=[(0,2), (1,3)],
                pre_tuples=[(0,0), (0,1), ...],
                solution_count=1
            )
        """
        puzzle_hash = kwargs.get("puzzle_hash")
        if not puzzle_hash:
            raise ValueError("puzzle_hash is required for instance registration")
        
        # Load existing registry or create new
        registry = self._load_json() or {}
        
        # Check if puzzle already exists
        if puzzle_hash in registry:
            # Update timestamp but keep original metadata
            registry[puzzle_hash]["last_accessed"] = datetime.now(timezone.utc).isoformat()
            self._save_json(registry)
            return puzzle_hash
        
        # Build instance metadata
        # Helper to filter out Mock objects
        def filter_mock(value):
            """Return None if value is a Mock object, otherwise return value."""
            if value is None:
                return None
            # Check if it's a Mock by checking for _mock_name attribute
            if hasattr(value, '_mock_name'):
                return None
            return value
        
        instance_data = {
            "puzzle_hash": puzzle_hash,
            "size": filter_mock(kwargs.get("size")),
            "subgrid_size": filter_mock(kwargs.get("subgrid_size")),
            "num_missing_cells": filter_mock(kwargs.get("num_missing_cells")),
            "board": filter_mock(kwargs.get("board")),
            "open_tuples": filter_mock(kwargs.get("open_tuples")),
            "pre_tuples": filter_mock(kwargs.get("pre_tuples")),
            "generation_timestamp": kwargs.get(
                "generation_timestamp",
                datetime.now(timezone.utc).isoformat()
            ),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "solution_count": filter_mock(kwargs.get("solution_count")),
            # TODO: Add prng_seed support in Phase 5.5 for reproducibility
            "prng_seed": filter_mock(kwargs.get("prng_seed")),  # Currently unused
        }
        
        # Register instance
        registry[puzzle_hash] = instance_data
        self._save_json(registry)
        
        return puzzle_hash
    
    def query(self, **filters) -> Union[List[Dict], Dict, None]:
        """Query puzzle instances by attributes.
        
        :param puzzle_hash: Exact puzzle hash lookup (returns single dict)
        :param size: Filter by board size
        :param subgrid_size: Filter by subgrid dimension
        :param num_missing_cells: Filter by difficulty (empty cells)
        :param min_missing_cells: Minimum difficulty
        :param max_missing_cells: Maximum difficulty
        :param date_range: Tuple of (start_datetime, end_datetime) for generation_timestamp
        :returns: List of matching puzzle metadata dicts, or single dict if puzzle_hash provided
        
        Example::
        
            # Exact lookup
            puzzle = manager.query(puzzle_hash="abc123def456")
            
            # Filter by attributes
            puzzles_4x4 = manager.query(size=4, num_missing_cells=2)
            
            # Date range
            from datetime import datetime
            recent = manager.query(
                date_range=(datetime(2025, 1, 1), datetime(2025, 1, 31))
            )
        """
        registry = self._load_json() or {}
        
        if not registry:
            return [] if "puzzle_hash" not in filters else None
        
        # Exact lookup by puzzle_hash
        if "puzzle_hash" in filters:
            return registry.get(filters["puzzle_hash"])
        
        # Filter instances
        matches = []
        for puzzle_hash, instance_data in registry.items():
            if not self._matches_filters(instance_data, filters):
                continue
            matches.append(instance_data)
        
        return matches
    
    def _matches_filters(self, instance_data: Dict, filters: Dict) -> bool:
        """Check if instance matches filter criteria.
        
        :param instance_data: Instance metadata dict
        :param filters: Filter criteria
        :returns: True if matches all filters
        """
        # Size filter
        if "size" in filters and instance_data.get("size") != filters["size"]:
            return False
        
        # Subgrid size filter
        if "subgrid_size" in filters and instance_data.get("subgrid_size") != filters["subgrid_size"]:
            return False
        
        # Exact difficulty filter
        if "num_missing_cells" in filters:
            if instance_data.get("num_missing_cells") != filters["num_missing_cells"]:
                return False
        
        # Min/max difficulty filters
        num_missing = instance_data.get("num_missing_cells")
        if num_missing is not None:
            if "min_missing_cells" in filters and num_missing < filters["min_missing_cells"]:
                return False
            if "max_missing_cells" in filters and num_missing > filters["max_missing_cells"]:
                return False
        
        # Date range filter (ISO timestamp string comparison)
        if "date_range" in filters:
            start_dt, end_dt = filters["date_range"]
            generation_ts = instance_data.get("generation_timestamp")
            if generation_ts:
                # Convert datetime to ISO string for comparison
                start_iso = start_dt.isoformat() if hasattr(start_dt, 'isoformat') else start_dt
                end_iso = end_dt.isoformat() if hasattr(end_dt, 'isoformat') else end_dt
                if not (start_iso <= generation_ts <= end_iso):
                    return False
        
        return True
