"""Configuration for stage-specific metadata management system."""

import os
from pathlib import Path
from typing import Optional


class MetadataConfig:
    """Configuration settings for metadata management.
    
    Controls feature flags, storage paths, and migration behavior.
    """
    
    # Feature flag: Enable new stage-specific architecture (now always enabled)
    ENABLE_NEW_ARCHITECTURE: bool = True
    
    # Storage path conventions
    STAGE_1_GLOBAL_REGISTRY = "instances/registry.json"
    STAGE_1_INSTANCE_LINK = "stage_1_instance_link.json"
    STAGE_2A_LOGICAL_IR = "stage_2a_logical_ir.json"
    STAGE_2B_IR_POLICY = "stage_2b_ir_policy.json"
    STAGE_3_COMPILATION = "stage_3_compilation.jsonl"
    STAGE_4_EXECUTABLE = "stage_4_executable.json"
    STAGE_5_EXECUTIONS = "stage_5_executions.jsonl"
    STAGE_6_7_METRICS = "stage_6_7_metrics.json"
    
    @classmethod
    def _check_stage_files_present(cls, puzzle_cache_dir: Path) -> bool:
        """Check if any new-style stage files are present.
        
        :param puzzle_cache_dir: Path to puzzle cache directory (contains puzzle_hash)
        :returns: True if any stage files exist
        """
        # Check for any stage files
        new_style_files = [
            cls.STAGE_2A_LOGICAL_IR,
            cls.STAGE_3_COMPILATION,
            cls.STAGE_5_EXECUTIONS,
        ]
        has_new_style = any((puzzle_cache_dir / f).exists() for f in new_style_files)
        
        return has_legacy and not has_new_style
    
    @classmethod
    def is_migration_needed(cls, cache_base: Path) -> bool:
        """Check if any puzzle caches need migration from legacy format.
        
        :param cache_base: Base cache directory (e.g., .quantum_solver_cache)
        :returns: True if any legacy caches found
        """
        if not cache_base.exists():
            return False
        
        for puzzle_dir in cache_base.iterdir():
            if puzzle_dir.is_dir() and puzzle_dir.name != "instances":
                if cls.is_legacy_cache(puzzle_dir):
                    return True
        
        return False
    
    @classmethod
    def get_cache_base(cls, custom_path: Optional[Path] = None) -> Path:
        """Get cache base directory with environment variable override.
        
        :param custom_path: Optional custom cache path
        :returns: Path to cache base directory
        """
        if custom_path:
            return Path(custom_path)
        
        env_cache = os.environ.get("SUDOKU_NISQ_CACHE_DIR")
        if env_cache:
            return Path(env_cache)
        
        return Path(".quantum_solver_cache")
    
    @classmethod
    def enable_new_architecture(cls) -> None:
        """Programmatically enable new architecture (for testing/migration)."""
        cls.ENABLE_NEW_ARCHITECTURE = True
    
    @classmethod
    def disable_new_architecture(cls) -> None:
        """Programmatically disable new architecture (for testing)."""
        cls.ENABLE_NEW_ARCHITECTURE = False
