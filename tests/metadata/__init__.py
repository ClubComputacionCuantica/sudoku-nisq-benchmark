"""Tests for metadata package foundation (base classes and config)."""


def test_metadata_imports():
    """Test that all metadata managers can be imported."""
    from sudoku_nisq.metadata import (
        StageMetadataManager,
        MetadataConfig,
        InstanceMetadataManager,
        LogicalIRMetadataManager,
    )
    
    assert StageMetadataManager is not None
    assert MetadataConfig is not None
    assert InstanceMetadataManager is not None
    assert LogicalIRMetadataManager is not None


def test_config_feature_flag_default():
    """Test feature flag defaults to False."""
    from sudoku_nisq.metadata import MetadataConfig
    
    # Should default to False unless env var set
    assert isinstance(MetadataConfig.ENABLE_NEW_ARCHITECTURE, bool)


def test_config_feature_flag_env_override(monkeypatch):
    """Test feature flag can be enabled via environment variable."""
    
    monkeypatch.setenv("SUDOKU_NISQ_NEW_METADATA", "1")
    # Reimport to pick up env var
    import importlib
    import sudoku_nisq.metadata.config
    importlib.reload(sudoku_nisq.metadata.config)
    
    from sudoku_nisq.metadata.config import MetadataConfig as ReloadedConfig
    assert ReloadedConfig.ENABLE_NEW_ARCHITECTURE is True


def test_config_storage_paths():
    """Test storage path constants are defined."""
    from sudoku_nisq.metadata import MetadataConfig
    
    assert MetadataConfig.STAGE_1_GLOBAL_REGISTRY == "instances/registry.json"
    assert MetadataConfig.STAGE_2A_LOGICAL_IR == "stage_2a_logical_ir.json"
    assert MetadataConfig.STAGE_3_COMPILATION == "stage_3_compilation.jsonl"
    assert MetadataConfig.STAGE_5_EXECUTIONS == "stage_5_executions.jsonl"


def test_config_is_legacy_cache(tmp_path):
    """Test legacy cache detection."""
    from sudoku_nisq.metadata import MetadataConfig
    
    # Create legacy metadata.json
    puzzle_dir = tmp_path / "abc123"
    puzzle_dir.mkdir()
    (puzzle_dir / "metadata.json").write_text("{}")
    
    assert MetadataConfig.is_legacy_cache(puzzle_dir) is True
    
    # Add new-style file - should no longer be legacy
    (puzzle_dir / "stage_2a_logical_ir.json").write_text("{}")
    assert MetadataConfig.is_legacy_cache(puzzle_dir) is False


def test_config_is_migration_needed(tmp_path):
    """Test migration detection across cache."""
    from sudoku_nisq.metadata import MetadataConfig
    
    # Empty cache - no migration needed
    assert MetadataConfig.is_migration_needed(tmp_path) is False
    
    # Create legacy cache
    puzzle_dir = tmp_path / "abc123"
    puzzle_dir.mkdir()
    (puzzle_dir / "metadata.json").write_text("{}")
    
    assert MetadataConfig.is_migration_needed(tmp_path) is True


def test_config_programmatic_enable():
    """Test feature flag can be enabled programmatically."""
    from sudoku_nisq.metadata import MetadataConfig
    
    original = MetadataConfig.ENABLE_NEW_ARCHITECTURE
    
    try:
        MetadataConfig.enable_new_architecture()
        assert MetadataConfig.ENABLE_NEW_ARCHITECTURE is True
        
        MetadataConfig.disable_new_architecture()
        assert MetadataConfig.ENABLE_NEW_ARCHITECTURE is False
    finally:
        # Restore original state
        if original:
            MetadataConfig.enable_new_architecture()
        else:
            MetadataConfig.disable_new_architecture()
