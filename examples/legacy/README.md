# Legacy Examples

This directory is a placeholder for examples that demonstrate the **legacy MetadataManager API**, which is deprecated.

## Status: Preserved for Reference Only

Future legacy examples will be moved here for historical reference and to help users understand the migration path from the old API to the new BenchmarkSession-based architecture.

**⚠️ DO NOT USE THESE EXAMPLES FOR NEW CODE**

## Migration Guide

For new code, use the examples in the parent directory which demonstrate the modern BenchmarkSession API:

- `migrate_to_benchmark_session.py` - Complete migration guide with side-by-side comparison
- `phase4_metrics_integration.py` - Modern metrics workflow
- `exact_cover_benchmark.py` - Quantum benchmarking workflow

## Documentation

For detailed migration instructions, see:
- **User guide**: `docs/guide/upgrading_from_metadata_manager.md`
- **Architecture**: `docs/internal/architecture/stage_metadata_manager_migration_plan.md`

## Candidates for Migration

Currently, this directory serves as a placeholder. As we identify examples that heavily use the legacy MetadataManager API, they will be moved here with appropriate deprecation notices.

Examples that will eventually move here:
- Direct `puzzle._metadata` usage
- Direct calls to `MetadataManager.set_main_circuit_resources()`
- Direct calls to `MetadataManager.set_backend_resources()`
- Direct calls to `MetadataManager.get_solver_data()`
- Direct calls to `MetadataManager.get_resource_summary()`

## Migration Tool

Run the automated migration script:

```bash
# Preview changes
python scripts/migrate_metadata.py --dry-run

# Perform migration
python scripts/migrate_metadata.py
```

## Questions?

If you have questions about migration, please:
1. Read the migration guide: `docs/guide/upgrading_from_metadata_manager.md`
2. Check the example: `examples/migrate_to_benchmark_session.py`
3. File an issue on GitHub with the `migration` label
