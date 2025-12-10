# Internal Documentation (Not Published)

This folder stores internal documentation that:
- Is useful for maintainers and contributors (e.g., TODO trackers, best practices)
- Should not be published on the public documentation site

Publication control:
- Sphinx excludes this folder via `exclude_patterns = ['internal/**']` in `docs/conf.py`

Current internal files:
- `DOCS_TODO.md`: Documentation roadmap and tracking
- `DOCUMENTATION_BEST_PRACTICES.md`: Guide on documenting code and features
- `architecture/`: Detailed architecture and implementation plans for the metrics system
  - `README.md`: Navigation guide for metrics architecture docs
  - `metrics_system_design.md`: Complete technical specification (~900 lines)
  - `metrics_implementation_roadmap.md`: 8-week implementation plan (~600 lines)
  - `metrics_implementation_checklist.md`: Detailed task checklist (~500 lines)
  - `metrics_architecture_summary.md`: Executive summary (~400 lines)
- `metrics_quick_start.md`: Developer-focused quick start (moved from guide)
- `metrics_quick_reference.md`: Quick reference card (moved from guide)

Guidelines:
- Keep user-facing docs under `docs/guide/`
- Keep developer-facing, non-public notes under `docs/internal/`
- Do not reference files here from any published guide or toctree
