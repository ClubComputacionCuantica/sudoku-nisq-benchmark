# Internal Documentation (Not Published)

This folder stores internal documentation that:
- Is useful for maintainers and contributors (e.g., TODO trackers, best practices)
- Should not be published on the public documentation site

Publication control:
- Sphinx excludes this folder via `exclude_patterns = ['internal/**']` in `docs/conf.py`

Current internal files:
- `DOCS_TODO.md`: Documentation roadmap and tracking
- `DOCUMENTATION_BEST_PRACTICES.md`: Guide on documenting code and features

Guidelines:
- Keep user-facing docs under `docs/guide/`
- Keep developer-facing, non-public notes under `docs/internal/`
- Do not reference files here from any published guide or toctree
