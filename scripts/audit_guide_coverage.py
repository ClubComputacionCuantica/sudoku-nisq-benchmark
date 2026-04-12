"""Audit which `sudoku_nisq` modules/symbols are mentioned in `docs/guide`.

Scope: ONLY `docs/guide/**/*.md` are treated as documentation coverage.

This script scans Python modules under `src/sudoku_nisq`, extracts public top-level
symbols (classes, functions, and module-level assigned names), and checks whether
those names (and the module itself) are mentioned anywhere in the guide.

It produces a markdown report at: `docs/guide/guide_coverage_audit.md`.

Usage:
    D:/VSCode/sudoku-nisq-benchmark/.venv/Scripts/python.exe scripts/audit_guide_coverage.py
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WORD_BOUNDARY = r"(?<![A-Za-z0-9_]){token}(?![A-Za-z0-9_])"


@dataclass(frozen=True)
class ModuleScan:
    rel_path: str  # e.g. sudoku_nisq/backends.py
    module: str  # e.g. sudoku_nisq.backends
    basename: str  # e.g. backends
    doc_summary: str
    public_symbols: tuple[str, ...]


def _iter_python_files(pkg_root: Path) -> Iterable[Path]:
    for path in pkg_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _module_name_from_path(src_root: Path, file_path: Path) -> tuple[str, str]:
    rel = file_path.relative_to(src_root)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    module = ".".join(parts)
    rel_posix = rel.as_posix()
    return module, rel_posix


def _first_docstring_line(tree: ast.AST) -> str:
    doc = ast.get_docstring(tree)
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _public_assigned_names(node: ast.AST) -> Iterable[str]:
    names: list[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            names.append(node.target.id)

    for name in names:
        if not name.startswith("_"):
            yield name


def _extract_public_symbols(py_file: Path) -> tuple[str, tuple[str, ...]]:
    text = py_file.read_text(encoding="utf-8")
    tree = ast.parse(text)
    summary = _first_docstring_line(tree)

    symbols: set[str] = set()
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            symbols.update(_public_assigned_names(node))

    return summary, tuple(sorted(symbols))


def _load_guide_text(guide_root: Path) -> tuple[str, list[Path]]:
    md_files = sorted(guide_root.rglob("*.md"))
    combined = "\n".join(p.read_text(encoding="utf-8") for p in md_files)
    return combined, md_files


def _contains_word(text: str, token: str) -> bool:
    pattern = WORD_BOUNDARY.format(token=re.escape(token))
    return re.search(pattern, text) is not None


def _module_is_mentioned(guide_text: str, scan: ModuleScan) -> bool:
    # Strict requirement: look for module basename, but also allow common ways users refer to it.
    # Basename should match word boundary; dotted/paths can be substring.
    if _contains_word(guide_text, scan.basename):
        return True
    if scan.rel_path in guide_text:
        return True
    if scan.module in guide_text:
        return True
    if f"{scan.basename}.py" in guide_text:
        return True
    return False


def _symbol_is_mentioned(guide_text: str, symbol: str) -> bool:
    return _contains_word(guide_text, symbol)


def _format_module_header(scan: ModuleScan) -> str:
    summary = scan.doc_summary or "(no module docstring)"
    return f"- **{scan.module}** ({scan.rel_path}) — {summary}"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    pkg_root = src_root / "sudoku_nisq"
    guide_root = repo_root / "docs" / "guide"

    if not pkg_root.exists():
        raise RuntimeError(f"Package root not found: {pkg_root}")
    if not guide_root.exists():
        raise RuntimeError(f"Guide root not found: {guide_root}")

    guide_text, guide_files = _load_guide_text(guide_root)

    scans: list[ModuleScan] = []
    for py_file in sorted(_iter_python_files(pkg_root)):
        module, rel_posix = _module_name_from_path(src_root, py_file)
        summary, public_symbols = _extract_public_symbols(py_file)
        scans.append(
            ModuleScan(
                rel_path=rel_posix,
                module=module,
                basename=py_file.stem,
                doc_summary=summary,
                public_symbols=public_symbols,
            )
        )

    uncovered_modules: list[ModuleScan] = []
    partial_modules: list[tuple[ModuleScan, list[str]]] = []
    full_modules: list[ModuleScan] = []

    for scan in scans:
        module_mentioned = _module_is_mentioned(guide_text, scan)
        symbol_coverage = {s: _symbol_is_mentioned(guide_text, s) for s in scan.public_symbols}

        if not module_mentioned and (not symbol_coverage or not any(symbol_coverage.values())):
            uncovered_modules.append(scan)
            continue

        missing = [s for s, ok in symbol_coverage.items() if not ok]
        if missing:
            partial_modules.append((scan, missing))
        else:
            full_modules.append(scan)

    # Doc mismatch checks (lightweight, guide-only scope): find guide mentions of symbols
    # that do not exist as public top-level symbols in any module.
    public_symbol_universe: set[str] = set()
    for scan in scans:
        public_symbol_universe.update(scan.public_symbols)

    suspicious_guide_tokens: set[str] = set()
    # Heuristic: symbols in code font in markdown are likely identifiers.
    for match in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", guide_text):
        token = match.group(1)
        if token.startswith("_"):
            continue
        suspicious_guide_tokens.add(token)

    guide_identifiers_missing = sorted(t for t in suspicious_guide_tokens if t not in public_symbol_universe)

    out_path = guide_root / "guide_coverage_audit.md"

    lines: list[str] = []
    lines.append("# Guide Coverage Audit")
    lines.append("")
    lines.append("This file is autogenerated by `scripts/audit_guide_coverage.py`.")
    lines.append("Coverage is based ONLY on mentions in `docs/guide/**/*.md`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Guide files scanned: {len(guide_files)}")
    lines.append(f"- Modules scanned: {len(scans)}")
    lines.append(f"- Uncovered modules: {len(uncovered_modules)}")
    lines.append(f"- Partially covered modules: {len(partial_modules)}")
    lines.append(f"- Fully covered modules: {len(full_modules)}")
    lines.append("")

    lines.append("## Uncovered Modules")
    lines.append("")
    if not uncovered_modules:
        lines.append("(none)")
    else:
        for scan in uncovered_modules:
            lines.append(_format_module_header(scan))
            if scan.public_symbols:
                lines.append(f"  - Public symbols: {', '.join(scan.public_symbols)}")
                lines.append("  - Placeholder: add guide coverage for this module and its symbols.")
            else:
                lines.append("  - Public symbols: (none)")
                lines.append("  - Placeholder: decide whether this module should be documented or remain internal.")
    lines.append("")

    lines.append("## Partially Covered Modules (Missing Public Symbols)")
    lines.append("")
    if not partial_modules:
        lines.append("(none)")
    else:
        for scan, missing in sorted(partial_modules, key=lambda t: t[0].module):
            lines.append(_format_module_header(scan))
            lines.append(f"  - Missing symbols: {', '.join(missing)}")
            lines.append("  - Placeholder: mention these symbols in the guide (even briefly) or mark as internal.")
    lines.append("")

    lines.append("## Fully Covered Modules")
    lines.append("")
    if not full_modules:
        lines.append("(none)")
    else:
        for scan in sorted(full_modules, key=lambda s: s.module):
            lines.append(_format_module_header(scan))

    lines.append("")
    lines.append("## Guide Identifiers Not Found As Public Symbols")
    lines.append("")
    lines.append(
        "These are identifiers found in backticks in the guide that do not match any public top-level symbol extracted from `src/sudoku_nisq`. "
        "They may be private methods, typos, or references to external SDK APIs."
    )
    lines.append("")
    if not guide_identifiers_missing:
        lines.append("(none)")
    else:
        for token in guide_identifiers_missing:
            lines.append(f"- `{token}`")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
