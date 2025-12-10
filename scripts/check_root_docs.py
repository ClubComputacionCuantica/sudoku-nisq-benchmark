import sys
from pathlib import Path

# Fail commit if any Markdown files are added/modified in repo root,
# except README.md. Intended to be used as a pre-commit hook.

ROOT = Path(__file__).resolve().parent.parent

ALLOWED = {"README.md"}

def is_root_markdown(path: Path) -> bool:
    # True if the file is in the repo root (no parent beyond ROOT) and is .md
    try:
        rel = path.resolve().relative_to(ROOT)
    except Exception:
        # File outside repo (shouldn't happen in pre-commit context)
        return False
    return (len(rel.parts) == 1) and rel.suffix.lower() == ".md"

def main(argv: list[str]) -> int:
    offending: list[Path] = []
    for arg in argv:
        p = Path(arg)
        if is_root_markdown(p) and p.name not in ALLOWED:
            offending.append(p)

    if offending:
        print("ERROR: Non-README Markdown files in repository root are not allowed:")
        for p in offending:
            print(f" - {p}")
        print("\nMove internal docs to docs/internal/ or user docs to docs/guide/.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
