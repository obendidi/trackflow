"""Print the CHANGELOG.md section for a release tag.

Usage:
    python scripts/release_notes.py v0.3.0             # release notes for v0.3.0
    python scripts/release_notes.py --previous v0.3.0  # tag of the release before it, if any

Exits with an error when the tag doesn't match the package version in pyproject.toml and
src/trackflow/__init__.py, or when CHANGELOG.md has no notes for it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")
LINK_REFERENCE = re.compile(r"^\[[^\]]+\]: \S+$")


def changelog_sections(text: str) -> list[tuple[str, str]]:
    """``(version, notes)`` for each ``## [version]`` section, newest first."""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            sections.append((match["version"], []))
        elif sections and not LINK_REFERENCE.match(line):
            sections[-1][1].append(line)
    return [(version, "\n".join(lines).strip()) for version, lines in sections]


def package_versions() -> dict[str, str]:
    files = {
        "pyproject.toml": r'(?m)^version = "([^"]+)"',
        "src/trackflow/__init__.py": r'(?m)^__version__ = "([^"]+)"',
    }
    found = {}
    for name, pattern in files.items():
        match = re.search(pattern, (ROOT / name).read_text(encoding="utf-8"))
        found[name] = match.group(1) if match else "missing"
    return found


def main(argv: list[str]) -> int:
    previous = "--previous" in argv
    args = [a for a in argv if a != "--previous"]
    if len(args) != 1 or not args[0].startswith("v"):
        print(__doc__, file=sys.stderr)
        return 2
    tag = args[0]
    version = tag[1:]

    mismatched = {f: v for f, v in package_versions().items() if v != version}
    if mismatched:
        print(f"Tag {tag} doesn't match the package version: {mismatched}", file=sys.stderr)
        return 1

    released = [
        (v, notes)
        for v, notes in changelog_sections((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
        if v != "Unreleased"
    ]
    versions = [v for v, _ in released]
    if version not in versions:
        print(f"CHANGELOG.md has no section for {version}.", file=sys.stderr)
        return 1
    index = versions.index(version)
    if previous:
        if index + 1 < len(versions):
            print(f"v{versions[index + 1]}")
        return 0
    notes = released[index][1]
    if not notes:
        print(f"The CHANGELOG.md section for {version} is empty.", file=sys.stderr)
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
