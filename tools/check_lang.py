"""Fail if any tracked source or doc file contains Cyrillic.

This repository is English-only by policy. The check is mechanical because on a
previous project "we'll keep it English" as a habit did not survive contact
with reality; a regex does.
"""

import re
import sys
from pathlib import Path

# U+0400..U+04FF, written as escapes so this file passes its own check
CYRILLIC = re.compile("[\u0400-\u04FF]")
EXTS = {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".jsonl", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "results"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).parent.parent
    offenders = []
    for path in root.rglob("*"):
        if path.is_dir() or path.suffix not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if CYRILLIC.search(line):
                offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()[:80]}")
    if offenders:
        print("Cyrillic found in an English-only repo:")
        for o in offenders[:20]:
            print(" ", o)
        return 1
    print("check_lang: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
