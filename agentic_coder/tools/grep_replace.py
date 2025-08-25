from __future__ import annotations
import os
import re

"""
Tool: replace_text
Description: Project-wide regex find/replace with optional dry-run preview. Safe by default; prevents path escape.
Args: {"pattern": "regex", "repl": "string", "path": ".", "dry_run": true}
"""

def replace_text(repo: str, pattern: str, repl: str, path: str = ".", dry_run: bool = True) -> str:
    rx = re.compile(pattern)
    base = os.path.abspath(repo)
    start = os.path.abspath(os.path.join(repo, path))
    if not start.startswith(base):
        raise ValueError("Path escape not allowed")

    changes: list[str] = []
    for root, _, files in os.walk(start):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                new = rx.sub(repl, text)
                if new != text:
                    rel = os.path.relpath(p, base)
                    changes.append(rel)
                    if not dry_run:
                        with open(p, "w", encoding="utf-8") as f:
                            f.write(new)
            except Exception:
                continue
    if not changes:
        return "No matches."
    header = "DRY-RUN: would change" if dry_run else "Changed"
    return header + "\n" + "\n".join(changes)