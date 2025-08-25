from __future__ import annotations
import os

"""
Tool: repo_tree
Description: Print a directory tree with depth control and simple ignore patterns.
Args: {"path": ".", "max_depth": 3, "ignore": [".git", "node_modules", ".venv"]}
"""

def repo_tree(repo: str, path: str = ".", max_depth: int = 3, ignore: list[str] | None = None) -> str:
    ignore = set(ignore or [".git", "node_modules", ".venv", "__pycache__"])
    root = os.path.abspath(os.path.join(repo, path))
    base = os.path.abspath(repo)
    if not root.startswith(base):
        raise ValueError("Path escape not allowed")

    lines: list[str] = []
    for current_root, dirs, files in os.walk(root):
        rel = os.path.relpath(current_root, base)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue
        # filter ignores in-place for traversal
        dirs[:] = [d for d in sorted(dirs) if d not in ignore]
        files = [f for f in sorted(files) if f not in ignore]
        indent = "  " * (depth - 1) if depth else ""
        prefix = "" if rel == "." else rel + "/"
        if rel != ".":
            lines.append(f"{indent}{os.path.basename(rel)}/")
        for d in dirs:
            lines.append(f"{indent}  {d}/")
        for f in files:
            lines.append(f"{indent}  {f}")
    return "\n".join(lines)