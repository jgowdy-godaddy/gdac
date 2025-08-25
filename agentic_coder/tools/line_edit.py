from __future__ import annotations
import os
from typing import List, Dict, Any

"""
Tool: edit_lines
Purpose: Apply atomic, line-number–anchored edits in one file.
Args:
{
  "path": "relative/file",
  "edits": [
     {"line": 42, "delete": 0, "insert": ["new line 1","new line 2"]},
     {"line": 108, "delete": 2, "insert": ["replacement"]},
  ]
}
Notes:
- 1-based line indexing
- 'delete' is number of existing lines to remove starting at 'line'
"""

def _abs(repo: str, path: str) -> str:
    ap = os.path.abspath(os.path.join(repo, path))
    if not ap.startswith(os.path.abspath(repo)):
        raise ValueError("Path escape not allowed")
    return ap

def edit_lines(repo: str, path: str, edits: List[Dict[str, Any]]) -> str:
    p = _abs(repo, path)
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    # Apply edits bottom-up so line numbers remain valid
    changed = 0
    for e in sorted(edits, key=lambda x: int(x.get("line",1)), reverse=True):
        ln = int(e.get("line", 1)) - 1
        delete = max(0, int(e.get("delete", 0)))
        insert = e.get("insert", [])
        if ln < 0 or ln > len(lines):
            return {"status":"error","error":f"line out of range: {ln+1}"}
        pre = lines[:ln]
        post = lines[ln+delete:]
        if insert and isinstance(insert, list):
            pre.extend(insert)
        lines = pre + post
        changed += 1
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return {"changed": changed, "path": path}