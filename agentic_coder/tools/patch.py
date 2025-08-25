from __future__ import annotations
import os
from unidiff import PatchSet

"""
Tool: apply_patch
Description: Apply a unified diff (create/modify/delete files). Use exact file paths rooted at repo. Validate before writing.
Args: {"diff": "
*** unified diff ***
"}
Returns: a summary of changed files and hunks; errors if malformed.
"""

def apply_patch(repo: str, diff: str) -> str:
    patch = PatchSet(diff.splitlines(True))
    changed = []
    for p in patch:
        target_rel = p.path or p.target_file or p.source_file
        target_rel = target_rel.replace("a/", "", 1).replace("b/", "", 1)
        target_abs = os.path.abspath(os.path.join(repo, target_rel))
        if not target_abs.startswith(os.path.abspath(repo)):
            raise ValueError("Path escape not allowed in patch")

        if p.is_removed_file:
            if os.path.exists(target_abs):
                os.remove(target_abs)
                changed.append(f"DELETE {target_rel}")
            continue

        os.makedirs(os.path.dirname(target_abs), exist_ok=True)

        original = []
        if os.path.exists(target_abs):
            with open(target_abs, "r", encoding="utf-8", errors="ignore") as f:
                original = f.read().splitlines(True)

        new_lines = []
        idx = 0
        for h in p:
            src_pos = h.source_start - 1
            new_lines.extend(original[idx:src_pos])
            idx = src_pos
            for l in h:
                if l.is_added:
                    new_lines.append(l.value)
                elif l.is_removed:
                    idx += 1
                else:
                    new_lines.append(original[idx])
                    idx += 1
        new_lines.extend(original[idx:])

        with open(target_abs, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        changed.append(f"WRITE {target_rel} (+{sum(1 for _ in new_lines)})")

    return "\n".join(changed) if changed else "NOOP"