from __future__ import annotations
import os
import re
import time
from typing import Optional

"""
Tool: read_file
Description: Read a text file from the repository. Use this to understand code before editing.
Args: {"path": "relative/path"}
Returns: file contents as string.

Tool: write_file
Description: Write/overwrite a text file in the repository. Use sparingly; prefer apply_patch for diffs.
Args: {"path": "relative/path", "content": "string"}

Tool: list_dir
Description: List files and directories. Supports shallow or recursive listing.
Args: {"path": "."|"subdir", "recursive": true|false}

Tool: search_text
Description: Regex search across repository with optional filters and modes.
Args: {
  "pattern": "regex", "path": ".",
  "glob": null,          # e.g., "**/*.py"
  "file_type": null,     # e.g., "py"
  "mode": "content",     # one of: "content" | "files_with_matches" | "count"
  "multiline": false
}
"""

# Track recently-read files so overwrites require prior read (policy: read-before-write)
_RECENT_READS: dict[str, float] = {}
_READ_AGE_SEC = 1800  # 30 min

def _mark_read(abs_path: str) -> None:
    _RECENT_READS[abs_path] = time.time()

def _recently_read(abs_path: str) -> bool:
    t = _RECENT_READS.get(abs_path)
    return t is not None and (time.time() - t) <= _READ_AGE_SEC

def _abs(repo: str, path: str) -> str:
    ap = os.path.abspath(os.path.join(repo, path))
    if not ap.startswith(os.path.abspath(repo)):
        raise ValueError("Path escape not allowed")
    return ap


def read_file(repo: str, path: str) -> str:
    ap = _abs(repo, path)
    with open(ap, "r", encoding="utf-8") as f:
        data = f.read()
    _mark_read(ap)
    return data


def write_file(repo: str, path: str, content: str) -> str:
    abspath = _abs(repo, path)
    exists = os.path.exists(abspath)
    # Overwrite requires prior read (safety)
    if exists and not _recently_read(abspath):
        return "ERROR: write denied; file must be read immediately before overwrite"
    # Discourage unsolicited docs unless explicitly allowed
    allow_new_docs = os.environ.get("AGENT_ALLOW_NEW_DOCS") == "1"
    is_doc = path.lower().endswith(".md") or os.path.basename(path).lower().startswith("readme")
    if (not exists) and is_doc and not allow_new_docs:
        return "ERROR: creating docs is disabled unless AGENT_ALLOW_NEW_DOCS=1"
    # Optional: basic emoji check for docs unless allowed
    if is_doc and os.environ.get("AGENT_ALLOW_EMOJI") != "1":
        # crude: disallow any chars in common emoji planes
        if any(0x1F300 <= ord(ch) <= 0x1FAFF for ch in content):
            return "ERROR: emojis not allowed in docs unless AGENT_ALLOW_EMOJI=1"
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    with open(abspath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"WROTE: {path} ({len(content)} bytes)"


def list_dir(repo: str, path: str = ".", recursive: bool = False) -> str:
    base = _abs(repo, path)
    if not recursive:
        return "\n".join(sorted(os.listdir(base)))
    acc = []
    for root, dirs, files in os.walk(base):
        rel_root = os.path.relpath(root, repo)
        for d in sorted(dirs):
            acc.append(os.path.join(rel_root, d) + "/")
        for fn in sorted(files):
            acc.append(os.path.join(rel_root, fn))
    return "\n".join(acc)


def search_text(repo: str, pattern: str, path: str = ".", glob: Optional[str] = None,
                file_type: Optional[str] = None, mode: str = "content",
                multiline: bool = False) -> str:
    flags = re.MULTILINE | (re.DOTALL if multiline else 0)
    rx = re.compile(pattern, flags)
    base = _abs(repo, path)
    hits = []
    def _glob_match(rel: str, g: str) -> bool:
        # simple ** glob -> regex
        rgx = re.escape(g).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", ".")
        return re.fullmatch(rgx, rel) is not None
    for root, _, files in os.walk(base):
        for fn in files:
            if file_type:
                ext = os.path.splitext(fn)[1].lstrip(".")
                if file_type.lower() != ext.lower():
                    continue
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, repo)
            if glob and not _glob_match(rel, glob):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if mode == "files_with_matches":
                    if rx.search(text):
                        hits.append(rel)
                elif mode == "count":
                    hits.append(f"{rel}:{len(rx.findall(text))}")
                else:
                    for i, line in enumerate(text.splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{rel}:{i}:{line.strip()}")
            except Exception:
                continue
    return "\n".join(sorted(hits))