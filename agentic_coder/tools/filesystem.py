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
    
    # Directories to exclude in recursive listings (huge, not useful for agents)
    EXCLUDE_DIRS = {
        '.git', '.svn', '.hg', '.bzr',  # Version control
        'node_modules', 'bower_components',  # JavaScript
        'venv', 'env', '.env', 'virtualenv', '.venv',  # Python virtual environments  
        '__pycache__', '.pytest_cache', '.tox',  # Python caches
        'target', 'build', 'dist', 'out',  # Build outputs
        '.gradle', '.mvn',  # Java/Gradle/Maven
        'vendor',  # Various languages (Go, PHP, etc)
        '.idea', '.vscode', '.vs',  # IDE directories
        'coverage', '.coverage',  # Test coverage
        '.terraform', '.serverless',  # Infrastructure
        'Pods',  # iOS/CocoaPods
        '.mypy_cache', '.ruff_cache',  # More Python caches
        'site-packages',  # Python packages
        '.asdf',  # Language version manager (massive)
        'Library',  # macOS system directory (massive)
        'go/pkg', 'go/src',  # Go packages/modules
        '.cache', '.config',  # User cache/config directories
        '.cursor', '.claude-docker',  # IDE and tool caches
        '.rye', '.conda', '.pyenv',  # Python version managers
    }
    
    if not recursive:
        # For non-recursive, still show the directories but don't enter them
        return "\n".join(sorted(os.listdir(base)))
    
    # Aggressive limits to prevent massive outputs
    MAX_TOTAL_LINES = 1000  # Never return more than 1000 lines total
    MAX_ENTRIES_PER_DIR = 50   # Reduced from 100
    MAX_FILES_PER_DIR = 20     # Reduced from 50
    
    # Smart directory truncation - sample size and truncate large ones
    def count_dir_contents(dir_path):
        """Count files and subdirs in a directory (non-recursive)."""
        try:
            entries = os.listdir(dir_path)
            return len(entries)
        except:
            return 0
    
    acc = []
    truncated_dirs = []
    line_count = 0
    
    for root, dirs, files in os.walk(base):
        # Hard limit - stop if we're approaching max lines
        if line_count >= MAX_TOTAL_LINES - 10:
            acc.append(f"[TRUNCATED - output limit reached, showing first {line_count} entries]")
            break
            
        rel_root = os.path.relpath(root, repo)
        
        # Check if this directory should be excluded entirely
        dirs_to_process = []
        for d in dirs:
            if line_count >= MAX_TOTAL_LINES - 10:
                break
                
            if d in EXCLUDE_DIRS:
                # Note that we excluded it
                if rel_root == ".":
                    acc.append(f"{d}/ [excluded - known large directory]")
                    line_count += 1
                continue
                
            dir_path = os.path.join(root, d)
            entry_count = count_dir_contents(dir_path)
            
            if entry_count > MAX_ENTRIES_PER_DIR:
                # This directory is huge - show it but don't recurse into it
                rel_dir = os.path.join(rel_root, d) if rel_root != "." else d
                acc.append(f"{rel_dir}/ [{entry_count:,} files]")
                line_count += 1
                truncated_dirs.append(dir_path)
            else:
                # Normal directory, will recurse into it
                dirs_to_process.append(d)
                rel_dir = os.path.join(rel_root, d) if rel_root != "." else d
                acc.append(f"{rel_dir}/")
                line_count += 1
        
        # Modify dirs in-place to control recursion
        dirs[:] = [d for d in dirs_to_process if os.path.join(root, d) not in truncated_dirs]
        
        # Add files from current directory
        file_count = 0
        skipped_count = 0
        for fn in sorted(files):
            if line_count >= MAX_TOTAL_LINES - 5:
                break
                
            # Skip common cache/compiled files (but not archives which might be important)
            if fn.endswith(('.pyc', '.pyo', '.so', '.dylib', '.dll', '.class')):
                skipped_count += 1
                continue
                
            file_count += 1
            if file_count <= MAX_FILES_PER_DIR:
                rel_file = os.path.join(rel_root, fn) if rel_root != "." else fn
                acc.append(rel_file)
                line_count += 1
            elif file_count == MAX_FILES_PER_DIR + 1:
                remaining = len(files) - MAX_FILES_PER_DIR - skipped_count
                if remaining > 0:
                    acc.append(f"[... and {remaining} more files in {rel_root}]")
                    line_count += 1
                break
    
    result = "\n".join(acc)
    
    # Final safety check - if result is still too large, truncate it
    MAX_CHARS = 50000  # 50KB max output
    if len(result) > MAX_CHARS:
        result = result[:MAX_CHARS] + f"\n[TRUNCATED - output was {len(result):,} chars, showing first {MAX_CHARS:,}]"
    
    return result


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
    # Exclude common large directories from search
    EXCLUDE_DIRS = {'node_modules', 'venv', '.git', '__pycache__', 'dist', 'build'}
    
    for root, dirs, files in os.walk(base):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
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