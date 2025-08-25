from __future__ import annotations
import os
import shutil
import subprocess
from typing import Literal

"""
Tool: format_code
Description: Run language-appropriate formatters/linters with optional auto-fix. For Python: ruff/black/isort (if present). For JS/TS: eslint/prettier (if present). Executes only tools available on PATH; no installation is performed here.
Args: {
  "paths": ["."],
  "auto_fix": true|false,
  "timeout": 900
}
"""


def _bin(name: str) -> bool:
    return shutil.which(name) is not None


def format_code(repo: str, paths: list[str] | None = None, auto_fix: bool = True, timeout: int = 900) -> str:
    paths = paths or ["."]
    results: list[str] = []

    def run(cmd: str) -> str:
        p = subprocess.run(cmd, cwd=repo, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
        return f"cmd: {cmd}\nexit={p.returncode}\n{p.stdout}{p.stderr}"

    # Python formatters
    if _bin("ruff"):
        cmd = f"ruff {'--fix ' if auto_fix else ''}{' '.join(paths)}"
        results.append(run(cmd))
    if _bin("black"):
        cmd = f"black {' '.join(paths)}" if auto_fix else f"black --check {' '.join(paths)}"
        results.append(run(cmd))
    if _bin("isort"):
        cmd = f"isort {' '.join(paths)}" if auto_fix else f"isort --check-only {' '.join(paths)}"
        results.append(run(cmd))

    # JS/TS formatters
    if os.path.exists(os.path.join(repo, "package.json")):
        if _bin("eslint"):
            cmd = f"eslint {'--fix ' if auto_fix else ''}{' '.join(paths)}"
            results.append(run(cmd))
        if _bin("prettier"):
            cmd = f"prettier {'--write ' if auto_fix else '--check '}{' '.join(paths)}"
            results.append(run(cmd))

    return "\n\n".join(results) if results else "No formatters/linters found on PATH."