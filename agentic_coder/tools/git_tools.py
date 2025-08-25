from __future__ import annotations
import subprocess

def _run(repo: str, args: list[str]) -> str:
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return p.stdout + p.stderr

"""
Tool: git_status
Description: Show status of working tree.
Args: {}

Tool: git_diff
Description: Show diff of working tree. Use this to generate context for commits.
Args: {"staged": false}

Tool: git_commit
Description: Commit staged or all changes with a message. Keep commits small and meaningful.
Args: {"message": "msg", "all": true|false}
"""

def git_status(repo: str) -> str:
    return _run(repo, ["status", "-sb"])


def git_diff(repo: str, staged: bool = False) -> str:
    return _run(repo, ["diff", "--staged" if staged else "--"])


def git_commit(repo: str, message: str, all: bool = False) -> str:
    if all:
        _run(repo, ["add", "-A"])  # stage all
    return _run(repo, ["commit", "-m", message])