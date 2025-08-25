from __future__ import annotations
import os
import shutil
import subprocess
from typing import Optional

"""
Tool: run_tests
Description: Run the repository's test suite with sensible auto-detection (Python/JS/TS/Go/Java). Prefer targeted tests when a path/pattern is provided. Return exit code and condensed output.
Args: {
  "pattern": "optional test target like tests/test_foo.py::TestBar::test_baz or 'pkg/...'/npm script",
  "watch": false,
  "timeout": 1800
}
"""


def _has_file(repo: str, *names: str) -> bool:
    return any(os.path.exists(os.path.join(repo, n)) for n in names)


def _bin_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_tests(repo: str, pattern: Optional[str] = None, watch: bool = False, timeout: int = 1800) -> str:
    cmds: list[str] = []

    # Python: pytest or nose/unittest via pytest
    if _has_file(repo, "pyproject.toml", "pytest.ini", "tox.ini", "requirements.txt") and _bin_exists("pytest"):
        if watch and _bin_exists("pytest-watcher"):
            base = "pytest -q"
            cmd = f"ptw -- {base} {pattern or ''}".strip()
        else:
            cmd = f"pytest -q {pattern or ''}".strip()
        cmds.append(cmd)

    # Node: npm/pnpm/yarn
    if _has_file(repo, "package.json"):
        if _bin_exists("pnpm"):
            cmd = f"pnpm test{' ' + pattern if pattern else ''}"
        elif _bin_exists("yarn"):
            cmd = f"yarn test{' ' + pattern if pattern else ''}"
        else:
            cmd = f"npm test{' -- ' + pattern if pattern else ''}"
        cmds.append(cmd)

    # Go
    if _has_file(repo, "go.mod") and _bin_exists("go"):
        cmd = f"go test {'./...' if not pattern else pattern}"
        cmds.append(cmd)

    # Java (Maven/Gradle)
    if _has_file(repo, "pom.xml") and _bin_exists("mvn"):
        cmds.append("mvn -q -DskipTests=false test")
    if _has_file(repo, "build.gradle", "build.gradle.kts"):
        if _bin_exists("./gradlew"):
            cmds.append("./gradlew test")
        elif _bin_exists("gradle"):
            cmds.append("gradle test")

    if not cmds:
        return "ERROR: No recognizable test runner/toolchain found."

    # Execute the first viable command and fallback through list
    last_out = ""
    for c in cmds:
        try:
            p = subprocess.run(c, cwd=repo, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash")
            out = f"cmd: {c}\nexit={p.returncode}\n{p.stdout}{p.stderr}"
            if p.returncode == 0:
                return out
            last_out = out
        except Exception as e:
            last_out = f"cmd: {c}\nERROR: {type(e).__name__}: {e}"
    return last_out