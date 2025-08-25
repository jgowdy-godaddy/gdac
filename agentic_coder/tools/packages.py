from __future__ import annotations
import os
import subprocess
import shutil

"""
Tool: install_deps
Description: Install project dependencies by detecting the package manager. Python: uv/pip/poetry/pip-tools; Node: pnpm/yarn/npm. Runs in the repo root.
Args: {"dev": false}
"""


def install_deps(repo: str, dev: bool = False) -> str:
    def run(cmd: str) -> str:
        p = subprocess.run(cmd, cwd=repo, shell=True, capture_output=True, text=True, executable="/bin/bash")
        return f"cmd: {cmd}\nexit={p.returncode}\n{p.stdout}{p.stderr}"

    # Python ecosystems
    if os.path.exists(os.path.join(repo, "pyproject.toml")):
        if shutil.which("uv"):
            return run("uv sync" + (" --dev" if dev else ""))
        if shutil.which("poetry"):
            return run("poetry install" + (" --with dev" if dev else ""))
    if os.path.exists(os.path.join(repo, "requirements.txt")) and shutil.which("pip"):
        return run("pip install -r requirements.txt")

    # Node ecosystems
    if os.path.exists(os.path.join(repo, "package.json")):
        if shutil.which("pnpm"):
            return run("pnpm install")
        if shutil.which("yarn"):
            return run("yarn install")
        return run("npm install")

    return "ERROR: No recognizable dependency manifest found."