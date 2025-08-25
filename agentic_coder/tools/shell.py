from __future__ import annotations
import subprocess

"""
Tool: run
Description: Run shell commands inside the repository. Use for builds, tests, or quick scripts. Defaults to bash.
Args: {"cmd": "string", "timeout": 120}
Returns: combined stdout/stderr and exit code.
"""

def run(repo: str, cmd: str, timeout: int = 120) -> str:
    lowered = f" {cmd.strip().lower()} "
    banned = [" grep ", " rg ", " rm -rf ", " git push -f ", " :(){ ", " dd if=", " mkfs", " shred ", " srm "]
    if any(b in lowered for b in banned):
        return "ERROR: command blocked by policy"
    proc = subprocess.run(
        cmd,
        cwd=repo,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        executable="/bin/bash",
    )
    return f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}"