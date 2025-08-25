# Repository: agentic-coder
# Layout
# ├─ pyproject.toml
# ├─ README.md
# ├─ prompts/AGENT.md
# └─ agentic_coder/
#    ├─ __init__.py
#    ├─ config.py
#    ├─ llm_registry.py
#    ├─ planner.py
#    ├─ runtime.py
#    ├─ tools/__init__.py
#    ├─ tools/filesystem.py
#    ├─ tools/patch.py
#    ├─ tools/shell.py
#    ├─ tools/git_tools.py
#    └─ cli.py

# -------------------------
# pyproject.toml
# -------------------------
[build-system]
requires = ["setuptools>=67", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agentic-coder"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
  "transformers>=4.44.0",
  "tokenizers>=0.19.0",
  "accelerate>=0.33.0",
  "huggingface_hub>=0.24.0",
  "datasets>=2.19.0",
  "rich>=13.7.0",
  "typer>=0.12.3",
  "unidiff>=0.7.5",
  "python-dotenv>=1.0.1",
  "httpx>=0.27.0",
  "openai>=1.40.0",
  "anthropic>=0.34.0",
]

[project.scripts]
agentic-coder = "agentic_coder.cli:app"

# -------------------------
# README.md
# -------------------------
# Agentic Coder

Claude Code–style coding agent using open-weight or remote LLMs.

```bash
pip install -e .
agentic-coder models
agentic-coder run --model qwen2.5-coder-14b --repo . --goal "Fix tests"
```

# -------------------------
# prompts/AGENT.md
# -------------------------
# Agent Rules (AGENT.md)

Follow THINK → PLAN → ACTION → OBSERVATION loop. Use small, verifiable patches.
Prefer `apply_patch` for edits. Verify with tests. Never invent files or APIs without reading first.

# -------------------------
# agentic_coder/__init__.py
# -------------------------
__all__ = []

# -------------------------
# agentic_coder/config.py
# -------------------------
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelSpec:
    id: str
    hf_revision: Optional[str] = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    chat_template: Optional[str] = None
    remote: bool = False
    api_type: Optional[str] = None

MODEL_PRESETS = {
    "qwen2.5-coder-14b": ModelSpec(id="Qwen/Qwen2.5-Coder-14B", dtype="bfloat16"),
    "qwen2.5-coder-32b": ModelSpec(id="Qwen/Qwen2.5-Coder-32B-Instruct", dtype="bfloat16"),
    "qwen3-coder-30b-a3b": ModelSpec(id="Qwen/Qwen3-Coder-30B-A3B-Instruct", dtype="bfloat16"),
    "deepseek-coder-v2-16b": ModelSpec(id="deepseek-ai/DeepSeek-Coder-V2-Instruct-0724", dtype="bfloat16"),
    "deepseek-coder-33b": ModelSpec(id="deepseek-ai/deepseek-coder-33b-instruct", dtype="bfloat16"),
    "llama-3.1-8b": ModelSpec(id="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16"),
    "remote-openai": ModelSpec(id="REMOTE", remote=True, api_type="openai"),
    "remote-anthropic": ModelSpec(id="REMOTE", remote=True, api_type="anthropic"),
}

# -------------------------
# agentic_coder/llm_registry.py
# -------------------------
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import ModelSpec, MODEL_PRESETS

HF_CACHE = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

def ensure_model_cached(spec: ModelSpec) -> str:
    return snapshot_download(
        repo_id=spec.id,
        revision=spec.hf_revision,
        cache_dir=HF_CACHE,
        resume_download=True,
    )

def load_model(preset: str):
    spec = MODEL_PRESETS[preset]
    if spec.remote:
        return None, None, spec
    local_path = ensure_model_cached(spec)
    tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=spec.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=spec.trust_remote_code, device_map="auto")
    return tok, model, spec

# -------------------------
# agentic_coder/planner.py
# -------------------------
SYSTEM_TEMPLATE = "<SYSTEM>Follow AGENT.md strictly.</SYSTEM>"
USER_TEMPLATE = "<GOAL>\n{goal}\n</GOAL>\n<CONTEXT>repo={repo}</CONTEXT>"

def make_prompt(goal: str, repo: str) -> str:
    return SYSTEM_TEMPLATE + "\n" + USER_TEMPLATE.format(goal=goal, repo=repo)

# -------------------------
# agentic_coder/runtime.py
# -------------------------
# (kept as in your current file; omitted here for brevity)

# -------------------------
# agentic_coder/tools/__init__.py
# -------------------------
from .filesystem import read_file, write_file, list_dir, search_text
from .patch import apply_patch
from .shell import run
from .git_tools import git_status, git_diff, git_commit

class ToolRegistry:
    def __init__(self, repo: str):
        self.repo = repo
        self._tools = {
            "read_file": lambda a: read_file(self.repo, **a),
            "write_file": lambda a: write_file(self.repo, **a),
            "list_dir": lambda a: list_dir(self.repo, **a),
            "search_text": lambda a: search_text(self.repo, **a),
            "apply_patch": lambda a: apply_patch(self.repo, **a),
            "run": lambda a: run(self.repo, **a),
            "git_status": lambda a: git_status(self.repo),
            "git_diff": lambda a: git_diff(self.repo, **a),
            "git_commit": lambda a: git_commit(self.repo, **a),
        }

    def dispatch(self, tool: str, args):
        return self._tools[tool](args)

    def goal_satisfied(self, goal: str) -> bool:
        return False

# -------------------------
# agentic_coder/tools/filesystem.py
# -------------------------
import os, re

def _abs(repo, path):
    ap = os.path.abspath(os.path.join(repo, path))
    if not ap.startswith(os.path.abspath(repo)):
        raise ValueError("path escape")
    return ap

def read_file(repo, path):
    with open(_abs(repo, path)) as f:
        return f.read()

def write_file(repo, path, content):
    ab = _abs(repo, path)
    os.makedirs(os.path.dirname(ab), exist_ok=True)
    with open(ab, "w") as f:
        f.write(content)
    return f"WROTE {path}"

def list_dir(repo, path=".", recursive=False):
    base = _abs(repo, path)
    if not recursive:
        return "\n".join(os.listdir(base))
    acc = []
    for r, _, fs in os.walk(base):
        for fn in fs:
            acc.append(os.path.relpath(os.path.join(r, fn), repo))
    return "\n".join(acc)

def search_text(repo, pattern, path="."):
    rx = re.compile(pattern)
    hits = []
    for r, _, fs in os.walk(_abs(repo, path)):
        for fn in fs:
            p = os.path.join(r, fn)
            try:
                with open(p) as f:
                    for i,l in enumerate(f,1):
                        if rx.search(l):
                            hits.append(f"{p}:{i}:{l.strip()}")
            except: pass
    return "\n".join(hits)

# -------------------------
# agentic_coder/tools/patch.py
# -------------------------
from unidiff import PatchSet
import os

def apply_patch(repo, diff):
    patch = PatchSet(diff.splitlines(True))
    changed = []
    for p in patch:
        target = p.path or p.target_file or p.source_file
        target = target.replace("a/", "", 1).replace("b/", "", 1)
        abs_p = os.path.join(repo, target)
        os.makedirs(os.path.dirname(abs_p), exist_ok=True)
        with open(abs_p, "w") as f:
            f.writelines([l.value for h in p for l in h if not l.is_removed])
        changed.append(target)
    return "\n".join(changed)

# -------------------------
# agentic_coder/tools/shell.py
# -------------------------
import subprocess

def run(repo, cmd, timeout=120):
    proc = subprocess.run(cmd, cwd=repo, shell=True, capture_output=True, text=True, timeout=timeout)
    return f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}"

# -------------------------
# agentic_coder/tools/git_tools.py
# -------------------------
import subprocess

def _run(repo, args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout

def git_status(repo):
    return _run(repo, ["status", "-sb"])

def git_diff(repo, staged=False):
    return _run(repo, ["diff", "--staged" if staged else "--"])

def git_commit(repo, message, all=False):
    if all:
        _run(repo, ["add", "-A"])
    return _run(repo, ["commit", "-m", message])

# -------------------------
# agentic_coder/cli.py
# -------------------------
import typer, os
from rich import print
from .runtime import Agent
from .config import MODEL_PRESETS

app = typer.Typer()

@app.command()
def models():
    for k,v in MODEL_PRESETS.items():
        print(f"[bold]{k}[/bold] -> {v.id}")

@app.command()
def run(model: str = typer.Option("qwen2.5-coder-14b"),
        repo: str = typer.Option("."),
        goal: str = typer.Option(...),
        max_iters: int = typer.Option(20),
        remote_model: str = typer.Option(None),
        remote_base_url: str = typer.Option(None),
        remote_api_key: str = typer.Option(None)):
    agent = Agent(model, repo, remote_model=remote_model,
                  remote_base_url=remote_base_url, remote_api_key=remote_api_key)
    agent.run(goal=goal, max_iters=max_iters)

if __name__ == "__main__":
    app()

# -------------------------
# README.md
# -------------------------
# Agentic Coder

A Claude Code–style coding agent with switchable **open-weight** models and optional **OpenAI-/Anthropic-compatible** remotes. Uses `prompts/AGENT.md` for system rules and a compact ReAct loop (THINK/PLAN/ACTION → OBSERVATION). Tools: file I/O, repo search, unified-diff patching, shell, and git.

## Install
```bash
pip install -e .
```

## List presets
```bash
agentic-coder models
```

## Run (local model)
```bash
agentic-coder run --model qwen2.5-coder-14b --repo . \
  --goal "Fix failing tests and add --dry-run flag"
```

## Run (remote OpenAI-compatible)
```bash
export OPENAI_BASE_URL="https://my-openai-proxy/v1"
export OPENAI_API_KEY="sk-..."
agentic-coder run --model remote-openai --remote-model gpt-4o-mini --repo . \
  --goal "Refactor module Y"
```

## Run (remote Anthropic-compatible)
```bash
export ANTHROPIC_BASE_URL="https://my-anthropic-proxy"
export ANTHROPIC_API_KEY="sk-ant-..."
agentic-coder run --model remote-anthropic --remote-model claude-3-5-sonnet --repo . \
  --goal "Harden error handling"
```

# -------------------------
# prompts/AGENT.md
# -------------------------
# Agent Rules (AGENT.md)

You are **Agentic Coder**, a decisive coding agent. Your job is to **plan, act, observe, and iterate** until the user's goal is satisfied. Prefer **small, safe, verifiable changes**. When uncertain, **run diagnostics** and gather evidence.

## Operating Principles
- Always follow this cycle:
  1) **THINK**: analyze repository context, tests, and goal.
  2) **PLAN**: list concrete steps; choose the highest-leverage next action.
  3) **ACT** using one tool per step when possible.
  4) **OBSERVE**: read outputs; update plan.
  5) **VERIFY** with tests, linters, or commands.
- Minimize churn. Prefer **surgical diffs** via `apply_patch`.
- Keep edits **compilable**. If a patch might break build, split into smaller patches.
- Use repository signals: `git status`, failing tests output, CI scripts, `pyproject`, `package.json`, etc.
- If the repo uses a language you don't know, run probes (e.g., `tree`, `grep`, `build`/`test` scripts) before editing.
- Never invent files or APIs—**read before you write** (`read_file`, `list_dir`, `search_text`).
- After substantial changes, run the test suite or a targeted subset.
- Always produce diffs with correct **unified format** and file paths rooted at repo.

## Tool Selection Heuristics
- **read_file / list_dir / search_text**: gather context before edits.
- **apply_patch**: create or modify files via unified diff; split risky changes.
- **run**: compile, test, run scripts, print logs.
- **git_status/git_diff/git_commit**: commit independently verifiable steps with meaningful messages.

## I/O Protocol
Speak in this protocol exactly:
```
THINK: <your private reasoning>
PLAN: <numbered steps>
ACTION: <tool_name> {"arg": "value", ...}
```
After a tool runs, you will receive `OBSERVATION:`. Update your plan and continue. Finish only when the goal is satisfied; then output a short summary and next steps.

## Safety & Idempotence
- Default to **idempotent** commands.
- Never run destructive commands without explicit user approval (`rm -rf`, force push, rewriting history).
- Preserve comments and formatting when editing configs or scripts.

# -------------------------
# agentic_coder/__init__.py
# -------------------------
from .config import ModelSpec, MODEL_PRESETS
from .llm_registry import load_model

__all__ = ["ModelSpec", "MODEL_PRESETS", "load_model"]

# -------------------------
# agentic_coder/planner.py
# -------------------------
SYSTEM_TEMPLATE = """<SYSTEM>
You follow AGENT.md. Output only the protocol blocks.
</SYSTEM>"""

USER_TEMPLATE = """<GOAL>
{goal}
</GOAL>
<CONTEXT>
repo={repo}
</CONTEXT>
"""

def make_prompt(goal: str, repo: str) -> str:
    return SYSTEM_TEMPLATE + "
" + USER_TEMPLATE.format(goal=goal, repo=repo)

# -------------------------
# agentic_coder/tools/__init__.py
# -------------------------
from __future__ import annotations
from typing import Dict, Any, Callable
from .filesystem import read_file, write_file, list_dir, search_text
from .patch import apply_patch
from .shell import run
from .git_tools import git_status, git_diff, git_commit

ToolFn = Callable[[Dict[str, Any]], str]

class ToolRegistry:
    def __init__(self, repo: str):
        self.repo = repo
        self._tools: dict[str, ToolFn] = {
            # File system
            "read_file": lambda a: read_file(self.repo, **a),
            "write_file": lambda a: write_file(self.repo, **a),
            "list_dir": lambda a: list_dir(self.repo, **a),
            "search_text": lambda a: search_text(self.repo, **a),
            # Patch
            "apply_patch": lambda a: apply_patch(self.repo, **a),
            # Shell and Git
            "run": lambda a: run(self.repo, **a),
            "git_status": lambda a: git_status(self.repo, **a),
            "git_diff": lambda a: git_diff(self.repo, **a),
            "git_commit": lambda a: git_commit(self.repo, **a),
        }

    def dispatch(self, tool: str, args: Dict[str, Any]) -> str:
        fn = self._tools.get(tool)
        if not fn:
            return f"ERROR: unknown tool '{tool}'"
        try:
            return fn(args or {})
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def goal_satisfied(self, goal: str) -> bool:
        return False

# -------------------------
# agentic_coder/tools/filesystem.py
# -------------------------
from __future__ import annotations
import os
import re
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
Description: Ripgrep-like simple search across repository for a pattern (regex).
Args: {"pattern": "regex", "path": "."}
"""

def _abs(repo: str, path: str) -> str:
    ap = os.path.abspath(os.path.join(repo, path))
    if not ap.startswith(os.path.abspath(repo)):
        raise ValueError("Path escape not allowed")
    return ap


def read_file(repo: str, path: str) -> str:
    with open(_abs(repo, path), "r", encoding="utf-8") as f:
        return f.read()


def write_file(repo: str, path: str, content: str) -> str:
    abspath = _abs(repo, path)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    with open(abspath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"WROTE: {path} ({len(content)} bytes)"


def list_dir(repo: str, path: str = ".", recursive: bool = False) -> str:
    base = _abs(repo, path)
    if not recursive:
        return "
".join(sorted(os.listdir(base)))
    acc = []
    for root, dirs, files in os.walk(base):
        rel_root = os.path.relpath(root, repo)
        for d in sorted(dirs):
            acc.append(os.path.join(rel_root, d) + "/")
        for fn in sorted(files):
            acc.append(os.path.join(rel_root, fn))
    return "
".join(acc)


def search_text(repo: str, pattern: str, path: str = ".") -> str:
    rx = re.compile(pattern)
    base = _abs(repo, path)
    hits = []
    for root, _, files in os.walk(base):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            rel = os.path.relpath(p, repo)
                            hits.append(f"{rel}:{i}:{line.strip()}")
            except Exception:
                continue
    return "
".join(hits)

# -------------------------
# agentic_coder/tools/patch.py
# -------------------------
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

    return "
".join(changed) if changed else "NOOP"

# -------------------------
# agentic_coder/tools/shell.py
# -------------------------
from __future__ import annotations
import subprocess

"""
Tool: run
Description: Run shell commands inside the repository. Use for builds, tests, or quick scripts. Defaults to bash.
Args: {"cmd": "string", "timeout": 120}
Returns: combined stdout/stderr and exit code.
"""

def run(repo: str, cmd: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        cmd,
        cwd=repo,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        executable="/bin/bash",
    )
    return f"exit={proc.returncode}
{proc.stdout}{proc.stderr}"

# -------------------------
# agentic_coder/tools/git_tools.py
# -------------------------
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

# -------------------------
# agentic_coder/cli.py
# -------------------------
from __future__ import annotations
import os
import typer
from rich import print
from .runtime import Agent
from .config import MODEL_PRESETS

app = typer.Typer(add_completion=False)

@app.command()
def models():
    for k, v in MODEL_PRESETS.items():
        print(f"[bold]{k}[/bold] -> {v.id}{' (remote)' if v.remote else ''}")

@app.command()
def run(model: str = typer.Option("qwen2.5-coder-14b", help="Model preset key"),
        repo: str = typer.Option(".", help="Path to a git repo or project root"),
        goal: str = typer.Option(..., help="High-level goal for the agent"),
        max_iters: int = typer.Option(20, help="Max reasoning/acting iterations"),
        remote_model: str = typer.Option(None, help="Remote model name (for remote presets)"),
        remote_base_url: str = typer.Option(None, help="Remote base URL (overrides env)"),
        remote_api_key: str = typer.Option(None, help="Remote API key (overrides env)")):
    if not os.path.isdir(repo):
        raise typer.BadParameter("repo must be a directory")
    agent = Agent(model, repo, remote_model=remote_model,
                  remote_base_url=remote_base_url, remote_api_key=remote_api_key)
    agent.run(goal=goal, max_iters=max_iters)

if __name__ == "__main__":
    app()
