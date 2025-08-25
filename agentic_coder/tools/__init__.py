from __future__ import annotations
from typing import Dict, Any, Callable
import json, os, json as _json
from .filesystem import read_file, write_file, list_dir, search_text
from .patch import apply_patch
from .shell import run
from .git_tools import git_status, git_diff, git_commit
try:
    from .web_io import fetch_url, search_web
except Exception:
    fetch_url = None
    search_web = None
try:
    from .line_edit import edit_lines
except Exception:
    edit_lines = None
from .tests import run_tests
from .format import format_code
from .packages import install_deps
from .tree import repo_tree
from .grep_replace import replace_text
from .repo_map import repo_map
from .ast_parser import enhanced_repo_map, analyze_code_structure, parse_ast

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
            # Extras (Claude Code–style)
            "run_tests": lambda a: run_tests(self.repo, **a),
            "format_code": lambda a: format_code(self.repo, **a),
            "install_deps": lambda a: install_deps(self.repo, **a),
            "repo_tree": lambda a: repo_tree(self.repo, **a),
            "replace_text": lambda a: replace_text(self.repo, **a),
            "repo_map": lambda a: repo_map(self.repo, **a),
            # AST tools (with fallback)
            "analyze_code": lambda a: analyze_code_structure(self.repo, **a),
            "parse_ast": lambda a: parse_ast(self.repo, **a),
        }
        # Optional tools are registered elsewhere (e.g., web_io, fs_nav, grep_tool, etc.)
        # Plan-mode allowlist (non-mutating + planning)
        self._plan_allow = {
            # base read/list/search
            "read_file", "list_dir", "search_text", "repo_map", "analyze_code", "parse_ast",
            # nav/grep-style if present
            "list_path", "match_files", "find_text",
            # web tools if present
            "fetch_url", "search_web",
            # planning / notes
            "begin_plan", "end_plan", "task_note",
            # purely informational git
            "git_status", "git_diff",
        }
        if fetch_url and search_web:
            self._tools.update({
                "fetch_url": lambda a: fetch_url(self.repo, **a),
                "search_web": lambda a: search_web(self.repo, **a),
            })
        if edit_lines:
            self._tools.update({"edit_lines": lambda a: edit_lines(self.repo, **a)})

    def _plan_active(self) -> bool:
        p = os.path.join(self.repo, ".agentic_plan.json")
        if not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("active"))
        except Exception:
            return False

    def dispatch(self, tool: str, args: Dict[str, Any]) -> str:
        # PLAN-MODE GATING: when active, only allow read/search/list/plan tools.
        if hasattr(self, "_plan_active") and self._plan_active() and tool not in getattr(self, "_plan_allow", set()):
            return _json.dumps({"status":"plan_active","tool":tool,"error":f"'{tool}' blocked until end_plan"})
        if tool not in self._tools:
            return _json.dumps({"status":"error","tool":tool,"error":"unknown tool"})
        try:
            out = self._tools[tool](args or {})
            # If tool already returns JSON, passthrough; else wrap
            if isinstance(out, (dict, list)):
                return _json.dumps({"status":"ok","tool":tool,"data":out})
            if isinstance(out, str):
                try:
                    parsed = _json.loads(out)
                    # already JSON-like string
                    return _json.dumps({"status":"ok","tool":tool,"data":parsed})
                except Exception:
                    return _json.dumps({"status":"ok","tool":tool,"output":out})
            return _json.dumps({"status":"ok","tool":tool,"output":str(out)})
        except Exception as e:
            return _json.dumps({"status":"error","tool":tool,"error":f"{type(e).__name__}: {e}"})

    def goal_satisfied(self, goal: str) -> bool:
        return False