from __future__ import annotations
import json
import os
import shlex
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.live import Live

from .runtime import Agent
from .planner import make_prompt
from .config import MODEL_PRESETS
from .commands import CommandProcessor

HELP = """
Enhanced Command System (Claude Code Style):
Use : or / prefix for commands. Type :help or /help for full command list.

Quick commands:
  :help                     Show full help
  :goal <text>              Set goal  
  :step                     Execute one step
  :go [n]                   Execute n steps
  :read <file>              Read file
  :write <file>             Write file
  :grep <pattern>           Search code
  :quit                     Exit

Free text input is appended to the goal. End multi-line with '.' on its own line.
"""

console = Console()

class Session:
    def __init__(self, repo: str, preset: str):
        self.repo = repo
        self.preset = preset
        self.agent = Agent(preset, repo)
        self.goal: Optional[str] = None
        self.history: Optional[str] = None
        self.max_history_chars: int = 8000  # crude token budget
        self.streaming: bool = True  # default ON

    @property
    def tools(self):
        return self.agent.tools

    def set_goal(self, text: str):
        self.goal = text.strip()
        self.history = make_prompt(self.goal, self.repo)

    def ensure_goal(self):
        if not self.goal:
            raise RuntimeError("No goal set. Use :goal <text> or type free text to set it.")

    def _display_stream(self, generator) -> str:
        """Render streaming text live and return the final concatenated text."""
        buf = ""
        with Live(Panel("", title="MODEL (streaming)"), refresh_per_second=24, console=console) as live:
            for piece in generator:
                if piece:
                    buf += piece
                    live.update(Panel(Syntax(buf, "markdown", theme="ansi_dark"), title="MODEL (streaming)"))
        # Final static display for scrollback readability
        console.rule("[bold]MODEL[/bold]")
        console.print(Syntax(buf, "markdown", theme="ansi_dark"))
        return buf

    def step(self) -> bool:
        """Run one THINK→PLAN→ACTION turn (streaming by default)."""
        self.ensure_goal()
        # Trim tail to avoid prompt blow-up
        if self.history and len(self.history) > self.max_history_chars:
            self.history = self.history[-self.max_history_chars:]
        if self.streaming:
            response = self._display_stream(self.agent.stream(self.history))
        else:
            response = self.agent._gen(self.history)
            console.rule("[bold]MODEL[/bold]")
            console.print(Syntax(response, "markdown", theme="ansi_dark"))

        action = self.agent._extract_action(response)
        if not action:
            console.print("[yellow]No ACTION detected.")
            return False

        tool_name, args = action
        console.rule("[bold]ACTION[/bold]")
        console.print(f"[bold]{tool_name}[/bold] {json.dumps(args, ensure_ascii=False)}")

        obs = self.tools.dispatch(tool_name, args)
        console.rule("[bold]OBSERVATION[/bold]")
        # If we just applied a patch or showed a diff, pretty print
        if tool_name in {"apply_patch"}:
            try:
                # show the diff argument (if present) with syntax highlighting
                diff_text = args.get("diff", "")
                if diff_text:
                    console.print(Syntax(diff_text, "diff"))
            except Exception:
                pass
        # Also show the tool's JSON envelope
        console.print(obs if isinstance(obs, str) else repr(obs))

        self.history += f"\nOBSERVATION: {obs}\n"
        return True

    def run(self, max_steps: int = 10):
        for _ in range(max_steps):
            if not self.step():
                break

    def switch_model(self, preset: str):
        self.preset = preset
        self.agent = Agent(preset, self.repo, temperature=self.agent.temperature)

def _list_models():
    tbl = Table(title="Model Presets", show_lines=False)
    tbl.add_column("Preset", style="bold")
    tbl.add_column("Target")
    for k, spec in MODEL_PRESETS.items():
        target = spec.id
        tbl.add_row(k, target)
    console.print(tbl)

def _list_tools(sess: Session):
    names = sorted(list(sess.tools._tools.keys()))
    console.print(Panel("\n".join(names), title="Available Tools"))

def _plan_state(repo: str) -> str:
    pf = os.path.join(repo, ".agentic_plan.json")
    if not os.path.exists(pf):
        return "inactive"
    try:
        data = json.load(open(pf, "r", encoding="utf-8"))
        return "active" if data.get("active") else "inactive"
    except Exception:
        return "unknown"

def _multiline_input(prompt="… "):
    lines = []
    console.print("[dim]Enter text. End with a single '.' on a new line.[/dim]")
    while True:
        s = Prompt.ask(prompt)
        if s.strip() == ".":
            break
        lines.append(s)
    return "\n".join(lines).strip()

def repl(repo: str = ".", preset: str = "qwen2.5-coder-14b"):
    """Enhanced REPL with Claude Code style commands."""
    sess = Session(repo, preset)
    cmd_processor = CommandProcessor(sess)
    
    console.print(Panel.fit("Agentic Coder REPL", subtitle="Enhanced Claude Code–style interface"))
    console.print(f"[dim]repo={repo}  model={preset}  plan={_plan_state(repo)}  streaming={'on' if sess.streaming else 'off'}[/dim]")
    console.print("[dim]Use :help or /help for commands. : or / prefixes supported.[/dim]")
    
    while True:
        try:
            raw = Prompt.ask("[bold cyan]agent[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        # Process command using the enhanced command processor
        should_continue = cmd_processor.process_command(raw)
        if not should_continue:
            break

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Path to repo")
    ap.add_argument("--model", default="qwen2.5-coder-14b", help="Model preset key")
    args = ap.parse_args()
    repl(args.repo, args.model)