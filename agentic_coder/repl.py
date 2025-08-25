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

HELP = """
Commands:
  :help                     Show this help
  :models                   List available models
  :use <preset>             Switch model preset (local or remote preset key)
  :who                      Show current model + repo + plan-mode state
  :tools                    List available tools in registry
  :todos                    Show TODO list
  :done <index>             Mark a TODO done (status=done)
  :plan                     Begin plan-mode (prompt for plan text)
  :approve                  End plan-mode (approved=True)
  :reject                   End plan-mode (approved=False)
  :run <shell>              Run a shell command via tool (guardrails)
  :read <path>              Read file via tool
  :write <path> <<<EOF      Write file via tool (read-before-write enforced)
  :grep <regex> [glob]      Repo regex search via tool
  :web <query>              Web search via tool
  :fetch <url>              Fetch URL via tool
  :goal <text>              Set/replace the active goal for agent loop
  :step                     Think→Plan→Action once (streams model output)
  :go [n]                   Run up to n streamed steps (default 10)
  :stream on|off            Toggle streaming (default: on)
  :temp <value>             Set sampling temperature (0.0–1.0; default 0.0)
  :reset                    Clear history and active goal
  :quit                     Exit

Free text input (not starting with ':') is appended to the goal (multi-line allowed).
End multi-line input with a single '.' on its own line.
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
    sess = Session(repo, preset)
    console.print(Panel.fit("Agentic Coder REPL", subtitle="Claude Code–style interactive loop (streaming)"))
    console.print(f"[dim]repo={repo}  model={preset}  plan={_plan_state(repo)}  streaming={'on' if sess.streaming else 'off'}[/dim]")
    while True:
        try:
            raw = Prompt.ask("[bold cyan]agent[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not raw:
            continue

        if raw.startswith(":"):
            parts = shlex.split(raw[1:])
            cmd = parts[0] if parts else ""
            args = parts[1:]

            if cmd in {"q","quit","exit"}:
                break
            elif cmd in {"h","help"}:
                console.print(Panel(HELP.strip(), title="Help"))
            elif cmd == "models":
                _list_models()
            elif cmd == "use" and args:
                sess.switch_model(args[0])
                console.print(f"[green]Switched model to {args[0]}[/green]")
            elif cmd == "who":
                console.print(f"repo={sess.repo}  model={sess.preset}  plan={_plan_state(sess.repo)}  streaming={'on' if sess.streaming else 'off'}")
            elif cmd == "tools":
                _list_tools(sess)
            elif cmd == "todos":
                out = sess.tools.dispatch("todo_list", {})
                console.print(out)
            elif cmd == "done" and args:
                try:
                    idx = int(args[0])
                except:
                    console.print("Usage: :done <index>")
                else:
                    # read existing, flip to done using todo_update
                    out = sess.tools.dispatch("todo_update", {"index": idx, "title":"", "status":"done", "tags":[]})
                    console.print(out)
            elif cmd == "stream" and args:
                val = args[0].lower()
                if val in {"on", "off"}:
                    sess.streaming = (val == "on")
                    console.print(f"Streaming: {val}")
                else:
                    console.print("Usage: :stream on|off")
            elif cmd == "temp" and args:
                try:
                    t = float(args[0])
                    t = max(0.0, min(1.0, t))
                except:
                    console.print("Usage: :temp <0.0..1.0>")
                else:
                    sess.agent = Agent(sess.preset, sess.repo, temperature=t)
                    console.print(f"Temperature set to {t:.2f}")
            elif cmd == "goal":
                text = " ".join(args) if args else _multiline_input()
                sess.set_goal(text)
                console.print(Panel(Syntax(sess.goal, "markdown"), title="Goal"))
            elif cmd == "reset":
                sess.goal = None
                sess.history = None
                console.print("[yellow]Goal & history cleared.")
            elif cmd == "plan":
                plan = _multiline_input()
                out = sess.tools.dispatch("begin_plan", {"plan": plan, "rationale": ""})
                console.print(out)
            elif cmd == "approve":
                out = sess.tools.dispatch("end_plan", {"approved": True, "comment": ""})
                console.print(out)
            elif cmd == "reject":
                out = sess.tools.dispatch("end_plan", {"approved": False, "comment": ""})
                console.print(out)
            elif cmd == "step":
                try:
                    sess.step()
                except Exception as e:
                    console.print(f"[red]ERROR:[/red] {e}")
            elif cmd == "go":
                n = int(args[0]) if args else 10
                try:
                    sess.run(n)
                except Exception as e:
                    console.print(f"[red]ERROR:[/red] {e}")
            elif cmd == "run" and args:
                out = sess.tools.dispatch("run", {"cmd": " ".join(args)})
                console.print(out)
            elif cmd == "read" and args:
                out = sess.tools.dispatch("read_file", {"path": args[0]})
                console.print(Syntax(out, "text"))
            elif cmd == "write" and args:
                path = args[0]
                console.print(f"[dim]Enter content for {path}. End with '.' line.[/dim]")
                body = _multiline_input()
                out = sess.tools.dispatch("write_file", {"path": path, "content": body})
                console.print(out)
            elif cmd == "grep" and args:
                regex = args[0]
                glob = args[1] if len(args) > 1 else None
                out = sess.tools.dispatch("search_text", {"pattern": regex, "glob": glob})
                console.print(out)
            elif cmd == "web" and args:
                q = " ".join(args)
                out = sess.tools.dispatch("search_web", {"query": q})
                console.print(out)
            elif cmd == "fetch" and args:
                out = sess.tools.dispatch("fetch_url", {"url": args[0]})
                console.print(out)
            else:
                console.print("[yellow]Unknown command. :help for options.")
            continue

        # Free text: append to goal (create if missing)
        text = raw
        if text.strip() == ".":
            continue
        if not sess.goal:
            sess.set_goal(text)
            console.print(Panel(Syntax(sess.goal, "markdown"), title="Goal"))
        else:
            sess.goal += "\n" + text
            sess.history = make_prompt(sess.goal, sess.repo)
            console.print(Panel(Syntax(sess.goal, "markdown"), title="Goal (updated)"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Path to repo")
    ap.add_argument("--model", default="qwen2.5-coder-14b", help="Model preset key")
    args = ap.parse_args()
    repl(args.repo, args.model)