from __future__ import annotations
import os
import shlex
import json
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

"""
Command system for Agentic Coder - supports both : and / prefixes like Claude Code
"""

console = Console()

class CommandProcessor:
    """Process slash and colon commands for interactive mode."""
    
    def __init__(self, session):
        self.session = session
        self.commands = {
            # Core commands (Claude Code style)
            'help': self.cmd_help,
            'h': self.cmd_help,
            'models': self.cmd_models,
            'use': self.cmd_use,
            'who': self.cmd_who,
            'tools': self.cmd_tools,
            'reset': self.cmd_reset,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,
            'q': self.cmd_quit,
            
            # Goal and execution
            'goal': self.cmd_goal,
            'step': self.cmd_step,
            'go': self.cmd_go,
            'run': self.cmd_run,
            
            # File operations (Claude Code style)
            'read': self.cmd_read,
            'write': self.cmd_write,
            'edit': self.cmd_edit,
            'add': self.cmd_add,
            
            # Plan mode
            'plan': self.cmd_plan,
            'approve': self.cmd_approve,
            'reject': self.cmd_reject,
            
            # Search and web
            'grep': self.cmd_grep,
            'search': self.cmd_grep,  # alias
            'web': self.cmd_web,
            'fetch': self.cmd_fetch,
            
            # Settings
            'stream': self.cmd_stream,
            'temp': self.cmd_temp,
            
            # Analysis (our enhanced features)
            'analyze': self.cmd_analyze,
            'ast': self.cmd_ast,
            'map': self.cmd_map,
            
            # Hooks system
            'hooks': self.cmd_hooks,
            'hook': self.cmd_hook,
            
            # Memory management
            'memory': self.cmd_memory,
            'sessions': self.cmd_sessions,
            'load': self.cmd_load_session,
        }

    def process_command(self, raw_input: str) -> bool:
        """Process a command. Returns True if should continue, False to quit."""
        if not raw_input.strip():
            return True
            
        # Support both : and / prefixes like Claude Code
        if raw_input.startswith(('::', '//')):
            # Double prefix shows raw command
            console.print(f"[dim]{raw_input}[/dim]")
            return True
        elif raw_input.startswith((':/', '/:')):
            # Mixed prefix - treat as regular text
            return self._handle_text_input(raw_input)
        elif raw_input.startswith((':', '/')):
            return self._handle_command(raw_input[1:])
        else:
            return self._handle_text_input(raw_input)

    def _handle_command(self, command_str: str) -> bool:
        """Handle a parsed command."""
        try:
            parts = shlex.split(command_str)
            if not parts:
                return True
                
            cmd_name = parts[0].lower()
            args = parts[1:]
            
            if cmd_name in self.commands:
                return self.commands[cmd_name](args)
            else:
                console.print(f"[red]Unknown command: {cmd_name}[/red]")
                console.print("[dim]Type :help or /help for available commands[/dim]")
                return True
        except Exception as e:
            console.print(f"[red]Command error: {e}[/red]")
            return True

    def _handle_text_input(self, text: str) -> bool:
        """Handle regular text input - append to goal."""
        if text.strip() == ".":
            return True  # Empty line
            
        current_goal = getattr(self.session, 'goal', '') or ''
        if current_goal:
            new_goal = f"{current_goal}\n{text}"
        else:
            new_goal = text
            
        self.session.set_goal(new_goal.strip())
        console.print(f"[dim]Goal updated. Use :step or /step to execute.[/dim]")
        return True

    # Command implementations
    def cmd_help(self, args: List[str]) -> bool:
        """Show help message."""
        help_text = """
Commands (use : or / prefix):
  help, h                   Show this help
  models                    List available models  
  use <preset>              Switch model preset
  who                       Show current status
  tools                     List available tools
  reset                     Clear history and goal
  quit, exit, q             Exit

Goal & Execution:
  goal <text>               Set goal for agent
  step                      Execute one agent step
  go [n]                    Execute n steps (default 10)
  run <command>             Run shell command

File Operations:
  read <path>               Read file
  write <path>              Write file (interactive)
  edit <path>               Edit file 
  add <path>                Create new file

Search & Analysis:
  grep <pattern>            Search in repository
  search <pattern>          Alias for grep
  web <query>               Web search
  fetch <url>               Fetch URL content
  analyze [file]            AST analysis
  ast <file>                Parse file AST
  map                       Repository map

Plan Mode:
  plan                      Enter plan mode
  approve                   Approve plan
  reject                    Reject plan

Settings:
  stream on|off             Toggle streaming
  temp <value>              Set temperature (0.0-1.0)

Hooks & Extensibility:
  hooks                     List all hooks
  hook add <name> <trigger> <cmd>  Add new hook
  hook enable <name>        Enable hook
  hook disable <name>       Disable hook

Memory & Sessions:
  memory                    Show current session memory
  sessions                  List all sessions
  load <session_id>         Load a specific session

Text input without : or / is appended to the goal.
End multi-line input with a single '.' on its own line.
        """
        console.print(Panel(help_text.strip(), title="Agentic Coder Commands"))
        return True

    def cmd_models(self, args: List[str]) -> bool:
        """List available models."""
        from .config import MODEL_PRESETS
        table = Table(title="Available Models")
        table.add_column("Preset", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Type", style="yellow")
        
        for preset, spec in MODEL_PRESETS.items():
            model_type = "Remote" if spec.remote else "Local"
            table.add_row(preset, spec.id, model_type)
        
        console.print(table)
        return True

    def cmd_use(self, args: List[str]) -> bool:
        """Switch model preset."""
        if not args:
            console.print("[red]Usage: :use <preset>[/red]")
            return True
            
        try:
            self.session.switch_model(args[0])
            console.print(f"[green]Switched to model preset: {args[0]}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to switch model: {e}[/red]")
        return True

    def cmd_who(self, args: List[str]) -> bool:
        """Show current status."""
        plan_state = self._get_plan_state()
        console.print(f"[bold]Status:[/bold]")
        console.print(f"  Repo: {self.session.repo}")
        console.print(f"  Model: {self.session.preset}")
        console.print(f"  Plan Mode: {plan_state}")
        console.print(f"  Streaming: {'on' if getattr(self.session, 'streaming', True) else 'off'}")
        console.print(f"  Goal: {getattr(self.session, 'goal', 'None')[:100]}{'...' if len(getattr(self.session, 'goal', '')) > 100 else ''}")
        return True

    def cmd_tools(self, args: List[str]) -> bool:
        """List available tools."""
        tools = list(self.session.tools._tools.keys())
        tools.sort()
        
        table = Table(title="Available Tools")
        table.add_column("Tool", style="cyan")
        
        for tool in tools:
            table.add_row(tool)
        
        console.print(table)
        return True

    def cmd_reset(self, args: List[str]) -> bool:
        """Reset session."""
        self.session.set_goal("")
        console.print("[green]Session reset - goal cleared[/green]")
        return True

    def cmd_quit(self, args: List[str]) -> bool:
        """Quit the session."""
        console.print("[yellow]Goodbye![/yellow]")
        return False

    def cmd_goal(self, args: List[str]) -> bool:
        """Set goal."""
        if not args:
            current = getattr(self.session, 'goal', 'None')
            console.print(f"[bold]Current goal:[/bold] {current}")
            return True
            
        goal_text = ' '.join(args)
        self.session.set_goal(goal_text)
        console.print(f"[green]Goal set:[/green] {goal_text}")
        return True

    def cmd_step(self, args: List[str]) -> bool:
        """Execute one step."""
        try:
            if self.session.step():
                console.print("[green]Step completed[/green]")
            else:
                console.print("[yellow]No action taken or goal satisfied[/yellow]")
        except Exception as e:
            console.print(f"[red]Step failed: {e}[/red]")
        return True

    def cmd_go(self, args: List[str]) -> bool:
        """Execute multiple steps."""
        max_steps = 10
        if args:
            try:
                max_steps = int(args[0])
            except ValueError:
                console.print(f"[red]Invalid step count: {args[0]}[/red]")
                return True
                
        try:
            result = self.session.run(max_steps)
            console.print(f"[green]Executed {max_steps} steps[/green]")
        except Exception as e:
            console.print(f"[red]Execution failed: {e}[/red]")
        return True

    def cmd_run(self, args: List[str]) -> bool:
        """Run shell command."""
        if not args:
            console.print("[red]Usage: :run <command>[/red]")
            return True
            
        command = ' '.join(args)
        try:
            result = self.session.tools.dispatch("run", {"cmd": command})
            console.print(Syntax(result, "json", theme="ansi_dark"))
        except Exception as e:
            console.print(f"[red]Command failed: {e}[/red]")
        return True

    def cmd_read(self, args: List[str]) -> bool:
        """Read file."""
        if not args:
            console.print("[red]Usage: :read <path>[/red]")
            return True
            
        try:
            result = self.session.tools.dispatch("read_file", {"path": args[0]})
            data = json.loads(result)
            if data.get("status") == "ok":
                content = data.get("output", "")
                console.print(Panel(Syntax(content, "python", theme="ansi_dark"), title=f"File: {args[0]}"))
            else:
                console.print(f"[red]Read failed: {data.get('error', 'Unknown error')}[/red]")
        except Exception as e:
            console.print(f"[red]Read failed: {e}[/red]")
        return True

    def cmd_write(self, args: List[str]) -> bool:
        """Write file interactively."""
        if not args:
            console.print("[red]Usage: :write <path>[/red]")
            return True
            
        from rich.prompt import Prompt
        path = args[0]
        console.print(f"[bold]Writing to {path}[/bold]")
        console.print("[dim]Enter content. End with a single '.' on its own line.[/dim]")
        
        lines = []
        while True:
            line = Prompt.ask("… ")
            if line.strip() == ".":
                break
            lines.append(line)
        
        content = '\n'.join(lines)
        try:
            result = self.session.tools.dispatch("write_file", {"path": path, "content": content})
            data = json.loads(result)
            if data.get("status") == "ok":
                console.print(f"[green]File written: {path}[/green]")
            else:
                console.print(f"[red]Write failed: {data.get('error', 'Unknown error')}[/red]")
        except Exception as e:
            console.print(f"[red]Write failed: {e}[/red]")
        return True

    def cmd_edit(self, args: List[str]) -> bool:
        """Edit file (alias for write after read)."""
        if not args:
            console.print("[red]Usage: :edit <path>[/red]")
            return True
            
        # First show current content
        self.cmd_read(args)
        console.print("\n[dim]Now enter new content:[/dim]")
        return self.cmd_write(args)

    def cmd_add(self, args: List[str]) -> bool:
        """Create new file (alias for write)."""
        return self.cmd_write(args)

    def cmd_grep(self, args: List[str]) -> bool:
        """Search in repository."""
        if not args:
            console.print("[red]Usage: :grep <pattern>[/red]")
            return True
            
        try:
            search_args = {"pattern": args[0]}
            if len(args) > 1:
                search_args["glob"] = args[1]
                
            result = self.session.tools.dispatch("search_text", search_args)
            console.print(Syntax(result, "json", theme="ansi_dark"))
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
        return True

    def cmd_web(self, args: List[str]) -> bool:
        """Web search."""
        if not args:
            console.print("[red]Usage: :web <query>[/red]")
            return True
            
        query = ' '.join(args)
        try:
            result = self.session.tools.dispatch("search_web", {"query": query})
            console.print(Syntax(result, "json", theme="ansi_dark"))
        except Exception as e:
            console.print(f"[red]Web search failed: {e}[/red]")
        return True

    def cmd_fetch(self, args: List[str]) -> bool:
        """Fetch URL content."""
        if not args:
            console.print("[red]Usage: :fetch <url>[/red]")
            return True
            
        try:
            result = self.session.tools.dispatch("fetch_url", {"url": args[0]})
            console.print(Syntax(result, "json", theme="ansi_dark"))
        except Exception as e:
            console.print(f"[red]Fetch failed: {e}[/red]")
        return True

    def cmd_plan(self, args: List[str]) -> bool:
        """Enter plan mode."""
        plan_file = os.path.join(self.session.repo, ".agentic_plan.json")
        data = {"active": True}
        try:
            with open(plan_file, 'w') as f:
                json.dump(data, f)
            console.print("[green]Plan mode activated[/green]")
        except Exception as e:
            console.print(f"[red]Failed to activate plan mode: {e}[/red]")
        return True

    def cmd_approve(self, args: List[str]) -> bool:
        """Approve plan and exit plan mode."""
        plan_file = os.path.join(self.session.repo, ".agentic_plan.json")
        try:
            if os.path.exists(plan_file):
                os.remove(plan_file)
            console.print("[green]Plan approved and plan mode deactivated[/green]")
        except Exception as e:
            console.print(f"[red]Failed to approve plan: {e}[/red]")
        return True

    def cmd_reject(self, args: List[str]) -> bool:
        """Reject plan and exit plan mode."""
        return self.cmd_approve(args)  # Same implementation for now

    def cmd_stream(self, args: List[str]) -> bool:
        """Toggle streaming."""
        if not args:
            current = getattr(self.session, 'streaming', True)
            console.print(f"Streaming is {'on' if current else 'off'}")
            return True
            
        setting = args[0].lower()
        if setting in ('on', 'true', '1'):
            self.session.streaming = True
            console.print("[green]Streaming enabled[/green]")
        elif setting in ('off', 'false', '0'):
            self.session.streaming = False
            console.print("[green]Streaming disabled[/green]")
        else:
            console.print("[red]Usage: :stream on|off[/red]")
        return True

    def cmd_temp(self, args: List[str]) -> bool:
        """Set temperature."""
        if not args:
            current = getattr(self.session, 'temperature', 0.0)
            console.print(f"Temperature: {current}")
            return True
            
        try:
            temp = float(args[0])
            if not (0.0 <= temp <= 1.0):
                console.print("[red]Temperature must be between 0.0 and 1.0[/red]")
                return True
            self.session.temperature = temp
            console.print(f"[green]Temperature set to {temp}[/green]")
        except ValueError:
            console.print(f"[red]Invalid temperature: {args[0]}[/red]")
        return True

    def cmd_analyze(self, args: List[str]) -> bool:
        """Analyze code structure."""
        try:
            analyze_args = {}
            if args:
                analyze_args["file_path"] = args[0]
            result = self.session.tools.dispatch("analyze_code", analyze_args)
            console.print(Panel(result, title="Code Analysis"))
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
        return True

    def cmd_ast(self, args: List[str]) -> bool:
        """Parse AST for file."""
        if not args:
            console.print("[red]Usage: :ast <file>[/red]")
            return True
            
        try:
            result = self.session.tools.dispatch("parse_ast", {"file_path": args[0]})
            console.print(Panel(result, title=f"AST: {args[0]}"))
        except Exception as e:
            console.print(f"[red]AST parsing failed: {e}[/red]")
        return True

    def cmd_map(self, args: List[str]) -> bool:
        """Show repository map."""
        try:
            result = self.session.tools.dispatch("repo_map", {})
            console.print(Panel(result, title="Repository Map"))
        except Exception as e:
            console.print(f"[red]Repository map failed: {e}[/red]")
        return True

    def _get_plan_state(self) -> str:
        """Get current plan mode state."""
        plan_file = os.path.join(self.session.repo, ".agentic_plan.json")
        try:
            if os.path.exists(plan_file):
                with open(plan_file) as f:
                    data = json.load(f)
                return "active" if data.get("active") else "inactive"
        except Exception:
            pass
        return "inactive"
    
    def cmd_hooks(self, args: List[str]) -> bool:
        """List all hooks."""
        try:
            from .hooks import HookManager
            hook_manager = HookManager(self.session.repo)
            hooks = hook_manager.list_hooks()
            
            if not hooks:
                console.print("[yellow]No hooks configured[/yellow]")
                return True
            
            table = Table(title="Configured Hooks")
            table.add_column("Name", style="cyan")
            table.add_column("Trigger", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Description", style="dim")
            
            for hook in hooks:
                status = "✓ Enabled" if hook.enabled else "✗ Disabled"
                table.add_row(hook.name, hook.trigger, status, hook.description)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Failed to list hooks: {e}[/red]")
        return True
    
    def cmd_hook(self, args: List[str]) -> bool:
        """Manage hooks."""
        if not args:
            console.print("[red]Usage: :hook <add|enable|disable|remove> [args...][/red]")
            return True
            
        try:
            from .hooks import HookManager
            hook_manager = HookManager(self.session.repo)
            action = args[0].lower()
            
            if action == 'add' and len(args) >= 4:
                name, trigger, command = args[1], args[2], ' '.join(args[3:])
                hook_manager.add_hook(name, command, trigger, f"Custom hook: {name}")
                console.print(f"[green]Added hook '{name}' for trigger '{trigger}'[/green]")
                
            elif action == 'enable' and len(args) >= 2:
                name = args[1]
                if hook_manager.enable_hook(name):
                    console.print(f"[green]Enabled hook '{name}'[/green]")
                else:
                    console.print(f"[red]Hook '{name}' not found[/red]")
                    
            elif action == 'disable' and len(args) >= 2:
                name = args[1]
                if hook_manager.disable_hook(name):
                    console.print(f"[yellow]Disabled hook '{name}'[/yellow]")
                else:
                    console.print(f"[red]Hook '{name}' not found[/red]")
                    
            elif action == 'remove' and len(args) >= 2:
                name = args[1]
                if hook_manager.remove_hook(name):
                    console.print(f"[yellow]Removed hook '{name}'[/yellow]")
                else:
                    console.print(f"[red]Hook '{name}' not found[/red]")
                    
            else:
                console.print("[red]Usage: :hook add <name> <trigger> <command>[/red]")
                console.print("[red]       :hook enable|disable|remove <name>[/red]")
                
        except Exception as e:
            console.print(f"[red]Hook operation failed: {e}[/red]")
        return True
    
    def cmd_memory(self, args: List[str]) -> bool:
        """Show current session memory."""
        try:
            memory_manager = getattr(self.session, 'memory_manager', None)
            if not memory_manager:
                console.print("[yellow]Memory management not enabled for this session[/yellow]")
                return True
            
            summary = memory_manager.get_session_summary()
            if not summary:
                console.print("[yellow]No active session[/yellow]")
                return True
            
            console.print("[bold]Current Session Memory:[/bold]")
            console.print(f"  Session ID: {summary['session_id']}")
            console.print(f"  Started: {summary['started_at']}")
            console.print(f"  Duration: {summary['duration_minutes']} minutes")
            console.print(f"  Total entries: {summary['total_entries']}")
            console.print(f"  Current goal: {summary['current_goal'] or 'None'}")
            
            if summary['entry_types']:
                console.print("  Entry types:")
                for entry_type, count in summary['entry_types'].items():
                    console.print(f"    {entry_type}: {count}")
            
            # Show recent memories if requested
            if args and args[0] == 'recent':
                recent = memory_manager.get_recent_memories(10)
                if recent:
                    console.print("\n[bold]Recent memories:[/bold]")
                    for entry in recent:
                        timestamp = entry.timestamp
                        content = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
                        console.print(f"  [{entry.type}] {content}")
            
        except Exception as e:
            console.print(f"[red]Memory operation failed: {e}[/red]")
        return True
    
    def cmd_sessions(self, args: List[str]) -> bool:
        """List all sessions."""
        try:
            from .memory import MemoryManager
            memory_manager = MemoryManager(self.session.repo)
            sessions = memory_manager.list_sessions()
            
            if not sessions:
                console.print("[yellow]No sessions found[/yellow]")
                return True
            
            table = Table(title="Available Sessions")
            table.add_column("Session ID", style="cyan")
            table.add_column("Started", style="green")
            table.add_column("Last Active", style="yellow")
            table.add_column("Model", style="blue")
            table.add_column("Entries", style="magenta")
            table.add_column("Goal", style="dim")
            
            for session in sessions:
                goal = session['current_goal'] or "None"
                if len(goal) > 50:
                    goal = goal[:47] + "..."
                
                table.add_row(
                    session['session_id'],
                    session['started_at'][:16],  # Just date and time
                    session['last_active'][:16],
                    session['model_preset'],
                    str(session['entries_count']),
                    goal
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Failed to list sessions: {e}[/red]")
        return True
    
    def cmd_load_session(self, args: List[str]) -> bool:
        """Load a specific session."""
        if not args:
            console.print("[red]Usage: :load <session_id>[/red]")
            return True
        
        try:
            from .memory import MemoryManager
            memory_manager = MemoryManager(self.session.repo)
            
            session_id = args[0]
            if memory_manager.load_session(session_id):
                # Update session's memory manager
                if hasattr(self.session, 'memory_manager'):
                    self.session.memory_manager = memory_manager
                
                console.print(f"[green]Loaded session: {session_id}[/green]")
                
                # Show session info
                summary = memory_manager.get_session_summary()
                if summary:
                    console.print(f"  Goal: {summary['current_goal'] or 'None'}")
                    console.print(f"  Entries: {summary['total_entries']}")
            else:
                console.print(f"[red]Failed to load session: {session_id}[/red]")
                
        except Exception as e:
            console.print(f"[red]Session load failed: {e}[/red]")
        return True