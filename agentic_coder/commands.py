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
Command system for Agentic Coder - uses / prefix for commands
"""

console = Console()

class CommandProcessor:
    """Process slash and colon commands for interactive mode."""
    
    def __init__(self, session):
        self.session = session
        self.commands = {
            # Core commands
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
            
            # File operations
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
            
            # MCP (Model Context Protocol)
            'mcp': self.cmd_mcp,
            'mcp-add': self.cmd_mcp_add,
            'mcp-connect': self.cmd_mcp_connect,
            'mcp-disconnect': self.cmd_mcp_disconnect,
            
            # GitHub Actions
            'workflows': self.cmd_workflows,
            'workflow-create': self.cmd_workflow_create,
            'workflow-validate': self.cmd_workflow_validate,
            
            # Model updates
            'update': self.cmd_update,
        }

    def process_command(self, raw_input: str) -> bool:
        """Process a command. Returns True if should continue, False to quit."""
        if not raw_input.strip():
            return True
            
        # Support only / prefix for commands
        if raw_input.startswith('//'):
            # Double slash shows raw command
            console.print(f"[dim]{raw_input}[/dim]")
            return True
        elif raw_input.startswith('/'):
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
                console.print("[dim]Type /help for available commands[/dim]")
                return True
        except Exception as e:
            console.print(f"[red]Command error: {e}[/red]")
            return True

    def _handle_text_input(self, text: str) -> bool:
        """Handle regular text input - set goal and auto-execute."""
        if text.strip() == ".":
            return True  # Empty line
            
        # Set the goal to the new text (don't append - just replace)
        self.session.set_goal(text.strip())
        
        # Auto-execute immediately
        try:
            # Run up to 10 steps automatically
            for i in range(10):
                if not self.session.step():
                    break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        
        return True

    # Command implementations
    def cmd_help(self, args: List[str]) -> bool:
        """Show help message."""
        # Check if specific topic requested
        if args and args[0] in ['file', 'files', 'search', 'plan', 'settings', 'session', 'mcp', 'github']:
            return self._show_topic_help(args[0])
        
        # Show concise main help
        help_text = """
[bold cyan]Quick Commands:[/]
  /help [topic]    Show help (topics: file, search, plan, settings)
  /models          List available models  
  /use <model>     Switch model
  /quit            Exit

[bold cyan]Execution:[/]
  /goal <text>     Set goal (auto-executes)
  /step            Execute one step
  /go [n]          Execute n steps (default 10)

[bold cyan]Files:[/]
  /read <path>     Read file
  /write <path>    Write file
  /grep <pattern>  Search in code

[bold cyan]Tips:[/]
• Type naturally without / and I'll auto-execute
• Press ESC to cancel processing
• Use /help <topic> for detailed help
        """
        console.print(help_text.strip())
        return True

    def _show_topic_help(self, topic: str) -> bool:
        """Show help for specific topic."""
        topics = {
            'file': """[bold cyan]File Operations:[/]
  /read <path>       Read file content
  /write <path>      Write/overwrite file (interactive)
  /edit <path>       Edit existing file
  /add <path>        Create new file""",
            
            'search': """[bold cyan]Search & Analysis:[/]
  /grep <pattern>    Search in repository
  /search <pattern>  Alias for grep
  /web <query>       Web search
  /fetch <url>       Fetch URL content
  /analyze [file]    AST analysis
  /map               Repository map""",
            
            'plan': """[bold cyan]Plan Mode:[/]
  /plan              Enter plan mode
  /approve           Approve plan
  /reject            Reject plan""",
            
            'settings': """[bold cyan]Settings:[/]
  /stream on|off     Toggle streaming
  /temp <value>      Set temperature (0.0-1.0)
  /who               Show current status
  /tools             List available tools
  /reset             Clear conversation
  /update            Update model configs""",
            
            'session': """[bold cyan]Sessions & Memory:[/]
  /memory            Show session memory
  /sessions          List all sessions  
  /load <id>         Load specific session""",
            
            'mcp': """[bold cyan]MCP Integration:[/]
  /mcp               List MCP servers
  /mcp-add <n> <c>   Add MCP server
  /mcp-connect <n>   Connect to server
  /mcp-disconnect    Disconnect server""",
            
            'github': """[bold cyan]GitHub Actions:[/]
  /workflows         List workflows
  /workflow-create   Create workflow
  /workflow-validate Validate workflow"""
        }
        
        # Normalize topic name
        if topic == 'files':
            topic = 'file'
        
        if topic in topics:
            console.print(topics[topic])
        else:
            console.print(f"[yellow]Unknown topic: {topic}[/]")
            console.print("[dim]Available topics: file, search, plan, settings, session, mcp, github[/]")
        
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
        """Switch model."""
        if not args:
            console.print("[red]Usage: /use <model>[/red]")
            console.print("[dim]Examples: /use claude-3-5-sonnet, /use gpt-4o, /use qwen2.5-coder-14b[/dim]")
            return True
            
        try:
            self.session.switch_model(args[0])
            console.print(f"[green]Switched to model: {args[0]}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to switch model: {e}[/red]")
        return True

    def cmd_who(self, args: List[str]) -> bool:
        """Show current status."""
        plan_state = self._get_plan_state()
        console.print(f"[bold]Status:[/bold]")
        console.print(f"  Repo: {self.session.repo}")
        console.print(f"  Model: {self.session.model}")
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
            console.print("[red]Usage: /run <command>[/red]")
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
            console.print("[red]Usage: /read <path>[/red]")
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
            console.print("[red]Usage: /write <path>[/red]")
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
            console.print("[red]Usage: /edit <path>[/red]")
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
            console.print("[red]Usage: /grep <pattern>[/red]")
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
            console.print("[red]Usage: /web <query>[/red]")
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
            console.print("[red]Usage: /fetch <url>[/red]")
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
            console.print("[red]Usage: /stream on|off[/red]")
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
            console.print("[red]Usage: /ast <file>[/red]")
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
            console.print("[red]Usage: /hook <add|enable|disable|remove> [args...][/red]")
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
                console.print("[red]Usage: /hook add <name> <trigger> <command>[/red]")
                console.print("[red]       /hook enable|disable|remove <name>[/red]")
                
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
            console.print("[red]Usage: /load <session_id>[/red]")
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
    
    def cmd_mcp(self, args: List[str]) -> bool:
        """List MCP servers."""
        try:
            mcp_client = getattr(self.session, 'mcp_client', None)
            if not mcp_client:
                from .mcp import MCPClient
                mcp_client = MCPClient()
                if hasattr(self.session, 'mcp_client'):
                    self.session.mcp_client = mcp_client
            
            servers = mcp_client.list_servers()
            
            if not servers:
                console.print("[yellow]No MCP servers configured[/yellow]")
                console.print("[dim]Use :mcp-add <name> <command> to add a server[/dim]")
                return True
            
            table = Table(title="MCP Servers")
            table.add_column("Name", style="cyan")
            table.add_column("Command", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Capabilities", style="dim")
            
            for server in servers:
                status = "✓ Connected" if server['connected'] else "✗ Disconnected"
                caps = ""
                if server['capabilities']:
                    cap_list = []
                    if server['capabilities'].get('resources'):
                        cap_list.append("resources")
                    if server['capabilities'].get('tools'):
                        cap_list.append("tools")
                    if server['capabilities'].get('prompts'):
                        cap_list.append("prompts")
                    caps = ", ".join(cap_list)
                
                table.add_row(server['name'], server['command'], status, caps)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]MCP operation failed: {e}[/red]")
        return True
    
    def cmd_mcp_add(self, args: List[str]) -> bool:
        """Add an MCP server."""
        if len(args) < 2:
            console.print("[red]Usage: /mcp-add <name> <command> [args...][/red]")
            return True
        
        try:
            from .mcp import MCPClient
            mcp_client = getattr(self.session, 'mcp_client', None)
            if not mcp_client:
                mcp_client = MCPClient()
                if hasattr(self.session, 'mcp_client'):
                    self.session.mcp_client = mcp_client
            
            name = args[0]
            command = args[1]
            cmd_args = args[2:] if len(args) > 2 else []
            
            mcp_client.add_server(name, command, cmd_args)
            mcp_client.save_config()
            
            console.print(f"[green]Added MCP server '{name}'[/green]")
            console.print(f"[dim]Use :mcp-connect {name} to connect[/dim]")
            
        except Exception as e:
            console.print(f"[red]Failed to add MCP server: {e}[/red]")
        return True
    
    def cmd_mcp_connect(self, args: List[str]) -> bool:
        """Connect to an MCP server."""
        if not args:
            console.print("[red]Usage: /mcp-connect <name>[/red]")
            return True
        
        try:
            import asyncio
            from .mcp import MCPClient
            
            mcp_client = getattr(self.session, 'mcp_client', None)
            if not mcp_client:
                mcp_client = MCPClient()
                if hasattr(self.session, 'mcp_client'):
                    self.session.mcp_client = mcp_client
            
            name = args[0]
            console.print(f"[yellow]Connecting to MCP server '{name}'...[/yellow]")
            
            # Run async connection
            connected = asyncio.run(mcp_client.connect_server(name))
            
            if connected:
                console.print(f"[green]Connected to MCP server '{name}'[/green]")
                
                # Register MCP tools with our tool registry if available
                if hasattr(self.session, 'tools'):
                    from .mcp import register_mcp_tools
                    register_mcp_tools(self.session.tools, mcp_client)
                    console.print(f"[green]MCP tools from '{name}' registered[/green]")
            else:
                console.print(f"[red]Failed to connect to MCP server '{name}'[/red]")
                
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/red]")
        return True
    
    def cmd_mcp_disconnect(self, args: List[str]) -> bool:
        """Disconnect from an MCP server."""
        if not args:
            console.print("[red]Usage: /mcp-disconnect <name>[/red]")
            return True
        
        try:
            import asyncio
            from .mcp import MCPClient
            
            mcp_client = getattr(self.session, 'mcp_client', None)
            if not mcp_client:
                console.print("[yellow]No MCP client initialized[/yellow]")
                return True
            
            name = args[0]
            disconnected = asyncio.run(mcp_client.disconnect_server(name))
            
            if disconnected:
                console.print(f"[green]Disconnected from MCP server '{name}'[/green]")
            else:
                console.print(f"[yellow]Server '{name}' was not connected[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Disconnect failed: {e}[/red]")
        return True
    
    def cmd_workflows(self, args: List[str]) -> bool:
        """List GitHub Actions workflows."""
        try:
            from .github_actions import GitHubActionsManager
            ga_manager = GitHubActionsManager(self.session.repo)
            workflows = ga_manager.list_workflows()
            
            if not workflows:
                console.print("[yellow]No GitHub Actions workflows found[/yellow]")
                console.print("[dim]Use :workflow-create <type> to create workflows[/dim]")
                return True
            
            table = Table(title="GitHub Actions Workflows")
            table.add_column("File", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Triggers", style="yellow")
            table.add_column("Jobs", style="blue")
            
            for workflow in workflows:
                if 'error' in workflow:
                    table.add_row(
                        workflow['file'],
                        workflow['name'],
                        f"[red]Error: {workflow['error']}[/red]",
                        ""
                    )
                else:
                    triggers = ", ".join(workflow['triggers'])
                    jobs = ", ".join(workflow['jobs'])
                    table.add_row(workflow['file'], workflow['name'], triggers, jobs)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Failed to list workflows: {e}[/red]")
        return True
    
    def cmd_workflow_create(self, args: List[str]) -> bool:
        """Create a GitHub Actions workflow."""
        if not args:
            console.print("[red]Usage: /workflow-create <type>[/red]")
            console.print("[dim]Types: review, test, docs, release, security, all[/dim]")
            return True
        
        try:
            from .github_actions import GitHubActionsManager
            ga_manager = GitHubActionsManager(self.session.repo)
            
            workflow_type = args[0].lower()
            created = []
            
            if workflow_type == 'review':
                created.append(ga_manager.create_code_review_workflow())
            elif workflow_type == 'test':
                created.append(ga_manager.create_test_automation_workflow())
            elif workflow_type == 'docs':
                created.append(ga_manager.create_documentation_workflow())
            elif workflow_type == 'release':
                created.append(ga_manager.create_release_workflow())
            elif workflow_type == 'security':
                created.append(ga_manager.create_security_scan_workflow())
            elif workflow_type == 'all':
                created = ga_manager.create_all_workflows()
            else:
                console.print(f"[red]Unknown workflow type: {workflow_type}[/red]")
                console.print("[dim]Available: review, test, docs, release, security, all[/dim]")
                return True
            
            for workflow_path in created:
                filename = os.path.basename(workflow_path)
                console.print(f"[green]Created workflow: {filename}[/green]")
            
            if created:
                console.print("[dim]Commit and push to activate workflows on GitHub[/dim]")
            
        except Exception as e:
            console.print(f"[red]Failed to create workflow: {e}[/red]")
        return True
    
    def cmd_workflow_validate(self, args: List[str]) -> bool:
        """Validate a GitHub Actions workflow file."""
        if not args:
            console.print("[red]Usage: /workflow-validate <file>[/red]")
            return True
        
        try:
            from .github_actions import GitHubActionsManager
            ga_manager = GitHubActionsManager(self.session.repo)
            
            workflow_file = args[0]
            result = ga_manager.validate_workflow(workflow_file)
            
            if result['valid']:
                console.print(f"[green]✓ Workflow '{workflow_file}' is valid[/green]")
                
                # Show workflow details
                if 'workflow' in result:
                    workflow = result['workflow']
                    console.print(f"  Name: {workflow.get('name', 'Unnamed')}")
                    console.print(f"  Triggers: {', '.join(workflow.get('on', {}).keys())}")
                    console.print(f"  Jobs: {', '.join(workflow.get('jobs', {}).keys())}")
            else:
                console.print(f"[red]✗ Workflow '{workflow_file}' has errors:[/red]")
                
                if 'error' in result:
                    console.print(f"  {result['error']}")
                elif 'errors' in result:
                    for error in result['errors']:
                        console.print(f"  - {error}")
            
        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")
        return True
    
    def cmd_update(self, args: List[str]) -> bool:
        """Update model configurations."""
        try:
            import sys
            from pathlib import Path
            
            # Add scripts directory to path
            scripts_dir = Path(__file__).parent.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            
            from update_models import update_models_if_needed
            
            console.print("[yellow]Checking for model updates...[/yellow]")
            
            # Force update when manually triggered
            if update_models_if_needed(force=True, silent=False):
                console.print("[green]✓ Model configurations updated[/green]")
                
                # Reload the config in the agent
                if hasattr(self.session.agent, '_load_model_config'):
                    self.session.agent._load_model_config()
                    console.print("[green]✓ Reloaded model configurations[/green]")
            else:
                console.print("[yellow]Model configurations are already up to date[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Update failed: {e}[/red]")
        return True