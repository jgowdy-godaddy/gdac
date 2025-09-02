from __future__ import annotations
import json
import os
import shlex
import sys
import termios
import tty
import threading
import logging
from typing import Optional, List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
# from rich.prompt import Prompt  # Not using Rich Prompt due to terminal compatibility issues
from rich.syntax import Syntax
from rich.live import Live

from .runtime import Agent
from .planner import make_prompt
from .config import MODEL_PRESETS
from .commands import CommandProcessor
from .ascii_art import print_godaddy_banner
from .sessions import SessionManager
from .textual_ui import create_textual_ui
from .enhanced_ui import get_session_status, get_status_hints

# Set up history debug logging
history_logger = logging.getLogger('history_debug')
history_logger.setLevel(logging.DEBUG)
if not history_logger.handlers:
    handler = logging.FileHandler(os.path.join(os.getcwd(), 'history_debug.log'), mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    history_logger.addHandler(handler)

HELP = """
GoDaddy Agentic Coder (gdac):
Type your request and I'll auto-execute. Use / prefix for commands.

Quick commands:
  /help                     Show full help
  /models                   List available models
  /use <model>              Switch model
  /read <file>              Read file
  /write <file>             Write file
  /grep <pattern>           Search code
  /quit                     Exit

Type naturally - I'll understand and execute immediately.
"""

console = Console(legacy_windows=False, force_interactive=True, force_terminal=True)

class Session:
    def __init__(self, repo: str, model: str, debug: bool = False,
                 base_url: str = None, api_key: str = None, session_id: str = None):
        self.repo = repo
        self.model = model
        self.agent = Agent(model=model, repo=repo,
                          base_url=base_url, api_key=api_key)
        self.goal: Optional[str] = None
        self.initial_prompt: Optional[str] = None
        self.history: Optional[str] = None
        self.turn_count: int = 0  # Track conversation turns
        self.streaming: bool = True  # default ON
        self.debug = debug
        # Command history for REPL
        self.command_history: List[str] = []
        self.history_index: int = -1
        self._load_history()
        # Session management
        self.session_manager = SessionManager(repo)
        self.session_id = session_id
        self.is_resumed = session_id is not None
        # Get model info for display
        self.model_info = self._get_model_info()
        # Cancellation support
        self.cancelled = False

    @property
    def tools(self):
        return self.agent.tools

    def set_goal(self, text: str):
        self.goal = text.strip()
        
        # Create new session if not already set
        if not self.session_id:
            self.session_id = self.session_manager.create_session(self.model, self.goal)
        
        # Only include goal + context in first prompt, not in ongoing history
        self.initial_prompt = make_prompt(self.goal, self.repo)
        self.history = None  # Start fresh - will be set on first turn
        self.turn_count = 0  # Reset turn count on new goal
        
        # Save initial user message
        if self.session_id:
            self.session_manager.save_message(self.session_id, "user", self.goal)

    def ensure_goal(self):
        if not self.goal:
            raise RuntimeError("No goal set. Use :goal <text> or type free text to set it.")

    def _display_structured_response(self, response: str) -> None:
        """Display response with structured sections and progress indicators."""
        lines = response.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line_strip = line.strip()
            
            # Detect section changes
            if line_strip.upper().startswith('THINK'):
                if current_section:
                    self._display_section(current_section, section_content)
                current_section = 'THINK'
                section_content = []
                console.print("[bold blue]🤔 Thinking...[/bold blue]")
            elif line_strip.upper().startswith('PLAN'):
                if current_section:
                    self._display_section(current_section, section_content)
                current_section = 'PLAN'
                section_content = []
                console.print("[bold green]📋 Planning...[/bold green]")
            elif line_strip.startswith('ACTION:'):
                if current_section:
                    self._display_section(current_section, section_content)
                current_section = 'ACTION'
                section_content = [line]
                console.print("[bold yellow]⚡ Taking action...[/bold yellow]")
                # Don't collect more lines for ACTION, it's just one line
                self._display_section(current_section, section_content)
                return
            else:
                if current_section:
                    section_content.append(line)
        
        # Display any remaining content
        if current_section and section_content:
            self._display_section(current_section, section_content)
    
    def _display_section(self, section: str, content: list) -> None:
        """Display a section with appropriate formatting."""
        if not content:
            return
            
        content_text = '\n'.join(content).strip()
        if not content_text:
            return
            
        if section == 'THINK':
            # Show thinking process in a more readable way
            console.print(f"[dim]{content_text}[/dim]")
        elif section == 'PLAN':
            # Highlight plan steps
            for line in content:
                if line.strip():
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '*')):
                        console.print(f"[cyan]  {line.strip()}[/cyan]")
                    else:
                        console.print(f"[dim]{line}[/dim]")
        elif section == 'ACTION':
            # Action line is handled in the main flow
            pass

    def _display_stream(self, generator) -> str:
        """Render streaming text live with progress indicators."""
        buf = ""
        current_phase = None
        
        if self.debug:
            with Live(Panel("", title="MODEL (streaming)"), refresh_per_second=24, console=console) as live:
                for piece in generator:
                    if piece:
                        buf += piece
                        live.update(Panel(Syntax(buf, "markdown", theme="ansi_dark"), title="MODEL (streaming)"))
            console.rule("[bold]MODEL[/bold]")
            console.print(Syntax(buf, "markdown", theme="ansi_dark"))
        else:
            # Enhanced streaming with phase indicators
            for piece in generator:
                if piece:
                    buf += piece
                    
                    # Check for phase transitions in the streaming content
                    recent_text = buf[-200:].upper() if len(buf) > 200 else buf.upper()
                    
                    if 'THINK' in recent_text and current_phase != 'THINK':
                        console.print("[bold blue]🤔 Thinking...[/bold blue]")
                        current_phase = 'THINK'
                    elif 'PLAN' in recent_text and current_phase != 'PLAN':
                        console.print("[bold green]📋 Planning...[/bold green]")
                        current_phase = 'PLAN'
                    elif 'ACTION:' in buf and current_phase != 'ACTION':
                        console.print("[bold yellow]⚡ Ready to act...[/bold yellow]")
                        current_phase = 'ACTION'
            
            # Show the complete response with structured formatting
            console.print()
            self._display_structured_response(buf.strip())
        return buf

    def step(self) -> bool:
        """Run one turn (streaming by default)."""
        # Check for cancellation
        if self.cancelled:
            console.print("[yellow]Processing cancelled[/yellow]")
            self.cancelled = False
            return False
            
        self.ensure_goal()
        self.turn_count += 1
        
        # First turn: use initial prompt with context
        # Subsequent turns: use conversation history only
        if self.history is None:
            prompt_to_send = self.initial_prompt
            self.history = ""  # Start tracking conversation
            history_logger.info("=== HISTORY INITIALIZED ===")
            history_logger.info(f"Initial prompt length: {len(self.initial_prompt):,} chars")
            history_logger.info(f"History set to empty string")
        else:
            # Use intelligent context compaction based on model's context window
            original_history = self.history
            history_logger.info(f"=== BEFORE CONTEXT COMPACTION ===")
            history_logger.info(f"Original history length: {len(original_history):,} chars")
            
            self.history = self.agent.compact_context(self.history)
            
            history_logger.info(f"=== AFTER CONTEXT COMPACTION ===")
            history_logger.info(f"Compacted history length: {len(self.history):,} chars")
            history_logger.info(f"Compression ratio: {len(self.history)/len(original_history)*100:.1f}%")
            
            # Show compression warning if context was compacted
            if len(self.history) < len(original_history):
                original_tokens = self.agent._count_tokens(original_history)
                compressed_tokens = self.agent._count_tokens(self.history)
                if self.debug:
                    console.print(f"[yellow]Context compressed: {original_tokens:,} → {compressed_tokens:,} tokens[/yellow]")
                else:
                    console.print(f"[dim]Context compressed ({original_tokens:,} → {compressed_tokens:,} tokens)[/dim]")
            
            # For subsequent turns, send only the goal + conversation, not full context
            prompt_to_send = f"<GOAL>{self.goal}</GOAL>\n\n{self.history}"
        
        # Show that we're starting to work
        console.print(f"[dim]🤔 Thinking about: {self.goal[:50]}{'...' if len(self.goal) > 50 else ''}[/dim]")
        
        # Check for cancellation before making API call
        if self.cancelled:
            console.print("[yellow]Processing cancelled[/yellow]")
            self.cancelled = False
            return False
        
        if self.streaming:
            response = self._display_stream(self.agent.stream(prompt_to_send))
        else:
            response = self.agent._gen(prompt_to_send)
        
        # Check for cancellation after API call
        if self.cancelled:
            console.print("[yellow]Processing cancelled[/yellow]")
            self.cancelled = False
            return False
            
        if self.debug:
            console.rule("[bold]MODEL[/bold]")
            console.print(Syntax(response, "markdown", theme="ansi_dark"))
        else:
            console.print()
            # Parse and display response with better formatting
            self._display_structured_response(response)

        # Track the model's response in history (just the conversation part)
        if self.history is not None:
            history_logger.info(f"=== ADDING RESPONSE TO HISTORY ===")
            history_logger.info(f"History length before adding response: {len(self.history):,} chars")
            history_logger.info(f"Response length: {len(response):,} chars")
            
            self.history += f"\n{response}\n"
            
            history_logger.info(f"History length after adding response: {len(self.history):,} chars")
        
        # Save assistant response to session
        if self.session_id:
            self.session_manager.save_message(self.session_id, "assistant", response)
        
        action = self.agent._extract_action(response)
        if not action:
            if self.debug:
                console.print("[yellow]No ACTION detected.")
            return False

        tool_name, args = action
        
        if self.debug:
            console.rule("[bold]ACTION[/bold]")
            console.print(f"[bold]{tool_name}[/bold] {json.dumps(args, ensure_ascii=False)}")
        else:
            # Enhanced progress indicators
            console.print()
            if tool_name == "write_file":
                path = args.get('path', 'file')
                console.print(f"[yellow]📝 Writing[/yellow] [blue]{path}[/blue]")
            elif tool_name == "read_file":
                path = args.get('path', 'file')
                console.print(f"[yellow]📖 Reading[/yellow] [blue]{path}[/blue]")
            elif tool_name == "apply_patch":
                console.print(f"[yellow]🔧 Applying changes[/yellow]")
            elif tool_name == "run":
                cmd = args.get('command', 'command')[:50]
                console.print(f"[yellow]⚡ Running[/yellow] [dim]{cmd}{'...' if len(str(args.get('command', ''))) > 50 else ''}[/dim]")
            elif tool_name == "search_text":
                pattern = args.get('pattern', '')
                console.print(f"[yellow]🔍 Searching for[/yellow] [cyan]'{pattern}'[/cyan]")
            elif tool_name == "list_dir":
                path = args.get('path', '.')
                recursive = args.get('recursive', False)
                console.print(f"[yellow]📁 Listing[/yellow] [blue]{path}[/blue]{' (recursive)' if recursive else ''}")
            else:
                console.print(f"[yellow]🔧 {tool_name}[/yellow]")

        # Show execution status
        console.print(f"[dim]  → Executing...[/dim]")
        
        obs = self.tools.dispatch(tool_name, args)
        
        # Show completion status
        console.print(f"[dim]  → Complete[/dim]")
        
        if self.debug:
            console.rule("[bold]OBSERVATION[/bold]")
            if tool_name in {"apply_patch"}:
                try:
                    diff_text = args.get("diff", "")
                    if diff_text:
                        console.print(Syntax(diff_text, "diff"))
                except Exception:
                    pass
            console.print(obs if isinstance(obs, str) else repr(obs))
        else:
            # Clean output - only show relevant results
            try:
                result = json.loads(obs) if isinstance(obs, str) else obs
                if isinstance(result, dict):
                    if result.get("status") == "error":
                        console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
                    elif tool_name == "apply_patch" and result.get("status") == "ok":
                        console.print(f"[green]✓ Changes applied successfully[/green]")
                    elif tool_name == "write_file" and result.get("status") == "ok":
                        console.print(f"[green]✓ File written[/green]")
                    elif tool_name == "run" and "output" in result:
                        output = result.get("output", "").strip()
                        if output:
                            console.print(f"[dim]{output[:500]}{'...' if len(output) > 500 else ''}[/dim]")
            except:
                if self.debug:
                    console.print(obs if isinstance(obs, str) else repr(obs))

        # Track conversation without repeating context
        if self.history is not None:
            obs_str = str(obs)
            history_logger.info(f"=== ADDING OBSERVATION TO HISTORY ===")
            history_logger.info(f"History length before adding observation: {len(self.history):,} chars")
            history_logger.info(f"Observation length: {len(obs_str):,} chars")
            
            # Truncate huge observations to prevent history corruption
            MAX_OBS_SIZE = 10000  # 10KB max per observation
            if len(obs_str) > MAX_OBS_SIZE:
                truncated_obs = obs_str[:MAX_OBS_SIZE] + f"\n...[TRUNCATED - original was {len(obs_str):,} chars]"
                history_logger.warning(f"Observation truncated from {len(obs_str):,} to {len(truncated_obs):,} chars")
                self.history += f"\nOBSERVATION: {truncated_obs}\n"
            else:
                self.history += f"\nOBSERVATION: {obs_str}\n"
            
            history_logger.info(f"History length after adding observation: {len(self.history):,} chars")
        
        # Save tool usage to session
        if self.session_id:
            self.session_manager.save_message(
                self.session_id, "tool", "", 
                tool_call={"name": tool_name, "args": args},
                tool_result=obs if isinstance(obs, dict) else {"output": str(obs)}
            )
        
        return True

    def run(self, max_steps: int = 10):
        for _ in range(max_steps):
            if not self.step():
                break

    def switch_model(self, model: str):
        self.model = model
        self.agent = Agent(model=model, repo=self.repo, temperature=self.agent.temperature)
        self.model_info = self._get_model_info()
    
    def _get_model_info(self) -> dict:
        """Get detailed model information for display."""
        info = {}
        
        if self.agent.is_remote:
            # Remote model info
            info['name'] = getattr(self.agent, 'resolved_model', self.agent.remote_model)
            info['type'] = 'remote'
            info['provider'] = self.agent.remote_provider
            info['base_url'] = self.agent.remote_base
            info['base_source'] = self.agent.remote_base_source
            info['key_source'] = self.agent.remote_key_source
        else:
            # Local model info
            from .config import MODEL_PRESETS
            model_info = MODEL_PRESETS.get(self.model, None)
            info['name'] = model_info.id if model_info else self.model
            info['type'] = 'local'
            info['dtype'] = model_info.dtype if model_info else 'unknown'
            
            # Get hardware acceleration info
            try:
                import torch
                if torch.cuda.is_available():
                    info['device'] = f"CUDA ({torch.cuda.get_device_name(0)})"
                    info['memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
                elif torch.backends.mps.is_available():
                    info['device'] = "Apple Silicon (MPS)"
                else:
                    info['device'] = "CPU"
            except:
                info['device'] = "unknown"
            
            # Extract parameter count from model name if possible
            import re
            match = re.search(r'(\d+)([bB])', info['name'])
            if match:
                info['params'] = match.group(0).upper()
        
        return info
    
    def _load_history(self):
        """Load command history from file."""
        history_file = os.path.join(self.repo, ".agentic_history")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    self.command_history = [line.strip() for line in f if line.strip()]
            except Exception:
                pass
    
    def _save_history(self):
        """Save command history to file."""
        history_file = os.path.join(self.repo, ".agentic_history")
        try:
            # Keep only last 1000 commands
            history_to_save = self.command_history[-1000:]
            with open(history_file, "w", encoding="utf-8") as f:
                for cmd in history_to_save:
                    f.write(cmd + "\n")
        except Exception:
            pass
    
    def add_to_history(self, command: str):
        """Add command to history if it's not empty and different from last."""
        if command.strip() and (not self.command_history or self.command_history[-1] != command.strip()):
            self.command_history.append(command.strip())
            self._save_history()
        self.history_index = -1
    
    def resume_from_session(self, session_id: str):
        """Resume a session by loading its conversation history."""
        messages = self.session_manager.load_session(session_id)
        if not messages:
            return False
        
        self.session_id = session_id
        self.is_resumed = True
        
        # Find the original goal from the first user message
        user_messages = [m for m in messages if m.role == "user"]
        if user_messages:
            self.goal = user_messages[0].content
            self.initial_prompt = make_prompt(self.goal, self.repo)
        
        # Rebuild conversation history from messages
        history_logger.info(f"=== REBUILDING HISTORY FROM SESSION ===")
        history_logger.info(f"Session ID: {session_id}")
        history_logger.info(f"Number of messages in session: {len(messages)}")
        
        self.history = self.session_manager.rebuild_conversation_history(session_id)
        
        history_logger.info(f"Rebuilt history length: {len(self.history):,} chars")
        history_logger.info(f"History preview (first 200 chars): {self.history[:200]}")
        history_logger.info(f"History preview (last 200 chars): {self.history[-200:]}")
        
        self.turn_count = len([m for m in messages if m.role == "assistant"])
        
        return True

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
        s = input(prompt + " ")
        if s.strip() == ".":
            break
        lines.append(s)
    return "\n".join(lines).strip()

def _readline_with_history(prompt: str, history: List[str]) -> str:
    """Custom readline implementation with up/down arrow history navigation."""
    if not sys.stdin.isatty():
        # Fallback to regular input if not in a terminal
        return input(prompt)
    
    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    current_line = ""
    cursor_pos = 0
    history_index = -1  # -1 means current input, 0+ means history entry
    
    try:
        # Set terminal to raw mode to capture individual keypresses
        tty.setraw(sys.stdin.fileno())
        
        # Print prompt
        sys.stdout.write(prompt)
        sys.stdout.flush()
        
        while True:
            char = sys.stdin.read(1)
            
            # Handle Enter (\r or \n)
            if char in ('\r', '\n'):
                sys.stdout.write('\n')
                sys.stdout.flush()
                break
                
            # Handle Ctrl+C
            elif ord(char) == 3:
                raise KeyboardInterrupt
                
            # Handle Ctrl+D (EOF)
            elif ord(char) == 4:
                if not current_line:
                    raise EOFError
                
            # Handle backspace (\x7f or \x08)
            elif char in ('\x7f', '\x08'):
                if cursor_pos > 0:
                    # Remove character before cursor
                    current_line = current_line[:cursor_pos-1] + current_line[cursor_pos:]
                    cursor_pos -= 1
                    # Redraw line
                    _redraw_line(prompt, current_line, cursor_pos)
                    
            # Handle escape sequences (arrows, etc.)
            elif ord(char) == 27:  # ESC
                next_chars = sys.stdin.read(2)
                if next_chars == '[A':  # Up arrow
                    if history and history_index < len(history) - 1:
                        history_index += 1
                        current_line = history[-(history_index + 1)]
                        cursor_pos = len(current_line)
                        _redraw_line(prompt, current_line, cursor_pos)
                        
                elif next_chars == '[B':  # Down arrow
                    if history_index > 0:
                        history_index -= 1
                        current_line = history[-(history_index + 1)]
                        cursor_pos = len(current_line)
                        _redraw_line(prompt, current_line, cursor_pos)
                    elif history_index == 0:
                        history_index = -1
                        current_line = ""
                        cursor_pos = 0
                        _redraw_line(prompt, current_line, cursor_pos)
                        
                elif next_chars == '[C':  # Right arrow
                    if cursor_pos < len(current_line):
                        cursor_pos += 1
                        sys.stdout.write('\x1b[C')  # Move cursor right
                        sys.stdout.flush()
                        
                elif next_chars == '[D':  # Left arrow
                    if cursor_pos > 0:
                        cursor_pos -= 1
                        sys.stdout.write('\x1b[D')  # Move cursor left
                        sys.stdout.flush()
                        
            # Handle regular printable characters
            elif ord(char) >= 32:
                # Insert character at cursor position
                current_line = current_line[:cursor_pos] + char + current_line[cursor_pos:]
                cursor_pos += 1
                # Redraw line
                _redraw_line(prompt, current_line, cursor_pos)
                
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    return current_line

def _redraw_line(prompt: str, line: str, cursor_pos: int):
    """Redraw the entire input line with cursor at specified position."""
    # Clear current line and return to beginning
    sys.stdout.write('\r\x1b[K')
    # Print prompt and line
    sys.stdout.write(prompt + line)
    # Move cursor to correct position
    if cursor_pos < len(line):
        sys.stdout.write(f'\x1b[{len(line) - cursor_pos}D')
    sys.stdout.flush()

def _check_model_updates_async():
    """Check for model updates in a background thread."""
    def update_thread():
        try:
            import sys
            from pathlib import Path
            
            # Add scripts directory to path
            scripts_dir = Path(__file__).parent.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            
            from update_models import update_models_if_needed
            
            # Check and update silently if needed (monthly check)
            if update_models_if_needed(force=False, silent=True):
                # Silent update succeeded, models refreshed
                pass
        except Exception:
            # Silent failure is fine for background updates
            pass
    
    # Start background thread
    thread = threading.Thread(target=update_thread, daemon=True)
    thread.start()

def _show_session_picker(session_manager: SessionManager) -> Optional[str]:
    """Show interactive session picker like Claude Code."""
    sessions = session_manager.list_sessions()
    
    if not sessions:
        console.print("[yellow]No previous sessions found for this repository.[/yellow]")
        return None
    
    console.print()
    console.print("[bold]Recent sessions:[/bold]")
    console.print()
    
    for i, session in enumerate(sessions, 1):
        # Get session preview
        summary = session_manager.get_session_summary(session.session_id)
        
        # Format the session entry
        console.print(f"[bold cyan]{i:2d}.[/bold cyan] {summary}")
        console.print()
    
    console.print("[bold cyan] 0.[/bold cyan] Start new session")
    console.print()
    
    while True:
        try:
            choice = input("Select session (number): ").strip()
            if choice == "0":
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(sessions):
                return sessions[choice_num - 1].session_id
            else:
                console.print(f"[red]Invalid choice. Enter 0-{len(sessions)}[/red]")
        except (ValueError, KeyboardInterrupt):
            console.print("[red]Invalid input. Enter a number or Ctrl+C to cancel.[/red]")
        except EOFError:
            return None

def repl(repo: str = ".", model: str = None, debug: bool = False,
         base_url: str = None, api_key: str = None, resume: bool = False, 
         continue_session: bool = False):
    """Enhanced REPL with conversational interface."""
    
    # Load environment variables
    load_dotenv()
    
    # Ensure terminal is in sane state at startup
    import sys
    if sys.stdin.isatty():
        try:
            import termios
            # Get current terminal settings
            attrs = termios.tcgetattr(sys.stdin)
            # Set sane defaults
            attrs[3] = attrs[3] | termios.ECHO  # Ensure echo is on
            attrs[3] = attrs[3] | termios.ICANON  # Ensure canonical mode
            termios.tcsetattr(sys.stdin, termios.TCSANOW, attrs)
        except:
            pass
    
    # Check for model updates in the background (monthly)
    _check_model_updates_async()
    
    # Auto-detect model if not specified
    if model is None:
        if os.environ.get("ANTHROPIC_API_KEY") or api_key:
            model = "claude-opus-4.1"  # Start with best Anthropic model (Opus 4.1)
        elif os.environ.get("OPENAI_API_KEY"):
            model = "gpt-5"  # Start with best OpenAI model
        else:
            model = "qwen2.5-coder-14b"
    
    # Handle session resumption
    session_id = None
    session_manager = SessionManager(repo)
    
    if continue_session:
        # Auto-continue most recent session
        latest = session_manager.get_latest_session()
        if latest:
            session_id = latest.session_id
            session_message = f"[green]Continuing session: {latest.title}[/green]"
        else:
            session_message = "[yellow]No previous session found, starting new session.[/yellow]"
    
    elif resume:
        # Show session picker
        session_id = _show_session_picker(session_manager)
        if session_id:
            metadata = session_manager._load_metadata(session_id)
            session_message = f"[green]Resuming session: {metadata.title}[/green]"
        else:
            session_message = None
    
    # Create session
    sess = Session(repo, model, debug, base_url=base_url, api_key=api_key, session_id=session_id)
    cmd_processor = CommandProcessor(sess)
    
    # Create Textual UI with session info
    ui = create_textual_ui(session=sess)
    
    # Resume session if specified
    if session_id and not sess.resume_from_session(session_id):
        session_message = f"[red]Failed to load session {session_id}[/red]"
        # Don't return, let Rich UI show the error
    
    # Build info banner with model details
    info = sess.model_info
    banner_lines = []
    
    # Add session info to banner
    if sess.is_resumed:
        banner_lines.append(f"[dim]Session: Resumed ({sess.turn_count} previous turns)[/dim]")
    else:
        banner_lines.append(f"[dim]Session: New[/dim]")
    
    # Add context window info to banner  
    context_info = f"Context: {sess.agent.context_window:,} tokens"
    if sess.history:
        current_tokens = sess.agent._count_tokens(sess.history)
        usage_pct = (current_tokens / sess.agent.context_window) * 100
        context_info += f" (using {current_tokens:,}, {usage_pct:.1f}%)"
    banner_lines.append(f"[dim]{context_info}[/dim]")
    
    if info['type'] == 'remote':
        # Remote model display
        banner_lines.append(f"[dim]Model: {info['name']} ({info['provider']})[/dim]")
        banner_lines.append(f"[dim]API: {info['base_url']}[/dim]")
        banner_lines.append(f"[dim]Auth: API key from {info['key_source']}[/dim]")
    else:
        # Local model display
        model_line = f"[dim]Model: {info['name']}"
        if 'params' in info:
            model_line += f" {info['params']}"
        model_line += "[/dim]"
        banner_lines.append(model_line)
        
        if 'device' in info:
            device_line = f"[dim]{info['device']}"
            if 'memory' in info:
                device_line += f" ({info['memory']})"
            device_line += f", {info['dtype']}[/dim]"
            banner_lines.append(device_line)
    
    banner_text = "\n".join(banner_lines)
    
    if debug:
        banner_text += f"\n[dim]repo={repo}  streaming={'on' if sess.streaming else 'off'}  debug=on[/dim]"
    
    # Add session message if any
    if 'session_message' in locals() and session_message:
        banner_text = session_message + "\n\n" + banner_text
    
    # Set up initial status and output
    ui.set_status(get_session_status(sess), get_status_hints(sess))
    ui.set_output(banner_text)
    
    # Set command history (Rich UI only)
    if hasattr(ui, 'history'):
        ui.history = sess.command_history
    
    def handle_command(raw_input: str):
        """Handle command from Rich UI."""
        if not raw_input.strip():
            return
        
        # Add to history
        sess.add_to_history(raw_input)
        if hasattr(ui, 'history'):
            ui.history = sess.command_history
        
        # Update status before processing  
        ui.set_status(get_session_status(sess), get_status_hints(sess))
        
        # Replace the global console with a capturing one
        from io import StringIO
        from rich.console import Console as RichConsole
        
        # Create capturing console
        captured_output = StringIO()
        capturing_console = RichConsole(file=captured_output, force_interactive=True)
        
        # Temporarily replace the module-level console
        import agentic_coder.repl as repl_module
        original_console = repl_module.console
        repl_module.console = capturing_console
        
        # Also replace in commands module
        import agentic_coder.commands as commands_module
        original_cmd_console = commands_module.console
        commands_module.console = capturing_console
        
        try:
            # Show immediate feedback that we're working
            ui.add_output(f"[dim]🚀 Processing: {raw_input[:50]}{'...' if len(raw_input) > 50 else ''}[/dim]")
            
            # Process command
            should_continue = cmd_processor.process_command(raw_input)
            
            # Get captured output and display through UI
            output = captured_output.getvalue()
            if output.strip():
                ui.set_output(output)
            
            if not should_continue:
                ui.stop()
                return
                
        finally:
            # Restore original consoles
            repl_module.console = original_console
            commands_module.console = original_cmd_console
        
        # Update status after processing
        ui.set_status(get_session_status(sess), get_status_hints(sess))
    
    # Clear screen before starting UI to prevent any duplicate display
    import os
    os.system('clear' if os.name != 'nt' else 'cls')
    
    # Start the UI
    try:
        ui.start(handle_command)
    except KeyboardInterrupt:
        # Don't print here - Rich UI handles the exit message
        pass

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Path to repo")
    ap.add_argument("--model", default="qwen2.5-coder-14b", help="Model preset key")
    ap.add_argument("--resume", action="store_true", help="Show session picker to resume a previous conversation")
    ap.add_argument("--continue", action="store_true", dest="continue_session", help="Automatically continue the most recent session")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = ap.parse_args()
    repl(args.repo, args.model, args.debug, resume=args.resume, continue_session=args.continue_session)