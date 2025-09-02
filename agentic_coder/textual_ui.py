"""Textual-based UI for Agentic Coder - clean terminal interface."""
from __future__ import annotations
from typing import Optional, Callable, Any
from textual.app import App, ComposeResult
from textual.widgets import Static, Input, RichLog
from textual.containers import Vertical
from textual.binding import Binding
from textual import events
from rich.panel import Panel
import asyncio
from queue import Queue, Empty
import threading
import logging
import os

# Set up debug logging for context tracking
debug_logger = logging.getLogger('agentic_debug')
debug_logger.setLevel(logging.DEBUG)

# Create a file handler that writes to debug.log in the current working directory
debug_log_path = os.path.join(os.getcwd(), 'agentic_debug.log')
debug_handler = logging.FileHandler(debug_log_path, mode='w')
debug_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(formatter)

# Add the handler to the logger
if not debug_logger.handlers:
    debug_logger.addHandler(debug_handler)


class AgenticCoderApp(App):
    """A Textual app for the Agentic Coder REPL."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    Vertical {
        height: 100vh;
        padding: 0;
        margin: 0;
    }
    
    #header {
        height: 3;
        background: $primary-darken-2;
        content-align: center middle;
        padding: 1;
    }
    
    #output {
        height: 1fr;
        background: $surface;
        border: solid $primary;
        padding: 0 1;
        margin: 0;
    }
    
    RichLog {
        scrollbar-gutter: stable;
    }
    
    #input {
        height: 3;
        margin: 0;
        border: solid $primary;
    }
    
    #status_bar {
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear", "Clear output"),
        Binding("ctrl+a", "go_home", "Home", show=False),
        Binding("ctrl+e", "go_end", "End", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]
    
    def __init__(self, input_callback: Optional[Callable[[str], Any]] = None, session=None):
        super().__init__()
        self.input_callback = input_callback
        self.session = session
        self.command_history = []
        self.history_index = -1
        self._output_queue = Queue()
        self._status_text = "Ready"
        self._should_exit = False
        self._cancel_requested = False
        self._current_task = None
        self._load_history()
        
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Vertical():
            yield Static("[bold]GoDaddy Agentic Coder (gdac)[/bold] - AI-powered coding assistant", id="header")
            yield RichLog(id="output", wrap=True, markup=True, auto_scroll=True)
            yield Input(
                placeholder="Type a command or request... (/ for commands, /help for help)",
                id="input"
            )
            yield Static("Ready", id="status_bar")
    
    def on_mount(self) -> None:
        """Called when app starts."""
        debug_logger.info("=== Textual UI Starting ===")
        debug_logger.info(f"Debug log path: {debug_log_path}")
        output = self.query_one("#output", RichLog)
        
        # Disable focus on non-input widgets
        output.can_focus = False
        status_bar = self.query_one("#status_bar", Static)
        status_bar.can_focus = False
        
        # Display ASCII art banner
        # Note: Textual's RichLog doesn't support raw terminal escape sequences for images,
        # so we can't display graphical logos within the UI window
        banner = """[bold cyan]
  ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██████╗ ██╗   ██╗
 ██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
 ██║  ███╗██║   ██║██║  ██║███████║██║  ██║██║  ██║ ╚████╔╝ 
 ██║   ██║██║   ██║██║  ██║██╔══██║██║  ██║██║  ██║  ╚██╔╝  
 ╚██████╔╝╚██████╔╝██████╔╝██║  ██║██████╔╝██████╔╝   ██║   
  ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═════╝    ╚═╝   
[/]"""
        
        # Build session info lines
        info_lines = []
        
        if self.session:
            # Session info
            if hasattr(self.session, 'is_resumed') and self.session.is_resumed:
                info_lines.append(f"[dim]Session: Resumed ({self.session.turn_count} previous turns)[/dim]")
            else:
                info_lines.append("[dim]Session: New[/dim]")
            
            # Context info
            if hasattr(self.session, 'agent') and hasattr(self.session.agent, 'context_window'):
                context_info = f"Context: {self.session.agent.context_window:,} tokens"
                if hasattr(self.session, 'history') and self.session.history:
                    try:
                        current_tokens = self.session.agent._count_tokens(self.session.history)
                        usage_pct = (current_tokens / self.session.agent.context_window) * 100
                        context_info += f" (using {current_tokens:,}, {usage_pct:.1f}%)"
                    except:
                        pass
                info_lines.append(f"[dim]{context_info}[/dim]")
            
            # Model info
            if hasattr(self.session, 'model_info'):
                info = self.session.model_info
                if info.get('type') == 'remote':
                    info_lines.append(f"[dim]Model: {info.get('name', 'Unknown')} ({info.get('provider', 'Unknown')})[/dim]")
                    info_lines.append(f"[dim]API: {info.get('base_url', 'Unknown')}[/dim]")
                    info_lines.append(f"[dim]Auth: API key from {info.get('key_source', 'Unknown')}[/dim]")
                else:
                    model_line = f"[dim]Model: {info.get('name', 'Unknown')}"
                    if 'params' in info:
                        model_line += f" {info['params']}"
                    model_line += "[/dim]"
                    info_lines.append(model_line)
                    
                    if 'device' in info:
                        device_line = f"[dim]{info['device']}"
                        if 'memory' in info:
                            device_line += f" ({info['memory']})"
                        if 'dtype' in info:
                            device_line += f", {info['dtype']}"
                        device_line += "[/dim]"
                        info_lines.append(device_line)
        
        # Build complete content
        content = f"{banner}\n\n[bold]Agentic Coder[/] - AI-powered coding assistant\n\n"
        
        if info_lines:
            content += "\n".join(info_lines) + "\n\n"
        
        content += "Type naturally and I'll understand and execute immediately.\n"
        content += "Use / prefix for commands. Type /help for available commands."
        
        output.write(Panel.fit(
            content,
            border_style="cyan"
        ))
        
        # Focus the input
        input_widget = self.query_one("#input", Input)
        self.set_focus(input_widget)
        
        # Update initial status
        self._update_status()
        
        # Start queue processor (fast refresh for real-time output)
        self.set_interval(0.1, self.process_queues)
        
        # Ensure input stays focused
        self.set_interval(0.5, self._enforce_input_focus)
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        cmd = event.value.strip()
        event.input.value = ""
        
        if not cmd:
            return
            
        # Add to history if not empty and different from last
        if cmd and (not self.command_history or self.command_history[-1] != cmd):
            self.command_history.append(cmd)
            self._save_history()
        self.history_index = len(self.command_history)
        
        # Show command in output
        output = self.query_one("#output", RichLog)
        output.write(f"[bold green]› [/]{cmd}")
        
        # Handle exit commands
        if cmd.lower() in ["/quit", "/exit", "/q"]:
            self._should_exit = True
            self.exit()
            return
        
        # Handle clear
        if cmd == "/clear":
            output.clear()
            return
        
        # Ensure input stays focused after command submission
        self.set_focus(event.input)
        
        # Pass to callback
        if self.input_callback:
            # Run callback in background to avoid blocking UI
            self._current_task = asyncio.create_task(self._handle_command_async(cmd))
    
    async def _handle_command_async(self, cmd: str) -> None:
        """Handle command in async context."""
        try:
            # Run the callback in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.input_callback, cmd)
        except asyncio.CancelledError:
            # Task was cancelled, this is expected
            pass
        except Exception as e:
            output = self.query_one("#output", RichLog)
            output.write(f"[red]Error: {e}[/]")
        finally:
            self._current_task = None
    
    def on_focus(self, event: events.Focus) -> None:
        """Ensure focus always stays on input widget."""
        input_widget = self.query_one("#input", Input)
        if event.widget != input_widget:
            self.set_focus(input_widget)
    
    def on_key(self, event: events.Key) -> None:
        """Handle key events for history navigation."""
        # Ensure input has focus for key events
        input_widget = self.query_one("#input", Input)
        self.set_focus(input_widget)
        
        # Prevent Tab from switching focus
        if event.key == "tab":
            event.stop()
            return
        
        if event.key == "up":
            if self.command_history and self.history_index > 0:
                self.history_index -= 1
                input_widget.value = self.command_history[self.history_index]
                input_widget.cursor_position = len(input_widget.value)
                event.stop()
        elif event.key == "down":
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                input_widget.value = self.command_history[self.history_index]
                input_widget.cursor_position = len(input_widget.value)
                event.stop()
            elif self.history_index == len(self.command_history) - 1:
                self.history_index = len(self.command_history)
                input_widget.value = ""
                event.stop()
    
    def add_output(self, text: str) -> None:
        """Add text to output (thread-safe)."""
        self._output_queue.put(text)
    
    def set_output(self, text: str) -> None:
        """Set/append output content (for compatibility with Rich UI)."""
        # For textual UI, we append rather than replace to maintain conversation flow
        self.add_output(text)
    
    def set_status(self, text: str, hints: str = None) -> None:
        """Update status (thread-safe). Hints parameter for Rich UI compatibility."""
        # Combine status and hints if provided
        if hints:
            self._status_text = f"{text} | {hints}"
        else:
            self._status_text = text
    
    def _enforce_input_focus(self) -> None:
        """Ensure input widget always has focus."""
        try:
            input_widget = self.query_one("#input", Input)
            if not input_widget.has_focus:
                self.set_focus(input_widget)
        except Exception:
            pass  # Ignore if widget not found
    
    def process_queues(self) -> None:
        """Process message queues."""
        # Process output queue
        output = self.query_one("#output", RichLog)
        while True:
            try:
                msg = self._output_queue.get_nowait()
                output.write(msg)
            except Empty:
                break
        
        # Update status periodically
        self._update_status()
    
    def action_quit(self) -> None:
        """Quit the application."""
        self._should_exit = True
        self.exit()
    
    def action_clear(self) -> None:
        """Clear the output."""
        output = self.query_one("#output", RichLog)
        output.clear()
    
    def action_cancel(self) -> None:
        """Cancel current processing."""
        output = self.query_one("#output", RichLog)
        
        # Set cancellation flag on session if available
        if self.session and hasattr(self.session, 'cancelled'):
            self.session.cancelled = True
            output.write("[yellow]⚠ Cancelling...[/yellow]")
        
        # Cancel any running async task
        if self._current_task and not self._current_task.done():
            self._cancel_requested = True
            self._current_task.cancel()
            self._current_task = None
            self._cancel_requested = False
            output.write("[dim]Cancelled[/dim]")
    
    def _update_status(self) -> None:
        """Update the status bar with session information."""
        status_bar = self.query_one("#status_bar", Static)
        
        status_parts = []
        
        if self.session:
            # Context usage
            if hasattr(self.session, 'agent'):
                if hasattr(self.session.agent, 'context_window'):
                    try:
                        if hasattr(self.session, 'history') and self.session.history:
                            # Make sure history is a string
                            history_text = str(self.session.history) if self.session.history else ""
                            if history_text:
                                # Debug logging - track what's happening
                                char_count = len(history_text)
                                
                                # Safety check - if history is excessively large, something is wrong
                                if char_count > 1000000:  # 1MB+ of text is suspicious
                                    debug_logger.error(f"EXTREMELY LARGE HISTORY DETECTED: {char_count:,} characters")
                                    debug_logger.error("This suggests a bug in history management - using fallback")
                                    # Use a very conservative estimate and cap it
                                    tokens = min(50000, self.session.agent.context_window // 2)
                                    usage_pct = (tokens / self.session.agent.context_window) * 100
                                    status_parts.append(f"Context: ~{tokens:,}/{self.session.agent.context_window:,} ({usage_pct:.1f}%) - CORRUPTED")
                                else:
                                    debug_logger.info(f"=== Context Calculation Debug ===")
                                    debug_logger.info(f"History type: {type(self.session.history)}")
                                    debug_logger.info(f"History character count: {char_count:,}")
                                    debug_logger.info(f"History preview (first 200 chars): {history_text[:200]}")
                                    debug_logger.info(f"History preview (last 200 chars): {history_text[-200:]}")
                                    
                                    # Use proper token counting
                                    try:
                                        # For Anthropic models, use the official tokenizer
                                        if hasattr(self.session.agent, 'anthropic_client') and self.session.agent.anthropic_client:
                                            debug_logger.info("Using Anthropic client tokenizer")
                                            try:
                                                # Use Anthropic's new beta count_tokens method
                                                token_count = self.session.agent.anthropic_client.beta.messages.count_tokens(
                                                    betas=["token-counting-2024-11-01"],
                                                    model=self.session.model,
                                                    messages=[{
                                                        "role": "user", 
                                                        "content": history_text[:100000]  # Limit to avoid huge requests
                                                    }]
                                                )
                                                tokens = token_count.input_tokens
                                                debug_logger.info(f"Anthropic tokenizer result: {tokens:,} tokens")
                                            except Exception as e:
                                                debug_logger.error(f"Anthropic beta tokenizer failed: {e}")
                                                # Fallback to existing method
                                                tokens = self.session.agent._count_tokens(history_text)
                                                debug_logger.info(f"Fallback _count_tokens result: {tokens:,} tokens")
                                        else:
                                            debug_logger.info("Using fallback _count_tokens method")
                                            # Fallback to the existing method
                                            tokens = self.session.agent._count_tokens(history_text)
                                            debug_logger.info(f"Fallback tokenizer result: {tokens:,} tokens")
                                    except Exception as e:
                                        debug_logger.error(f"Token counting failed: {e}")
                                        # Final fallback: conservative estimation
                                        tokens = max(1, char_count // 4)
                                        debug_logger.info(f"Using character estimation: {tokens:,} tokens")
                                    
                                    # Log if tokens seem excessive
                                    if tokens > 10000:
                                        debug_logger.warning(f"HIGH TOKEN COUNT: {tokens:,} tokens from {char_count:,} characters")
                                        debug_logger.warning(f"Ratio: {char_count/tokens:.2f} chars per token")
                                    
                                    # Cap at context window for display
                                    original_tokens = tokens
                                    tokens = min(tokens, self.session.agent.context_window)
                                    if original_tokens != tokens:
                                        debug_logger.warning(f"Token count capped from {original_tokens:,} to {tokens:,}")
                                    
                                    usage_pct = (tokens / self.session.agent.context_window) * 100
                                    status_parts.append(f"Context: {tokens:,}/{self.session.agent.context_window:,} ({usage_pct:.1f}%)")
                                    debug_logger.info(f"Final display: {tokens:,}/{self.session.agent.context_window:,} ({usage_pct:.1f}%)")
                                    debug_logger.info(f"=== End Context Debug ===\n")
                            else:
                                status_parts.append(f"Context: 0/{self.session.agent.context_window:,} (0.0%)")
                        else:
                            status_parts.append(f"Context: 0/{self.session.agent.context_window:,} (0.0%)")
                    except Exception as e:
                        # Include context info even if count fails
                        status_parts.append(f"Context: 0/{self.session.agent.context_window:,} tokens")
            
            # Model info
            if hasattr(self.session, 'model'):
                status_parts.append(f"Model: {self.session.model}")
            
            # Turn count
            if hasattr(self.session, 'turn_count') and self.session.turn_count > 0:
                status_parts.append(f"Turn: {self.session.turn_count}")
            
            # Streaming status
            if hasattr(self.session, 'streaming'):
                status_parts.append(f"Streaming: {'ON' if self.session.streaming else 'OFF'}")
            
            # Debug status
            if hasattr(self.session, 'debug') and self.session.debug:
                status_parts.append("Debug: ON")
        
        # Add keyboard hints at the end
        status_parts.append("Ctrl+C: Quit | Ctrl+L: Clear")
        
        # Update the status bar
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        status_bar.update(status_text)
    
    def _load_history(self) -> None:
        """Load command history from file."""
        try:
            # Get repo path from session or use current directory
            repo_path = os.getcwd()
            if self.session and hasattr(self.session, 'repo'):
                repo_path = self.session.repo
            
            history_file = os.path.join(repo_path, ".agentic_history")
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    self.command_history = [line.strip() for line in f if line.strip()]
                    self.history_index = len(self.command_history)
                debug_logger.info(f"Loaded {len(self.command_history)} commands from history")
        except Exception as e:
            debug_logger.error(f"Failed to load history: {e}")
    
    def _save_history(self) -> None:
        """Save command history to file."""
        try:
            # Get repo path from session or use current directory
            repo_path = os.getcwd()
            if self.session and hasattr(self.session, 'repo'):
                repo_path = self.session.repo
            
            history_file = os.path.join(repo_path, ".agentic_history")
            # Keep only last 1000 commands
            history_to_save = self.command_history[-1000:]
            with open(history_file, "w", encoding="utf-8") as f:
                for cmd in history_to_save:
                    f.write(cmd + "\n")
            debug_logger.info(f"Saved {len(history_to_save)} commands to history")
        except Exception as e:
            debug_logger.error(f"Failed to save history: {e}")
    
    def action_go_home(self) -> None:
        """Move cursor to beginning of line (Ctrl+A)."""
        input_widget = self.query_one("#input", Input)
        input_widget.cursor_position = 0
    
    def action_go_end(self) -> None:
        """Move cursor to end of line (Ctrl+E)."""
        input_widget = self.query_one("#input", Input)
        input_widget.cursor_position = len(input_widget.value)


class TextualUI:
    """Wrapper for Textual UI that works with the REPL."""
    
    def __init__(self, session=None):
        self.app = None
        self.session = session
        self.input_callback = None
        self._message_queue = Queue()
        self._app_thread = None
        
    def start(self, input_callback: Callable[[str], None]) -> None:
        """Start the Textual UI - must be called from main thread."""
        self.input_callback = self._wrap_callback(input_callback)
        self.app = AgenticCoderApp(input_callback=self.input_callback, session=self.session)
        
        # Run the app (this blocks until the app exits)
        self.app.run()
    
    def _wrap_callback(self, callback: Callable[[str], None]) -> Callable[[str], None]:
        """Wrap the callback to handle command processing."""
        def wrapped(cmd: str):
            try:
                # Process the command
                result = callback(cmd)
                # Check if we should exit
                if result is False and self.app:
                    self.app._should_exit = True
            except Exception as e:
                if self.app:
                    self.app.add_output(f"[red]Error: {e}[/]")
        return wrapped
    
    def stop(self) -> None:
        """Stop the UI."""
        if self.app and not self.app._should_exit:
            self.app._should_exit = True
            self.app.exit()
    
    # Compatibility methods for REPL integration
    def add_output(self, text: str) -> None:
        """Add output text."""
        if self.app:
            self.app.add_output(text)
    
    def set_output(self, text: str) -> None:
        """Add output text (alias for add_output for Rich UI compatibility)."""
        if self.app:
            self.app.add_output(text)
    
    def set_status(self, status: str, hints: list = None) -> None:
        """Update status."""
        if self.app:
            # Instead of using the passed status, trigger an update
            # to pull fresh info from the session
            self.app._update_status()
    
    def set_header(self, text: str) -> None:
        """Compatibility method - header is fixed in Textual."""
        pass
    
    def set_input_callback(self, callback) -> None:
        """Set the input callback."""
        self.input_callback = callback
    

def create_textual_ui(session=None) -> TextualUI:
    """Create a Textual UI instance."""
    return TextualUI(session=session)


if __name__ == "__main__":
    # Test the UI standalone
    def test_callback(cmd: str):
        print(f"Got command: {cmd}")
        return None
    
    ui = TextualUI()
    ui.start(test_callback)