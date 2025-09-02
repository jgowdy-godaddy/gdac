"""Enhanced UI components for Claude Code-style interface."""

from __future__ import annotations
import os
import sys
import termios
import tty
from typing import List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align

class StatusBar:
    """Status bar for displaying hints and state information."""
    
    def __init__(self, console: Console):
        self.console = console
        self._status_text = ""
        self._hints = []
        
    def set_status(self, text: str):
        """Set the main status text."""
        self._status_text = text
        
    def set_hints(self, hints: List[str]):
        """Set the hint text list."""
        self._hints = hints
        
    def render(self) -> str:
        """Render the status bar."""
        # Get terminal width
        try:
            terminal_width = os.get_terminal_size().columns
        except:
            terminal_width = 80
            
        # Create status line
        left_text = self._status_text
        right_text = " | ".join(self._hints) if self._hints else ""
        
        # Calculate spacing
        available_space = terminal_width - len(left_text) - len(right_text) - 2
        if available_space < 1:
            # Truncate if too long
            max_right = terminal_width - len(left_text) - 5
            if max_right > 10:
                right_text = right_text[:max_right] + "..."
            else:
                right_text = ""
            available_space = terminal_width - len(left_text) - len(right_text) - 2
            
        spacing = " " * max(1, available_space)
        
        # Create status bar content
        status_line = f" {left_text}{spacing}{right_text} "
        
        # Create status line without panel for now (simpler)
        return Text(status_line, style="dim")

class BoxedPrompt:
    """Boxed prompt input that auto-expands for multiple lines."""
    
    def __init__(self, console: Console, prompt_text: str = "> "):
        self.console = console
        self.prompt_text = prompt_text
        self.lines = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        
    def _get_terminal_width(self) -> int:
        """Get terminal width with fallback."""
        try:
            return os.get_terminal_size().columns
        except:
            return 80
            
    def _render_box(self):
        """Render the input box with current content."""
        terminal_width = self._get_terminal_width()
        box_width = min(terminal_width - 4, 120)  # Max width like Claude Code
        
        # Calculate content area (subtract borders and padding)
        content_width = box_width - 4
        
        # Prepare lines for display
        display_lines = []
        for i, line in enumerate(self.lines):
            if i == 0:
                # First line has prompt
                prompt_len = len(self.prompt_text)
                available_width = content_width - prompt_len
                
                if len(line) <= available_width:
                    display_lines.append(f"{self.prompt_text}{line}")
                else:
                    # Wrap long lines properly at word boundaries when possible
                    words = line.split(' ')
                    current_line = f"{self.prompt_text}"
                    current_length = prompt_len
                    
                    for word in words:
                        if current_length + len(word) + 1 <= content_width:
                            if current_length > prompt_len:
                                current_line += " "
                                current_length += 1
                            current_line += word
                            current_length += len(word)
                        else:
                            display_lines.append(current_line)
                            current_line = " " * prompt_len + word
                            current_length = prompt_len + len(word)
                    
                    if current_line.strip():
                        display_lines.append(current_line)
            else:
                # Continuation lines
                if len(line) <= content_width - 2:
                    display_lines.append(f"  {line}")
                else:
                    # Wrap long continuation lines at word boundaries
                    words = line.split(' ')
                    current_line = "  "
                    current_length = 2
                    
                    for word in words:
                        if current_length + len(word) + 1 <= content_width:
                            if current_length > 2:
                                current_line += " "
                                current_length += 1
                            current_line += word
                            current_length += len(word)
                        else:
                            display_lines.append(current_line)
                            current_line = "  " + word
                            current_length = 2 + len(word)
                    
                    if current_line.strip():
                        display_lines.append(current_line)
        
        # Ensure at least one line
        if not display_lines:
            display_lines = [self.prompt_text]
            
        # Create the box content
        box_content = "\n".join(display_lines)
        
        # Create panel
        return Panel(
            box_content,
            width=box_width,
            padding=(0, 1),
            title="[dim]Input[/dim]",
            title_align="left"
        )
    
    def _handle_key(self, key: str) -> bool:
        """Handle a single keypress. Returns True to continue, False to submit."""
        current_line = self.lines[self.cursor_line]
        
        if key == '\r' or key == '\n':
            # Enter key
            if len(self.lines) == 1 and not current_line.strip():
                # Empty input, don't submit
                return True
                
            # Check if this is a multi-line continuation
            if current_line.rstrip().endswith('\\') or current_line.strip() in ['def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:']:
                # Multi-line mode
                self.lines.insert(self.cursor_line + 1, "")
                self.cursor_line += 1
                self.cursor_col = 0
                return True
            else:
                # Submit
                return False
                
        elif key == '\x7f' or key == '\x08':  # Backspace
            if self.cursor_col > 0:
                # Delete character before cursor
                new_line = current_line[:self.cursor_col-1] + current_line[self.cursor_col:]
                self.lines[self.cursor_line] = new_line
                self.cursor_col -= 1
            elif self.cursor_line > 0:
                # Join with previous line
                prev_line = self.lines[self.cursor_line - 1]
                self.cursor_col = len(prev_line)
                self.lines[self.cursor_line - 1] = prev_line + current_line
                self.lines.pop(self.cursor_line)
                self.cursor_line -= 1
                
        elif key == '\x04':  # Ctrl+D
            if not current_line:
                raise EOFError
                
        elif key == '\x03':  # Ctrl+C
            raise KeyboardInterrupt
            
        elif ord(key) >= 32:  # Printable characters
            # Insert character at cursor
            new_line = current_line[:self.cursor_col] + key + current_line[self.cursor_col:]
            self.lines[self.cursor_line] = new_line
            self.cursor_col += 1
            
        # Handle escape sequences (arrow keys, etc.)
        elif ord(key) == 27:  # ESC
            return True  # Ignore for now
            
        return True
        
    def get_input(self, history: List[str] = None) -> str:
        """Get input with boxed prompt interface."""
        if not sys.stdin.isatty():
            # Fallback for non-terminal
            return input(self.prompt_text)
        
        # Simpler implementation: show the box, then use existing readline
        self.lines = [""]
        
        # Render the input box
        box_panel = self._render_box()
        self.console.print(box_panel)
        
        # Use built-in input for simplicity and reliability
        # The visual box provides the Claude Code-like appearance
        try:
            return input()  # Empty prompt since the box shows the prompt
        except (EOFError, KeyboardInterrupt):
            raise

def create_enhanced_ui_components(console: Console) -> Tuple[BoxedPrompt, StatusBar]:
    """Create enhanced UI components."""
    prompt = BoxedPrompt(console, "> ")
    status_bar = StatusBar(console)
    return prompt, status_bar

def get_session_status(session) -> str:
    """Get session status text for status bar."""
    if hasattr(session, 'history') and session.history:
        tokens = session.agent._count_tokens(session.history)
        usage_pct = (tokens / session.agent.context_window) * 100
        return f"Context: {tokens:,}/{session.agent.context_window:,} tokens ({usage_pct:.1f}%)"
    else:
        return f"Context: 0/{session.agent.context_window:,} tokens (0.0%)"

def get_status_hints(session) -> List[str]:
    """Get hint text for status bar."""
    hints = []
    
    # Model info
    if hasattr(session, 'model'):
        hints.append(f"Model: {session.model}")
        
    # Session info
    if hasattr(session, 'turn_count') and session.turn_count > 0:
        hints.append(f"Turn: {session.turn_count}")
        
    # Special hints
    if hasattr(session, 'streaming') and session.streaming:
        hints.append("Streaming: ON")
        
    if hasattr(session, 'debug') and session.debug:
        hints.append("Debug: ON")
        
    # Commands hint
    hints.append("Press Tab for commands")
    
    return hints