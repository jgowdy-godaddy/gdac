"""Rich-based persistent UI with proper Live/Input handling (Rich 14.x compatible)."""
from __future__ import annotations
import atexit
import threading
from queue import SimpleQueue, Empty
from time import monotonic, sleep
from typing import Callable, List, Optional, Sequence, Union

from rich.console import Console, ConsoleRenderable
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.table import Table

Renderable = Union[str, ConsoleRenderable]

class RichPersistentUI:
    def __init__(self, console: Optional[Console] = None) -> None:
        self.console: Console = console or Console()
        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="header", size=1),
            Layout(name="body", ratio=1),
            Layout(name="input", size=3),
            Layout(name="status", size=1),
        )
        self._header: Renderable = Text("")
        self._status: Renderable = Text("")
        self._input_hint: Renderable = Text("Type /help for commands")
        self._output_lines: List[Renderable] = []
        self._max_lines = 1000
        self._running = False
        self._input_cb: Optional[Callable[[str], None]] = None
        self._msg_queue: "SimpleQueue[Renderable]" = SimpleQueue()
        self._lock = threading.RLock()
        # Prompt helpers (installed when run() starts)
        self._prompt_func = None  # type: ignore
        self._prompt_lines_func = None  # type: ignore

    # ---------- Public API ----------
    def set_header(self, text: str) -> None:
        with self._lock:
            self._header = Text.from_markup(text)

    def set_status(self, text: str, hints: List[str] = None) -> None:
        with self._lock:
            if hints:
                status_text = f"{text} | {' | '.join(hints)}"
            else:
                status_text = text
            self._status = Text.from_markup(status_text)

    def set_input_hint(self, text: str) -> None:
        with self._lock:
            self._input_hint = Text.from_markup(text)

    def set_input_callback(self, cb: Callable[[str], None]) -> None:
        self._input_cb = cb

    def clear_output(self) -> None:
        with self._lock:
            self._output_lines.clear()

    def post_line(self, renderable: Renderable) -> None:
        self._msg_queue.put(renderable)

    def append_line(self, renderable: Renderable) -> None:
        with self._lock:
            self._output_lines.append(renderable)
            if len(self._output_lines) > self._max_lines:
                del self._output_lines[0:len(self._output_lines)-self._max_lines]

    def set_output(self, content) -> None:
        """Set output content (for compatibility with existing code)."""
        self.clear_output()
        self.append_line(content)

    def prompt(self, prompt: str = "") -> str:
        if self._prompt_func is None:
            return self.console.input(prompt)
        return self._prompt_func(prompt)  # type: ignore[misc]

    def prompt_lines(self, prompt: str = "... ", terminator: str = ".") -> list[str]:
        if self._prompt_lines_func is None:
            lines: list[str] = []
            while True:
                line = self.console.input(prompt)
                if line.strip() == terminator:
                    break
                lines.append(line)
            return lines
        return self._prompt_lines_func(prompt, terminator)  # type: ignore[misc]

    def start(self, input_callback: Callable[[str], None]) -> None:
        """Start the UI (for compatibility with existing code)."""
        self.set_input_callback(input_callback)
        self.run()

    def run(self) -> None:
        if self._running:
            return
        self._running = True

        @atexit.register
        def _restore_cursor() -> None:
            try:
                self.console.show_cursor(True)
            except Exception:
                pass

        self._render_all()
        
        live = Live(
            self.layout,
            console=self.console,
            screen=False,  # Don't use screen mode - it conflicts with stop/start
            auto_refresh=True,
            transient=False,
            refresh_per_second=10
        )
        live.start()
        try:
            # Ensure the UI is fully visible before any input happens
            self.console.show_cursor(False)
            live.refresh()
            sleep(0.05)  # brief first-paint delay to avoid initial prompt flash

            # install prompt helpers that safely stop/start live
            def _prompt_once(p: str = "") -> str:
                live.stop()
                try:
                    self.console.show_cursor(True)
                    return self.console.input(p)
                finally:
                    self.console.show_cursor(False)
                    live.start(refresh=True)

            def _prompt_lines(p: str = "... ", term: str = ".") -> list[str]:
                lines: list[str] = []
                while True:
                    live.stop()
                    try:
                        self.console.show_cursor(True)
                        line = self.console.input(p)
                    finally:
                        self.console.show_cursor(False)
                        live.start(refresh=True)
                    if line.strip() == term:
                        break
                    lines.append(line)
                return lines

            self._prompt_func = _prompt_once
            self._prompt_lines_func = _prompt_lines

            next_tick = monotonic() + 0.25
            while self._running:
                # Drain queued messages
                drained = 0
                while True:
                    try:
                        item = self._msg_queue.get_nowait()
                    except Empty:
                        break
                    else:
                        self.append_line(item)
                        drained += 1
                if drained:
                    self._update_body()
                    live.refresh()

                # Periodic tick
                now = monotonic()
                if now >= next_tick:
                    next_tick = now + 0.25

                # ---- READ ONE COMMAND LINE ----
                live.stop()
                try:
                    self.console.show_cursor(True)
                    cmd = self.console.input("[bold green]› [/]")
                except (EOFError, KeyboardInterrupt):
                    self._running = False
                    break
                finally:
                    self.console.show_cursor(False)
                    live.start(refresh=True)

                cmd = cmd.rstrip()
                if cmd.strip() == "":
                    self._update_input()
                    live.refresh()
                    continue

                if cmd.strip().lower() in {"q", "quit", "exit"}:
                    self._running = False
                    break

                # No echo of the command into the body (prevents stray prompts up top)
                # Invoke callback to handle the command
                if self._input_cb is not None:
                    try:
                        self._input_cb(cmd)
                    except Exception as ex:
                        self.append_line(Panel(f"[red]Error:[/] {ex!r}"))
                        self._update_body()
                        live.refresh()

                sleep(0.01)
        finally:
            try:
                self.console.show_cursor(True)
                # Exit alternate screen
                self.console.print("\033[?1049l", end="")
            except Exception:
                pass
            live.stop()
            self._running = False

    def stop(self) -> None:
        self._running = False

    # ---------- Rendering helpers ----------
    def _render_all(self) -> None:
        self._update_header()
        self._update_body()
        self._update_input()
        self._update_status()

    def _update_header(self) -> None:
        self.layout["header"].update(Align.left(self._to_renderable(self._header)))

    def _update_body(self) -> None:
        self.layout["body"].update(
            Panel(self._stacked_renderables(self._output_lines), title="Live", border_style="dim")
        )

    def _update_input(self) -> None:
        self.layout["input"].update(
            Panel(Align.left(self._to_renderable(self._input_hint)), title="Input", border_style="green")
        )

    def _update_status(self) -> None:
        self.layout["status"].update(Align.left(self._to_renderable(self._status)))

    def _to_renderable(self, r: Renderable) -> ConsoleRenderable:
        if isinstance(r, str):
            return Text.from_markup(r)
        return r  # type: ignore[return-value]

    def _stacked_renderables(self, renderables: Sequence[Renderable]) -> ConsoleRenderable:
        only_text = all(isinstance(r, (str, Text)) for r in renderables)
        if only_text:
            t = Text()
            for r in renderables:
                if isinstance(r, Text):
                    t.append(r)
                else:
                    # Process markup for strings
                    t.append(Text.from_markup(str(r)))
                t.append("\n")
            return t
        grid = Table.grid(padding=0)
        grid.expand = True
        for r in renderables:
            grid.add_row(self._to_renderable(r))
        return grid

def create_rich_persistent_ui(console: Optional[Console] = None) -> RichPersistentUI:
    return RichPersistentUI(console or Console())