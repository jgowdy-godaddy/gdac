from __future__ import annotations
import json
import os
from typing import Dict, Any, Optional, Iterator

from rich.console import Console

from .llm_registry import load_model
from .planner import make_prompt
from .tools import ToolRegistry
from .config import MODEL_PRESETS

# Local streaming (HF)
from transformers import TextIteratorStreamer
import threading

console = Console()

class Agent:
    def __init__(self, model_preset: str, repo: str,
                 remote_model: str | None = None,
                 remote_base_url: str | None = None,
                 remote_api_key: str | None = None,
                 temperature: float = 0.0):
        self.repo = repo
        self.tok, self.model, self.spec = load_model(model_preset)
        self.tools = ToolRegistry(repo)

        self.remote_provider = None
        self.remote_model = None
        self.remote_base = None
        self.remote_key = None

        if self.model is None:
            # remote mode
            self.remote_provider = "openai" if model_preset == "remote-openai" else "anthropic"
            self.remote_model = remote_model or os.environ.get("AGENTIC_REMOTE_MODEL", "gpt-4o-mini")
            if self.remote_provider == "openai":
                self.remote_base = remote_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                self.remote_key = remote_api_key or os.environ.get("OPENAI_API_KEY")
            else:
                self.remote_base = remote_base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
                self.remote_key = remote_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.remote_key:
                raise RuntimeError("Missing API key for remote provider")
        self.temperature = max(0.0, float(temperature))
        self.stop = ["\nACTION:", "\nOBSERVATION:", "</SYSTEM>"]

    # -------- non-stream (kept for compatibility) --------

    def _gen(self, prompt: str, max_new_tokens: int = 1024) -> str:
        if self.model is None:
            return self._gen_remote(prompt, max_new_tokens)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def _gen_remote(self, prompt: str, max_new_tokens: int) -> str:
        if self.remote_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.remote_key, base_url=self.remote_base)
            resp = client.chat.completions.create(
                model=self.remote_model,
                messages=[{"role":"system","content":"Follow AGENT.md protocol strictly."},
                          {"role":"user","content":prompt}],
                temperature=self.temperature,
                max_tokens=max_new_tokens,
                stop=self.stop,
            )
            return resp.choices[0].message.content
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=self.remote_key, base_url=self.remote_base)
            msg = client.messages.create(
                model=self.remote_model,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                system="Follow AGENT.md protocol strictly.",
                messages=[{"role":"user","content":prompt}],
                stop_sequences=self.stop,
            )
            parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "".join(parts) or str(msg)

    # -------- streaming APIs --------
    def stream(self, prompt: str, max_new_tokens: int = 1024) -> Iterator[str]:
        """
        Yields text chunks as they arrive. Works with:
        - Local HF models (TextIteratorStreamer)
        - OpenAI-compatible chat completions (stream=True)
        - Anthropic-compatible messages streaming
        """
        if self.model is None:
            if self.remote_provider == "openai":
                yield from self._stream_openai(prompt, max_new_tokens)
            else:
                yield from self._stream_anthropic(prompt, max_new_tokens)
            return

        # Local HF streaming with TextIteratorStreamer
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_special_tokens=True, skip_prompt=True)
        gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.daemon = True
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()

    def _stream_openai(self, prompt: str, max_new_tokens: int) -> Iterator[str]:
        from openai import OpenAI
        client = OpenAI(api_key=self.remote_key, base_url=self.remote_base)
        # Chat Completions streaming
        stream = client.chat.completions.create(
            model=self.remote_model,
            messages=[{"role": "system", "content": "Follow AGENT.md protocol strictly."},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_new_tokens,
            stream=True,
            stop=self.stop,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    yield piece
            except Exception:
                # ignore control frames
                continue

    def _stream_anthropic(self, prompt: str, max_new_tokens: int) -> Iterator[str]:
        import anthropic
        client = anthropic.Anthropic(api_key=self.remote_key, base_url=self.remote_base)
        with client.messages.stream(
            model=self.remote_model,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            system="Follow AGENT.md protocol strictly.",
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=self.stop,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text

    # -------- agent loop (unchanged) --------
    def run(self, goal: str, max_iters: int = 20):
        # Always rebuild the head so plan-mode instructions reflect current state
        history = make_prompt(goal, self.repo)
        for step in range(max_iters):
            console.rule(f"Step {step+1}")
            # refresh plan header each turn
            base = make_prompt(goal, self.repo)
            # keep only the tail after the first header
            tail = history.split("</CONTEXT>", 1)[-1] if "</CONTEXT>" in history else ""
            prompt = base + tail
            response = self._gen(prompt)
            action = self._extract_action(response)
            if not action:
                console.print("[yellow]No ACTION detected. Stopping.")
                break
            tool_name, args = action
            console.print(f"[bold]ACTION[/bold]: {tool_name} {args}")
            obs = self.tools.dispatch(tool_name, args)
            console.print("[bold]OBSERVATION[/bold]:", (obs[:2000] + "…") if isinstance(obs, str) and len(obs) > 2000 else obs)
            history = prompt + f"\nOBSERVATION: {obs}\n"
            if self.tools.goal_satisfied(goal):
                console.print("[green]Goal satisfied. Exiting.")
                break

    @staticmethod
    def _extract_action(text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        for line in text.splitlines():
            if line.startswith("ACTION:"):
                try:
                    name_and_json = line[len("ACTION:"):].strip()
                    parts = name_and_json.split(" ", 1)
                    tool = parts[0].strip()
                    payload = json.loads(parts[1]) if len(parts) > 1 else {}
                    return tool, payload
                except Exception:
                    continue
        return None