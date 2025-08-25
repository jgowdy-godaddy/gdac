# Repository: agentic-coder
# Layout
# ├─ pyproject.toml
# ├─ README.md
# ├─ prompts/AGENT.md
# └─ agentic_coder/
#    ├─ __init__.py
#    ├─ config.py
#    ├─ llm_registry.py
#    ├─ planner.py
#    ├─ runtime.py
#    ├─ tools/__init__.py
#    ├─ tools/filesystem.py
#    ├─ tools/patch.py
#    ├─ tools/shell.py
#    ├─ tools/git_tools.py
#    └─ cli.py

# -------------------------
# pyproject.toml
# -------------------------
[build-system]
requires = ["setuptools>=67", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agentic-coder"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
  "transformers>=4.44.0",
  "tokenizers>=0.19.0",
  "accelerate>=0.33.0",
  "huggingface_hub>=0.24.0",
  "datasets>=2.19.0",
  "rich>=13.7.0",
  "typer>=0.12.3",
  "unidiff>=0.7.5",
  "python-dotenv>=1.0.1",
  "httpx>=0.27.0",
]

[project.scripts]
agentic-coder = "agentic_coder.cli:app"

# -------------------------
# agentic_coder/config.py
# -------------------------
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelSpec:
    id: str                 # Hugging Face repo id or remote alias
    hf_revision: Optional[str] = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    chat_template: Optional[str] = None
    remote: bool = False     # True for OpenAI/Anthropic API use
    api_type: Optional[str] = None  # "openai" or "anthropic"

# Curated, non-gated open-weight models + remote options
MODEL_PRESETS = {
    # Qwen2.5 Coder instruct variants (open)
    "qwen2.5-coder-14b": ModelSpec(
        id="Qwen/Qwen2.5-Coder-14B",
        dtype="bfloat16",
    ),
    "qwen2.5-coder-32b": ModelSpec(
        id="Qwen/Qwen2.5-Coder-32B-Instruct",
        dtype="bfloat16",
    ),
    # Qwen3 Coder (MoE A3B) — open-weight
    "qwen3-coder-30b-a3b": ModelSpec(
        id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        dtype="bfloat16",
    ),
    # DeepSeek Coder V2 (MoE active ~16B)
    "deepseek-coder-v2-16b": ModelSpec(
        id="deepseek-ai/DeepSeek-Coder-V2-Instruct-0724",
        dtype="bfloat16",
    ),
    # DeepSeek dense 33B instruct
    "deepseek-coder-33b": ModelSpec(
        id="deepseek-ai/deepseek-coder-33b-instruct",
        dtype="bfloat16",
    ),
    # Llama 3.1 8B Instruct as a lightweight baseline
    "llama-3.1-8b": ModelSpec(
        id="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
    ),
    # Remote API presets (OpenAI-/Anthropic-compatible). These do not download weights.
    # Provide base URL and API key via env vars indicated below, or CLI options.
    "remote-openai": ModelSpec(
        id="REMOTE",
        dtype="auto",
        trust_remote_code=False,
        chat_template=None,
    ),
    "remote-anthropic": ModelSpec(
        id="REMOTE",
        dtype="auto",
        trust_remote_code=False,
        chat_template=None,
    ),
}

# -------------------------
# agentic_coder/llm_registry.py
# -------------------------
from __future__ import annotations
import os
from typing import Tuple
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import ModelSpec, MODEL_PRESETS
import httpx

HF_CACHE = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

def ensure_model_cached(spec: ModelSpec) -> str:
    local_dir = snapshot_download(
        repo_id=spec.id,
        revision=spec.hf_revision,
        local_files_only=False,
        ignore_patterns=["*.md5", "*.lock"],
        cache_dir=HF_CACHE,
        allow_patterns=None,
        resume_download=True,
    )
    return local_dir


def load_model(preset: str):
    if preset not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise KeyError(f"Unknown model preset: {preset}. Available: {available}")
    spec = MODEL_PRESETS[preset]
    if spec.remote:
        return None, None, spec
    local_path = ensure_model_cached(spec)
    tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=spec.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        trust_remote_code=spec.trust_remote_code,
        torch_dtype="auto" if spec.dtype == "auto" else None,
        device_map="auto",
    )
    return tok, model, spec

# -------------------------
# agentic_coder/runtime.py
# -------------------------
from __future__ import annotations
import json
import os
import httpx
from typing import Dict, Any, Optional
import os
from rich.console import Console
from .llm_registry import load_model
from .planner import make_prompt
from .tools import ToolRegistry
from .config import MODEL_PRESETS

console = Console()

class Agent:
    def __init__(self, model_preset: str, repo: str,
                 remote_model: str | None = None,
                 remote_base_url: str | None = None,
                 remote_api_key: str | None = None):
        self.repo = repo
        self.tok, self.model, self.spec = load_model(model_preset)
        self.tools = ToolRegistry(repo)
        self.remote_provider = None
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
                temperature=0,
                max_tokens=max_new_tokens,
            )
            return resp.choices[0].message.content
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=self.remote_key, base_url=self.remote_base)
            msg = client.messages.create(
                model=self.remote_model,
                max_tokens=max_new_tokens,
                temperature=0,
                system="Follow AGENT.md protocol strictly.",
                messages=[{"role":"user","content":prompt}],
            )
            parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "".join(parts) or str(msg)

    def _remote_gen(self, prompt: str) -> str:
        base_url = os.environ.get("AGENTIC_BASE_URL")
        api_key = os.environ.get("AGENTIC_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError("AGENTIC_BASE_URL and AGENTIC_API_KEY must be set for remote models")

        if self.spec.api_type == "openai":
            payload = {
                "model": "default",
                "messages": [
                    {"role": "system", "content": "Follow AGENT.md strictly."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1024,
            }
            headers = {"Authorization": f"Bearer {api_key}"}
            r = httpx.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        elif self.spec.api_type == "anthropic":
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "system": "Follow AGENT.md strictly.",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {"x-api-key": api_key}
            r = httpx.post(f"{base_url}/v1/messages", json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        else:
            raise RuntimeError(f"Unsupported remote api_type: {self.spec.api_type}")

    def run(self, goal: str, max_iters: int = 20):
        history = make_prompt(goal, self.repo)
        for step in range(max_iters):
            console.rule(f"Step {step+1}")
            response = self._gen(history)
            action = self._extract_action(response)
            if not action:
                console.print("[yellow]No ACTION detected. Stopping.")
                break
            tool_name, args = action
            console.print(f"[bold]ACTION[/bold]: {tool_name} {args}")
            obs = self.tools.dispatch(tool_name, args)
            console.print("[bold]OBSERVATION[/bold]:", (obs[:2000] + "…") if isinstance(obs, str) and len(obs) > 2000 else obs)
            history += f"\nOBSERVATION: {obs}\n"
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
