from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelSpec:
    id: str
    hf_revision: Optional[str] = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    chat_template: Optional[str] = None
    remote: bool = False
    api_type: Optional[str] = None
    context_window: int = 8192  # Default context window in tokens

MODEL_PRESETS = {
    # Qwen models - 128k context window
    "qwen2.5-coder-14b": ModelSpec(id="Qwen/Qwen2.5-Coder-14B", dtype="bfloat16", context_window=131072),
    "qwen2.5-coder-32b": ModelSpec(id="Qwen/Qwen2.5-Coder-32B-Instruct", dtype="bfloat16", context_window=131072),
    "qwen3-coder-30b-a3b": ModelSpec(id="Qwen/Qwen3-Coder-30B-A3B-Instruct", dtype="bfloat16", context_window=131072),
    # DeepSeek models - 128k for v2, 16k for v1
    "deepseek-coder-v2-16b": ModelSpec(id="deepseek-ai/DeepSeek-Coder-V2-Instruct-0724", dtype="bfloat16", context_window=131072),
    "deepseek-coder-33b": ModelSpec(id="deepseek-ai/deepseek-coder-33b-instruct", dtype="bfloat16", context_window=16384),
    # Llama models - 128k context
    "llama-3.1-8b": ModelSpec(id="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16", context_window=131072),
    # Remote API models
    "remote-openai": ModelSpec(id="REMOTE", remote=True, api_type="openai", context_window=128000),
    "remote-anthropic": ModelSpec(id="REMOTE", remote=True, api_type="anthropic", context_window=200000)
}