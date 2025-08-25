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

MODEL_PRESETS = {
    "qwen2.5-coder-14b": ModelSpec(id="Qwen/Qwen2.5-Coder-14B", dtype="bfloat16"),
    "qwen2.5-coder-32b": ModelSpec(id="Qwen/Qwen2.5-Coder-32B-Instruct", dtype="bfloat16"),
    "qwen3-coder-30b-a3b": ModelSpec(id="Qwen/Qwen3-Coder-30B-A3B-Instruct", dtype="bfloat16"),
    "deepseek-coder-v2-16b": ModelSpec(id="deepseek-ai/DeepSeek-Coder-V2-Instruct-0724", dtype="bfloat16"),
    "deepseek-coder-33b": ModelSpec(id="deepseek-ai/deepseek-coder-33b-instruct", dtype="bfloat16"),
    "llama-3.1-8b": ModelSpec(id="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16"),
    "remote-openai": ModelSpec(id="REMOTE", remote=True, api_type="openai"),
    "remote-anthropic": ModelSpec(id="REMOTE", remote=True, api_type="anthropic"),
}