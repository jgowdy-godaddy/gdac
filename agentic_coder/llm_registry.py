import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import ModelSpec, MODEL_PRESETS

HF_CACHE = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

def ensure_model_cached(spec: ModelSpec) -> str:
    return snapshot_download(
        repo_id=spec.id,
        revision=spec.hf_revision,
        cache_dir=HF_CACHE,
        resume_download=True,
    )

def load_model(preset: str):
    spec = MODEL_PRESETS[preset]
    if spec.remote:
        return None, None, spec
    local_path = ensure_model_cached(spec)
    tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=spec.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=spec.trust_remote_code, device_map="auto")
    return tok, model, spec