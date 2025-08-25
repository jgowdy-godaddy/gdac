# Agentic Coder

A Claude Code–style coding agent with switchable **open-weight** models and optional **OpenAI-/Anthropic-compatible** remotes. Uses `prompts/AGENT.md` for system rules and a compact ReAct loop (THINK/PLAN/ACTION → OBSERVATION). Tools: file I/O, repo search, unified-diff patching, shell, and git.

## Install
```bash
pip install -e .
```

## List presets
```bash
agentic-coder models
```

## Run (local model)
```bash
agentic-coder run --model qwen2.5-coder-14b --repo . \
  --goal "Fix failing tests and add --dry-run flag"
```

## Run (remote OpenAI-compatible)
```bash
export OPENAI_BASE_URL="https://my-openai-proxy/v1"
export OPENAI_API_KEY="sk-..."
agentic-coder run --model remote-openai --remote-model gpt-4o-mini --repo . \
  --goal "Refactor module Y"
```

## Run (remote Anthropic-compatible)
```bash
export ANTHROPIC_BASE_URL="https://my-anthropic-proxy"
export ANTHROPIC_API_KEY="sk-ant-..."
agentic-coder run --model remote-anthropic --remote-model claude-3-5-sonnet --repo . \
  --goal "Harden error handling"
```