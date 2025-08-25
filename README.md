# Agentic Coder - Production-Ready Claude Code Style Agent

A powerful, autonomous coding agent with **comprehensive Claude Code feature parity**. Features AST-enhanced code analysis, MCP integration, GitHub Actions automation, and 30+ built-in tools. Supports both **open-weight** models (Qwen, DeepSeek) and **remote APIs** (OpenAI/Anthropic).

## 🚀 Key Features

### Core Capabilities
- **Multi-Model Support**: Local HuggingFace models + OpenAI/Anthropic APIs
- **AST-Enhanced Analysis**: Tree-sitter powered code understanding (36x faster)
- **Repository Intelligence**: Aider-style context in every interaction
- **MCP Integration**: Connect to external tools via Model Context Protocol
- **GitHub Actions**: Automated CI/CD workflow generation
- **Memory Management**: Persistent sessions with context retention
- **Hooks System**: Extensible workflow automation
- **Enhanced Commands**: Both `:` and `/` prefixes (Claude Code style)

### Superior to Basic Claude Code
- **Real AST Parsing** vs regex patterns
- **Repository Mapping** with symbol extraction
- **Plan Mode** for safe exploration
- **30+ Built-in Tools** for comprehensive automation

## 📦 Installation

```bash
# Clone and install
git clone https://github.com/yourusername/agentic-coder.git
cd agentic-coder
pip install -e .

# Install with AST support
pip install -e . tree-sitter-python

# For development
pip install -e ".[dev]" pytest
```

## 🎯 Quick Start

### Interactive REPL Mode
```bash
# Start with default model
agentic-coder

# With specific model
agentic-coder --model remote-openai

# With specific repository  
agentic-coder --repo /path/to/project
```

### Command-Line Usage
```bash
# List available models
agentic-coder models

# Run with local model
agentic-coder run --model qwen2.5-coder-14b --repo . \
  --goal "Fix failing tests and add --dry-run flag"

# Run with OpenAI
export OPENAI_API_KEY="sk-..."
agentic-coder run --model remote-openai --remote-model gpt-4o --repo . \
  --goal "Refactor authentication module"

# Run with Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
agentic-coder run --model remote-anthropic --remote-model claude-3-opus --repo . \
  --goal "Add comprehensive error handling"
```

## 🛠️ Command Reference

### File Operations
```bash
:read <file>         # Read file
:write <file>        # Write file
:edit <file>         # Edit existing file
/add <file>          # Create new file
```

### Code Analysis
```bash
:analyze [file]      # AST analysis
:map                 # Repository map
:ast <file>          # Parse file AST
/grep <pattern>      # Search codebase
```

### Git Operations
```bash
:git status          # Check status
:git diff            # View changes
:git commit          # Create commit
```

### MCP Integration
```bash
:mcp                       # List MCP servers
:mcp-add <name> <cmd>      # Add MCP server
:mcp-connect <name>        # Connect to server
```

### GitHub Actions
```bash
:workflows                 # List workflows
:workflow-create <type>    # Create workflow
:workflow-validate <file>  # Validate workflow
```

### Memory & Sessions
```bash
:memory              # Show current session
:sessions            # List all sessions
:load <session_id>   # Load session
```

### Hooks
```bash
:hooks                     # List hooks
:hook add <name> <trigger> <cmd>  # Add hook
:hook enable <name>        # Enable hook
```

## 🏗️ Architecture

```
agentic_coder/
├── Core
│   ├── config.py          # Model configurations
│   ├── runtime.py         # Agent runtime with streaming
│   ├── planner.py         # Context-aware prompts
│   └── repl.py           # Interactive interface
├── Tools (30+)
│   ├── filesystem.py      # File operations
│   ├── repo_map.py        # Repository mapping
│   ├── ast_parser.py      # AST analysis
│   ├── git_tools.py       # Git integration
│   └── ...               # 25+ more tools
├── Advanced Features
│   ├── mcp.py            # Model Context Protocol
│   ├── hooks.py          # Hooks system
│   ├── memory.py         # Session persistence
│   ├── github_actions.py # CI/CD automation
│   └── commands.py       # Command processor
└── Tests
    └── test_agentic_coder.py # Comprehensive test suite
```

## 🔧 Configuration

### Environment Variables
```bash
# API Keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Optional Base URLs
export OPENAI_BASE_URL="https://api.openai.com/v1"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# Feature Flags
export AGENTIC_ALLOW_DOCS=true    # Allow doc creation
export AGENTIC_READ_EXPIRY=1800   # Read cache expiry (seconds)
```

### Model Presets
- `qwen2.5-coder-14b` - Qwen 2.5 Coder 14B
- `qwen2.5-coder-32b` - Qwen 2.5 Coder 32B  
- `deepseek-coder-v2-16b` - DeepSeek Coder V2
- `remote-openai` - OpenAI API
- `remote-anthropic` - Anthropic API

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=agentic_coder --cov-report=html

# Run specific tests
pytest tests/test_agentic_coder.py::TestMCP -v
```

## 📊 Performance

- **AST Parsing**: 36x faster than regex approaches
- **Repository Mapping**: Handles 10,000+ file codebases
- **Streaming**: Real-time response visualization
- **Context**: 800+ tokens of repository context per prompt

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure tests pass
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Inspired by Claude Code from Anthropic
- Repository mapping from Aider AI  
- Tree-sitter for AST capabilities
- Open-source community contributions

---

**Built for developers who want powerful, autonomous coding assistance**