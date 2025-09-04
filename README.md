# GoDaddy Agentic Coder (gdac) - Production-Ready Autonomous Development Agent

A powerful, autonomous coding agent built by GoDaddy with **comprehensive development features**. Features AST-enhanced code analysis, MCP integration, GitHub Actions automation, and 30+ built-in tools. Supports both **open-weight** models (Qwen, DeepSeek) and **remote APIs** (OpenAI/Anthropic).

## 🚀 Key Features

### Core Capabilities
- **Multi-Model Support**: Local HuggingFace models + OpenAI/Anthropic APIs
- **AST-Enhanced Analysis**: Tree-sitter powered code understanding (36x faster)
- **Repository Intelligence**: Full codebase context in every interaction
- **MCP Integration**: Connect to external tools via Model Context Protocol
- **GitHub Actions**: Automated CI/CD workflow generation
- **Memory Management**: Persistent sessions with context retention
- **Hooks System**: Extensible workflow automation
- **Enhanced Commands**: Both `:` and `/` prefixes for flexible command entry

### Advanced Features
- **Real AST Parsing** vs regex patterns
- **Repository Mapping** with symbol extraction
- **Plan Mode** for safe exploration
- **30+ Built-in Tools** for comprehensive automation

## 📦 Installation

### Using pip
```bash
# Clone and install
git clone https://github.com/jgowdy-godaddy/gdac.git
cd gdac
pip install -e .

# For development
pip install -e ".[dev]" pytest
```

### Using uv
```bash
# Clone and install
git clone https://github.com/jgowdy-godaddy/gdac.git
cd gdac
uv pip install -e .

# For development
uv pip install -e ".[dev]" pytest
```

### Shell Aliases (for uv users)
Add these aliases to your shell profile (`.bashrc`, `.zshrc`, etc.) for easier usage:
```bash
# Add to ~/.bashrc, ~/.zshrc, or equivalent
alias gdac='uv run gdac'
alias gdac-models='uv run gdac models'
alias gdac-test='uv run pytest'
```

After adding aliases, reload your shell:
```bash
source ~/.zshrc  # or ~/.bashrc
```

Then use gdac normally:
```bash
gdac --model qwen2.5-coder-14b --repo .
```

## 🎯 Quick Start

### Interactive REPL Mode
```bash
# Start with default model
gdac                              # with pip or uv aliases
uv run gdac                      # with uv (no aliases)

# With specific model
gdac --model remote-openai       # with pip or uv aliases
uv run gdac --model remote-openai # with uv (no aliases)

# With specific repository  
gdac --repo /path/to/project     # with pip or uv aliases
uv run gdac --repo /path/to/project # with uv (no aliases)
```

### Command-Line Usage
```bash
# List available models
gdac models                         # with pip or uv aliases
uv run gdac models                 # with uv (no aliases)

# Run with local model
gdac run --model qwen2.5-coder-14b --repo . \
  --goal "Fix failing tests and add --dry-run flag"
# or with uv (no aliases):
uv run gdac run --model qwen2.5-coder-14b --repo . \
  --goal "Fix failing tests and add --dry-run flag"

# Run with OpenAI
export OPENAI_API_KEY="sk-..."
gdac run --model remote-openai --remote-model gpt-4o --repo . \
  --goal "Refactor authentication module"

# Run with Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
gdac run --model remote-anthropic --remote-model claude-3-opus --repo . \
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
pytest tests/                        # with pip or gdac-test alias
uv run pytest tests/                # with uv (no aliases)

# Run with coverage
pytest --cov=agentic_coder --cov-report=html        # with pip or uv aliases  
uv run pytest --cov=agentic_coder --cov-report=html # with uv (no aliases)

# Run specific tests
pytest tests/test_agentic_coder.py::TestMCP -v        # with pip or uv aliases
uv run pytest tests/test_agentic_coder.py::TestMCP -v # with uv (no aliases)
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

- Advanced repository mapping techniques  
- Tree-sitter for AST capabilities
- Open-source community contributions

---

**Built for developers who want powerful, autonomous coding assistance**