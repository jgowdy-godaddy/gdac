from __future__ import annotations
import json
import os
import uuid
import time
import logging
from typing import Dict, Any, Optional, Iterator
from pathlib import Path

from rich.console import Console

from .llm_registry import load_model
from .planner import make_prompt
from .tools import ToolRegistry
from .config import MODEL_PRESETS

# Local streaming (HF)
from transformers import TextIteratorStreamer
import threading

console = Console()

# Set up context compaction debug logging  
compact_logger = logging.getLogger('compact_debug')
compact_logger.setLevel(logging.DEBUG)
if not compact_logger.handlers:
    handler = logging.FileHandler(os.path.join(os.getcwd(), 'compact_debug.log'), mode='w')
    handler.setLevel(logging.DEBUG) 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    compact_logger.addHandler(handler)

class Agent:
    def __init__(self, model: str, repo: str,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 temperature: float = 0.0):
        self.repo = repo
        self.tools = ToolRegistry(repo)
        self.context_window = 8192  # Default, will be updated based on model
        self.anthropic_client = None
        self.openai_client = None
        
        # Detect if this is a local or remote model
        self.is_local = False
        self.is_remote = False
        
        # Check if it's a known local model preset
        if model in MODEL_PRESETS and not MODEL_PRESETS[model].remote:
            self.is_local = True
            self.tok, self.model, self.spec = load_model(model)
        else:
            # It's a remote model
            self.is_remote = True
            self.tok = None
            self.model = None
            self.spec = None

        self.remote_provider = None
        self.remote_model = None
        self.remote_base = None
        self.remote_key = None
        self.remote_key_source = None
        self.remote_base_source = None
        self.resolved_model = None

        if self.is_remote:
            # Detect provider based on model name or API key
            self._load_model_config()
            
            # Determine provider from model name patterns
            model_lower = model.lower()
            if 'claude' in model_lower or 'anthropic' in model_lower:
                self.remote_provider = "anthropic"
            elif 'gpt' in model_lower or 'openai' in model_lower:
                self.remote_provider = "openai"
            elif api_key and not base_url:
                # Try to detect from API key if no base URL specified
                if os.environ.get("ANTHROPIC_API_KEY") == api_key:
                    self.remote_provider = "anthropic"
                elif os.environ.get("OPENAI_API_KEY") == api_key:
                    self.remote_provider = "openai"
                else:
                    # Default based on available keys
                    if os.environ.get("ANTHROPIC_API_KEY"):
                        self.remote_provider = "anthropic"
                    else:
                        self.remote_provider = "openai"
            else:
                # Default to anthropic if we have the key, otherwise openai
                if os.environ.get("ANTHROPIC_API_KEY"):
                    self.remote_provider = "anthropic"
                else:
                    self.remote_provider = "openai"
            
            # Use the model name directly
            self.remote_model = model
            
            # Resolve model aliases to actual model IDs
            self.resolved_model = self._resolve_model_alias(self.remote_model)
            # Get failover chain
            self.failover_chain = self._get_failover_chain(self.resolved_model)
            self.current_failover_index = 0
            # Get context window from config
            self._update_context_window()
            
            if self.remote_provider == "openai":
                # Track base URL source
                if base_url:
                    self.remote_base = base_url
                    self.remote_base_source = "CLI"
                elif os.environ.get("OPENAI_BASE_URL"):
                    self.remote_base = os.environ.get("OPENAI_BASE_URL")
                    self.remote_base_source = "env"
                else:
                    self.remote_base = "https://api.openai.com/v1"
                    self.remote_base_source = "default"
                
                # Track API key source
                if api_key:
                    self.remote_key = api_key
                    self.remote_key_source = "CLI"
                elif os.environ.get("OPENAI_API_KEY"):
                    self.remote_key = os.environ.get("OPENAI_API_KEY")
                    self.remote_key_source = "env"
            else:
                # Track base URL source
                if base_url:
                    self.remote_base = base_url
                    self.remote_base_source = "CLI"
                elif os.environ.get("ANTHROPIC_BASE_URL"):
                    self.remote_base = os.environ.get("ANTHROPIC_BASE_URL")
                    self.remote_base_source = "env"
                else:
                    self.remote_base = "https://api.anthropic.com"
                    self.remote_base_source = "default"
                
                # Track API key source
                if api_key:
                    self.remote_key = api_key
                    self.remote_key_source = "CLI"
                elif os.environ.get("ANTHROPIC_API_KEY"):
                    self.remote_key = os.environ.get("ANTHROPIC_API_KEY")
                    self.remote_key_source = "env"
            
            if not self.remote_key:
                raise RuntimeError("Missing API key for remote provider")
            
            # Initialize SDK clients
            self._init_sdk_clients()
        elif self.is_local:
            # For local models
            self.resolved_model = None
            self.failover_chain = []
            self.current_failover_index = 0
            # Set context window from spec
            if self.spec:
                self.context_window = self.spec.context_window
            
        self.temperature = max(0.0, float(temperature))
        self.stop = ["\nOBSERVATION:", "</SYSTEM>"]
    
    def _init_sdk_clients(self):
        """Initialize SDK clients for remote providers."""
        if self.remote_provider == "anthropic":
            import anthropic
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.remote_key,
                base_url=self.remote_base if self.remote_base != "https://api.anthropic.com" else None
            )
        elif self.remote_provider == "openai":
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=self.remote_key,
                base_url=self.remote_base
            )
    
    def _get_system_prompt(self, include_agent_instructions=True):
        """Generate system prompt INCLUDING agent instructions, not just tools."""
        import os, json
        
        # Load AGENT.md into system prompt to avoid duplicating in user messages
        agent_instructions = ""
        if include_agent_instructions:
            try:
                # Check for custom instructions first
                for filename in ["AGENTS.md", "CLAUDE.md"]:
                    filepath = os.path.join(self.repo, filename)
                    if os.path.exists(filepath):
                        with open(filepath, "r") as f:
                            agent_instructions = f.read() + "\n\n"
                        break
                
                # Fallback to default AGENT.md
                if not agent_instructions:
                    agent_md_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "AGENT.md")
                    with open(agent_md_path, "r") as f:
                        agent_instructions = f.read() + "\n\n"
            except:
                pass
        
        # Check for plan mode
        plan_mode_instruction = ""
        plan_path = os.path.join(self.repo, ".agentic_plan.json")
        try:
            if os.path.exists(plan_path):
                with open(plan_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("active"):
                    plan_mode_instruction = "\n<SYSTEM-PLAN-MODE>\nPlan mode is active. Produce a step-by-step plan ONLY.\nDo NOT emit ACTION lines. Wait for end_plan.\n</SYSTEM-PLAN-MODE>\n\n"
        except:
            pass
        
        # Comprehensive tool descriptions with proper documentation
        tool_instructions = """
## TOOL USAGE FORMAT
After your THINK and PLAN sections, you MUST output an ACTION line:
ACTION: tool_name {"param": "value"}

## AVAILABLE TOOLS

### read_file
Read the complete contents of a file from the repository.
Parameters:
  - path (string, required): File path relative to repository root
Example: ACTION: read_file {"path": "src/main.py"}

### write_file  
Create or overwrite a file with new content. Use sparingly - prefer apply_patch for modifications.
Parameters:
  - path (string, required): File path relative to repository root
  - content (string, required): Complete file content to write
Example: ACTION: write_file {"path": "README.md", "content": "# Project Title\\n\\nDescription here"}

### list_dir
List files and directories at a given path.
Parameters:
  - path (string, optional): Directory path relative to repo root, defaults to "."
  - recursive (boolean, optional): List subdirectories recursively, defaults to false
Example: ACTION: list_dir {"path": "src", "recursive": true}

### search_text
Search for text patterns in files using regular expressions.
Parameters:
  - pattern (string, required): Regex pattern to search for
  - path (string, optional): Directory to search in, defaults to "."  
  - glob (string, optional): File glob pattern like "**/*.py" or "src/**/*.js"
  - file_type (string, optional): File extension filter like "py", "js", "tsx"
  - mode (string, optional): Output mode - "content" (default), "files_with_matches", or "count"
  - multiline (boolean, optional): Enable multiline regex matching, defaults to false
Examples:
  ACTION: search_text {"pattern": "class.*Controller", "file_type": "py"}
  ACTION: search_text {"pattern": "TODO|FIXME", "glob": "src/**/*.js", "mode": "files_with_matches"}

### apply_patch
Apply a unified diff patch to modify files. This is the preferred way to edit existing files.
Parameters:
  - diff (string, required): Unified diff format patch with proper headers
Example: ACTION: apply_patch {"diff": "--- a/file.txt\\n+++ b/file.txt\\n@@ -1,3 +1,3 @@\\n line1\\n-old line\\n+new line\\n line3"}

### run
Execute shell commands in the repository directory. Use for builds, tests, scripts, and tools.
Parameters:
  - cmd (string, required): Shell command to execute (runs in bash)
  - timeout (integer, optional): Command timeout in seconds, defaults to 120
Examples:
  ACTION: run {"cmd": "python -m pytest tests/"}
  ACTION: run {"cmd": "npm test", "timeout": 60}
  ACTION: run {"cmd": "make build && ./run_tests.sh"}

### git_status
Show the current git repository status including staged, modified, and untracked files.
No parameters required.
Example: ACTION: git_status {}

### git_diff  
Show differences in the repository. Essential for understanding changes before committing.
Parameters:
  - staged (boolean, optional): Show staged changes instead of working tree, defaults to false
Examples:
  ACTION: git_diff {}
  ACTION: git_diff {"staged": true}

### git_commit
Create a git commit with staged changes or all changes.
Parameters:
  - message (string, required): Commit message describing the changes
  - all (boolean, optional): Stage all changes before committing, defaults to false
Examples:
  ACTION: git_commit {"message": "Fix type errors in authentication module"}
  ACTION: git_commit {"message": "Add user profile feature", "all": true}

## TOOL SELECTION GUIDELINES
- Always read files before editing them
- Use search_text to understand codebase structure
- Prefer apply_patch over write_file for existing files
- Run tests after making changes
- Check git_diff before committing"""
        
        return agent_instructions + plan_mode_instruction + tool_instructions
    
    def _get_agent_instructions(self, exclude_action_format=False):
        """Get just the agent instructions (AGENT.md) without tool descriptions.
        
        Args:
            exclude_action_format: If True, removes ACTION format instructions for native tool use
        """
        import os, json
        
        agent_instructions = ""
        try:
            # Check for custom instructions first
            for filename in ["AGENTS.md", "CLAUDE.md"]:
                filepath = os.path.join(self.repo, filename)
                if os.path.exists(filepath):
                    with open(filepath, "r") as f:
                        agent_instructions = f.read()
                    break
            
            # Fallback to default AGENT.md
            if not agent_instructions:
                agent_md_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "AGENT.md")
                with open(agent_md_path, "r") as f:
                    agent_instructions = f.read()
        except:
            pass
        
        # Remove ACTION format instructions if using native tools
        if exclude_action_format and agent_instructions:
            # Remove the I/O Protocol section that describes ACTION format
            lines = agent_instructions.split('\n')
            filtered_lines = []
            skip_mode = False
            for line in lines:
                if '## I/O Protocol' in line:
                    skip_mode = True
                    # Add replacement instructions for native tools
                    filtered_lines.append("## Native Tool Usage")
                    filtered_lines.append("You have access to tools that you can call directly.")
                    filtered_lines.append("Think step-by-step, then use the appropriate tools to accomplish the task.")
                    filtered_lines.append("After using a tool, analyze the result and decide on next steps.")
                    filtered_lines.append("")
                elif skip_mode and line.startswith('## '):
                    skip_mode = False
                    filtered_lines.append(line)
                elif not skip_mode:
                    filtered_lines.append(line)
            agent_instructions = '\n'.join(filtered_lines)
        
        # Check for plan mode
        plan_mode_instruction = ""
        plan_path = os.path.join(self.repo, ".agentic_plan.json")
        try:
            if os.path.exists(plan_path):
                with open(plan_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("active"):
                    plan_mode_instruction = "\n\nPlan mode is active. Produce a step-by-step plan ONLY. Do NOT use tools."
        except:
            pass
        
        return agent_instructions + plan_mode_instruction
    
    def _build_messages(self, prompt: str):
        """Build proper message list for Anthropic API with tool results."""
        messages = []
        
        # Parse the prompt to extract conversation history with tool uses
        lines = prompt.split("\n")
        current_content = []
        current_role = "user"
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for OBSERVATION marker (tool result)
            if line.startswith("OBSERVATION:"):
                # Add any pending user content
                if current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content)})
                    current_content = []
                
                # Extract the tool result
                obs_content = line[len("OBSERVATION:"):].strip()
                # Continue collecting until next ACTION or end
                i += 1
                while i < len(lines) and not lines[i].startswith("ACTION:"):
                    obs_content += "\n" + lines[i]
                    i += 1
                
                # Add as tool_result message
                messages.append({
                    "role": "user", 
                    "content": [{"type": "tool_result", "tool_use_id": "placeholder", "content": obs_content}]
                })
                continue
                
            # Check for ACTION marker (tool use)
            elif line.startswith("ACTION:"):
                # Add any pending content
                if current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content)})
                    current_content = []
                
                # Parse the action
                try:
                    name_and_json = line[len("ACTION:"):].strip()
                    parts = name_and_json.split(" ", 1)
                    tool_name = parts[0].strip()
                    tool_input = json.loads(parts[1]) if len(parts) > 1 else {}
                    
                    # Add as assistant message with tool use
                    messages.append({
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": "placeholder",
                            "name": tool_name,
                            "input": tool_input
                        }]
                    })
                except:
                    # If parsing fails, treat as regular content
                    current_content.append(line)
            else:
                current_content.append(line)
            
            i += 1
        
        # Add any remaining content
        if current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content)})
        
        # If no messages or last message is not user, ensure we end with user message
        if not messages or messages[-1]["role"] != "user":
            # Fallback to simple format
            return [{"role": "user", "content": prompt}]
        
        return messages
    
    def _get_openai_tools(self):
        """Get tool definitions in OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the complete contents of a file from the repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path relative to repository root"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or overwrite a file with new content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path relative to repository root"
                            },
                            "content": {
                                "type": "string",
                                "description": "Complete file content to write"
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List files and directories at a given path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path relative to repo root",
                                "default": "."
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "List subdirectories recursively",
                                "default": False
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": "Search for text patterns in files using regular expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Regex pattern to search for"
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to search in",
                                "default": "."
                            },
                            "glob": {
                                "type": "string",
                                "description": "File glob pattern like '**/*.py'"
                            },
                            "file_type": {
                                "type": "string",
                                "description": "File extension filter"
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["content", "files_with_matches", "count"],
                                "default": "content"
                            },
                            "multiline": {
                                "type": "boolean",
                                "description": "Enable multiline regex matching",
                                "default": False
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Apply a unified diff patch to modify files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diff": {
                                "type": "string",
                                "description": "Unified diff format patch"
                            }
                        },
                        "required": ["diff"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Execute shell commands in the repository directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {
                                "type": "string",
                                "description": "Shell command to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Command timeout in seconds",
                                "default": 120
                            }
                        },
                        "required": ["cmd"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show git repository status",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show git diff",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "staged": {
                                "type": "boolean",
                                "description": "Show staged changes",
                                "default": False
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Create a git commit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Commit message"
                            },
                            "all": {
                                "type": "boolean",
                                "description": "Stage all changes before committing",
                                "default": False
                            }
                        },
                        "required": ["message"]
                    }
                }
            }
        ]
    
    def _get_anthropic_tools(self):
        """Get tool definitions in proper Anthropic format."""
        return [
            {
                "name": "read_file",
                "description": "Read the complete contents of a file from the repository",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repository root"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Create or overwrite a file with new content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repository root"
                        },
                        "content": {
                            "type": "string", 
                            "description": "Complete file content to write"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "list_dir",
                "description": "List files and directories at a given path",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to repo root",
                            "default": "."
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "List subdirectories recursively",
                            "default": False
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "search_text",
                "description": "Search for text patterns in files using regular expressions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in",
                            "default": "."
                        },
                        "glob": {
                            "type": "string",
                            "description": "File glob pattern like '**/*.py'"
                        },
                        "file_type": {
                            "type": "string",
                            "description": "File extension filter"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["content", "files_with_matches", "count"],
                            "default": "content"
                        },
                        "multiline": {
                            "type": "boolean",
                            "description": "Enable multiline regex matching",
                            "default": False
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "apply_patch",
                "description": "Apply a unified diff patch to modify files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "diff": {
                            "type": "string",
                            "description": "Unified diff format patch"
                        }
                    },
                    "required": ["diff"]
                }
            },
            {
                "name": "run",
                "description": "Execute shell commands in the repository directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "Shell command to execute"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds",
                            "default": 120
                        }
                    },
                    "required": ["cmd"]
                }
            },
            {
                "name": "git_status",
                "description": "Show git repository status",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "git_diff",
                "description": "Show git diff",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "staged": {
                            "type": "boolean",
                            "description": "Show staged changes",
                            "default": False
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "git_commit",
                "description": "Create a git commit",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Commit message"
                        },
                        "all": {
                            "type": "boolean",
                            "description": "Stage all changes before committing",
                            "default": False
                        }
                    },
                    "required": ["message"]
                }
            }
        ]
    
    def _load_model_config(self):
        """Load model configuration from JSON file."""
        config_path = Path(__file__).parent / "models.json"
        try:
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load models.json: {e}[/yellow]")
            self.model_config = {"providers": {}}
    
    def _get_failover_chain(self, model_name: str) -> list[str]:
        """Get failover chain for a model from config."""
        if self.remote_provider and self.remote_provider in self.model_config.get("providers", {}):
            provider_config = self.model_config["providers"][self.remote_provider]
            chains = provider_config.get("failover_chains", {})
            return chains.get(model_name, [model_name])
        return [model_name]
    
    def _resolve_model_alias(self, model_name: str) -> str:
        """Resolve model aliases to actual model IDs from config."""
        if self.remote_provider and self.remote_provider in self.model_config.get("providers", {}):
            provider_config = self.model_config["providers"][self.remote_provider]
            aliases = provider_config.get("aliases", {})
            return aliases.get(model_name, model_name)
        return model_name
    
    def _update_context_window(self):
        """Update context window based on resolved model."""
        if self.resolved_model and "context_windows" in self.model_config:
            self.context_window = self.model_config["context_windows"].get(
                self.resolved_model, 
                128000 if self.remote_provider == "openai" else 200000  # Sensible defaults
            )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer when available."""
        if hasattr(self, 'tok') and self.tok is not None and not self.is_remote:
            # Use actual tokenizer for local models
            try:
                tokens = self.tok(text, return_tensors="pt", add_special_tokens=False)
                return tokens.input_ids.shape[1]
            except Exception:
                pass
        
        # Fallback: Conservative estimation for remote models
        # Different models have different token ratios:
        # - Claude: ~4 chars per token
        # - GPT: ~4 chars per token  
        # - Most models: 3-5 chars per token
        return int(len(text) / 3.5)
    
    def compact_context(self, history: str, max_tokens: Optional[int] = None) -> str:
        """Compact conversation history to fit within context window.
        
        Strategy:
        1. Keep the initial prompt/goal
        2. Summarize middle portions
        3. Keep recent exchanges
        4. Use actual tokenizer when available
        """
        compact_logger.info("=== COMPACT_CONTEXT CALLED ===")
        compact_logger.info(f"Input history length: {len(history):,} chars")
        compact_logger.info(f"Context window: {self.context_window:,} tokens")
        
        if max_tokens is None:
            # Reserve 30% for response, use 70% for context (more conservative)
            max_tokens = int(self.context_window * 0.7)
        
        compact_logger.info(f"Max tokens allowed: {max_tokens:,}")
        
        # Use proper token counting
        current_tokens = self._count_tokens(history)
        compact_logger.info(f"Current tokens: {current_tokens:,}")
        compact_logger.info(f"History preview (first 200): {history[:200]}")
        compact_logger.info(f"History preview (last 200): {history[-200:]}")
        
        if current_tokens <= max_tokens:
            compact_logger.info("History fits within token limit - no compaction needed")
            return history
        
        # Split by OBSERVATION markers to identify turns
        compact_logger.info("Starting compaction process...")
        parts = history.split("\nOBSERVATION:")
        compact_logger.info(f"Split history into {len(parts)} parts")
        
        if len(parts) <= 3:
            # Too short to meaningfully compact
            compact_logger.info("History too short to compact meaningfully")
            return history
        
        # Keep first part (goal/initial prompt)
        first = parts[0]
        compact_logger.info(f"First part length: {len(first):,} chars")
        
        # Keep last few exchanges (most recent context)
        recent_count = min(3, len(parts) // 2)
        recent = "\nOBSERVATION:".join(parts[-recent_count:])
        compact_logger.info(f"Keeping {recent_count} recent parts, length: {len(recent):,} chars")
        
        # Middle section - create summary
        middle_parts = parts[1:-recent_count] if recent_count < len(parts) - 1 else []
        compact_logger.info(f"Compressing {len(middle_parts)} middle parts")
        
        if middle_parts:
            summary = f"\n[CONTEXT COMPRESSED: {len(middle_parts)} earlier exchanges omitted for brevity]\n"
        else:
            summary = ""
        
        compacted = first + summary + "\nOBSERVATION:" + recent
        compact_logger.info(f"Compacted result length: {len(compacted):,} chars")
        
        # Final check - if still too long, truncate from the middle
        compacted_tokens = self._count_tokens(compacted)
        compact_logger.info(f"Compacted tokens: {compacted_tokens:,}")
        if compacted_tokens > max_tokens:
            # More aggressive truncation needed
            # Target 90% of max_tokens to leave some buffer
            target_tokens = int(max_tokens * 0.9)
            
            # Binary search approach for precise truncation
            lines = compacted.split('\n')
            if len(lines) > 10:
                # Keep first 20% and last 20% of lines
                keep_start = max(2, len(lines) // 5)
                keep_end = max(2, len(lines) // 5)
                
                truncated_lines = (
                    lines[:keep_start] + 
                    [f"[...CONTEXT TRUNCATED: {len(lines) - keep_start - keep_end} lines omitted...]"] +
                    lines[-keep_end:]
                )
                compacted = '\n'.join(truncated_lines)
            else:
                # For very short content, just truncate characters
                char_limit = target_tokens * 4  # Conservative estimate
                keep_start = char_limit // 3
                keep_end = char_limit * 2 // 3
                compacted = compacted[:keep_start] + "\n[...TRUNCATED...]\n" + compacted[-keep_end:]
        
        return compacted

    # -------- non-stream (kept for compatibility) --------

    def _gen(self, prompt: str, max_new_tokens: int = 1024) -> str:
        if self.is_remote:
            return self._gen_remote(prompt, max_new_tokens)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def _gen_remote(self, prompt: str, max_new_tokens: int) -> str:
        """Generate with automatic failover on quota/rate limit errors."""
        for attempt in range(len(self.failover_chain)):
            current_model = self.failover_chain[self.current_failover_index]
            
            try:
                if self.remote_provider == "openai":
                    # Use OpenAI SDK for OpenAI-compatible endpoints
                    resp = self.openai_client.chat.completions.create(
                        model=current_model,
                        messages=[{"role":"system","content":self._get_system_prompt()},
                                  {"role":"user","content":prompt}],
                        temperature=self.temperature,
                        max_tokens=max_new_tokens,
                        stop=self.stop,
                    )
                    return resp.choices[0].message.content
                else:
                    # Use Anthropic SDK for Anthropic-compatible endpoints
                    # For ACTION-based parsing (backward compatibility)
                    msg = self.anthropic_client.messages.create(
                        model=current_model,
                        max_tokens=max_new_tokens,
                        temperature=self.temperature,
                        system=self._get_system_prompt(),  # Include ACTION instructions
                        messages=[{"role":"user","content":prompt}],
                        stop_sequences=self.stop,
                    )
                    # Extract text content for ACTION parsing
                    parts = []
                    for block in msg.content:
                        if getattr(block, "type", None) == "text":
                            parts.append(block.text)
                    return "".join(parts) or str(msg)
                    
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a quota or rate limit error
                if any(x in error_str for x in ['quota', 'rate limit', 'rate_limit', 'insufficient_quota', '429']):
                    if self.current_failover_index < len(self.failover_chain) - 1:
                        self.current_failover_index += 1
                        next_model = self.failover_chain[self.current_failover_index]
                        console.print(f"[yellow]Quota/rate limit hit for {current_model}, failing over to {next_model}[/yellow]")
                        continue
                    else:
                        console.print(f"[red]All models in failover chain exhausted[/red]")
                        raise
                else:
                    # Not a quota error, re-raise
                    raise
        
        raise RuntimeError("Failed to get response from any model in failover chain")

    # -------- streaming APIs --------
    def stream(self, prompt: str, max_new_tokens: int = 1024) -> Iterator[str]:
        """
        Yields text chunks as they arrive. Works with:
        - Local HF models (TextIteratorStreamer)
        - OpenAI-compatible chat completions (stream=True)
        - Anthropic-compatible messages streaming
        """
        if self.is_remote:
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
            model=self.failover_chain[self.current_failover_index],
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
        """Stream responses using Anthropic SDK for ACTION-based parsing."""
        # For backward compatibility with ACTION format
        with self.anthropic_client.messages.stream(
            model=self.failover_chain[self.current_failover_index],
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            system=self._get_system_prompt(),  # Include ACTION instructions
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=self.stop,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text

    # -------- agent loop --------
    def run(self, goal: str, max_iters: int = 20, debug: bool = False):
        """Main agent loop - dispatches to provider-specific implementations."""
        if self.is_remote:
            if self.remote_provider == "anthropic":
                return self._run_anthropic_native(goal, max_iters, debug)
            elif self.remote_provider == "openai":
                return self._run_openai_native(goal, max_iters, debug)
        # Fall back to ACTION-based parsing for local models
        return self._run_with_actions(goal, max_iters, debug)
    
    def _run_openai_native(self, goal: str, max_iters: int, debug: bool):
        """Run agent loop using OpenAI's native function calling."""
        if debug:
            console.print("[yellow]Using OpenAI native function calling[/yellow]")
        
        # Initialize conversation with goal
        messages = [
            {"role": "system", "content": self._get_agent_instructions(exclude_action_format=True)},
            {"role": "user", "content": make_prompt(goal, self.repo)}
        ]
        
        for step in range(max_iters):
            if debug:
                console.rule(f"Step {step+1}")
            
            # Add delay to avoid rate limits
            if step > 0:
                time.sleep(0.5)
            
            try:
                # Call OpenAI with tools using SDK client
                response = self.openai_client.chat.completions.create(
                    model=self.failover_chain[self.current_failover_index],
                    messages=messages,
                    tools=self._get_openai_tools(),
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=4096,
                )
                
                # Get the message
                message = response.choices[0].message
                messages.append(message)
                
                # Display any text content
                if message.content:
                    console.print(message.content)
                
                # Process tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if debug:
                            console.print(f"[bold]NATIVE TOOL CALL[/bold]: {tool_call.function.name} {tool_call.function.arguments}")
                        
                        # Parse arguments
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        
                        # Execute tool
                        result = self.tools.dispatch(tool_call.function.name, args)
                        
                        # Truncate large results
                        MAX_RESULT_LENGTH = 5000
                        truncated_result = result
                        if isinstance(result, str) and len(result) > MAX_RESULT_LENGTH:
                            truncated_result = result[:MAX_RESULT_LENGTH] + "\n[... Output truncated ...]"
                        
                        if debug:
                            console.print(f"[bold]RESULT[/bold]: {(result[:500] + '...') if isinstance(result, str) and len(result) > 500 else result}")
                        
                        # Add truncated tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(truncated_result)
                        })
                else:
                    # No tool calls, we're done
                    if debug:
                        console.print("[yellow]No tools called. Agent finished.[/yellow]")
                    break
                    
                # Check if goal is satisfied
                if self.tools.goal_satisfied(goal):
                    if debug:
                        console.print("[green]Goal satisfied. Exiting.[/green]")
                    break
                    
            except Exception as e:
                error_str = str(e).lower()
                # Handle rate limits with failover
                if any(x in error_str for x in ['quota', 'rate limit', 'rate_limit', '429']):
                    if self.current_failover_index < len(self.failover_chain) - 1:
                        self.current_failover_index += 1
                        next_model = self.failover_chain[self.current_failover_index]
                        console.print(f"[yellow]Rate limit hit, failing over to {next_model}[/yellow]")
                        continue
                    else:
                        console.print(f"[red]All models exhausted[/red]")
                        raise
                else:
                    raise
    
    def _run_anthropic_native(self, goal: str, max_iters: int, debug: bool):
        """Run agent loop using Anthropic's native tool calling."""
        if debug:
            console.print("[yellow]Using Anthropic native tool calling[/yellow]")
        
        # Initialize conversation with goal
        messages = [{"role": "user", "content": make_prompt(goal, self.repo)}]
        
        for step in range(max_iters):
            if debug:
                console.rule(f"Step {step+1}")
            
            # Add small delay to avoid rate limit bursts
            if step > 0:
                time.sleep(0.5)  # 500ms delay between API calls
            
            try:
                # Call Anthropic with tools using SDK client
                response = self.anthropic_client.messages.create(
                    model=self.failover_chain[self.current_failover_index],
                    max_tokens=4096,
                    temperature=self.temperature,
                    system=self._get_agent_instructions(exclude_action_format=True),
                    tools=self._get_anthropic_tools(),
                    messages=messages,
                )
                
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": response.content})
                
                # Process response blocks
                tool_uses = []
                if debug:
                    console.print(f"[yellow]Response has {len(response.content)} content blocks[/yellow]")
                for block in response.content:
                    if debug:
                        console.print(f"[yellow]Block type: {block.type}[/yellow]")
                    if block.type == "text":
                        # Display text to user
                        if block.text.strip():
                            console.print(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)
                        
                        # Execute tool
                        if debug:
                            console.print(f"[bold]NATIVE TOOL USE[/bold]: {block.name} {json.dumps(block.input)}")
                        else:
                            # Clean output
                            if block.name == "write_file":
                                console.print(f"\n[dim]Writing {block.input.get('path', 'file')}...[/dim]")
                            elif block.name == "read_file":
                                console.print(f"\n[dim]Reading {block.input.get('path', 'file')}...[/dim]")
                            elif block.name == "apply_patch":
                                console.print(f"\n[dim]Applying changes...[/dim]")
                            elif block.name == "run":
                                cmd = block.input.get('cmd', '')
                                console.print(f"\n[dim]Running: {cmd[:50]}{'...' if len(cmd) > 50 else ''}[/dim]")
                        
                        # Execute the tool
                        result = self.tools.dispatch(block.name, block.input)
                        
                        # Truncate large results to prevent token explosion
                        MAX_RESULT_LENGTH = 5000  # Characters, roughly 1250 tokens
                        truncated_result = result
                        was_truncated = False
                        if isinstance(result, str) and len(result) > MAX_RESULT_LENGTH:
                            truncated_result = result[:MAX_RESULT_LENGTH] + "\n[... Output truncated - " + str(len(result) - MAX_RESULT_LENGTH) + " characters omitted ...]"
                            was_truncated = True
                        
                        if debug:
                            console.print(f"[bold]RESULT[/bold]: {(result[:2000] + '…') if isinstance(result, str) and len(result) > 2000 else result}")
                            if was_truncated:
                                console.print("[yellow]Note: Full output truncated in conversation history to prevent token limits[/yellow]")
                        else:
                            # Show errors
                            try:
                                res_obj = json.loads(result) if isinstance(result, str) else result
                                if isinstance(res_obj, dict) and res_obj.get("status") == "error":
                                    console.print(f"[red]Error: {res_obj.get('error', 'Unknown error')}[/red]")
                            except:
                                pass
                        
                        # Add truncated tool result to messages
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(truncated_result)
                            }]
                        })
                
                # If no tools were used, we're done
                if not tool_uses:
                    if debug:
                        console.print("[yellow]No tools used. Agent finished.[/yellow]")
                    break
                    
                # Check if goal is satisfied
                if self.tools.goal_satisfied(goal):
                    if debug:
                        console.print("[green]Goal satisfied. Exiting.[/green]")
                    break
                    
            except Exception as e:
                error_str = str(e).lower()
                # Handle rate limits with failover
                if any(x in error_str for x in ['quota', 'rate limit', 'rate_limit', 'insufficient_quota', '429']):
                    if self.current_failover_index < len(self.failover_chain) - 1:
                        self.current_failover_index += 1
                        next_model = self.failover_chain[self.current_failover_index]
                        console.print(f"[yellow]Rate limit hit, failing over to {next_model}[/yellow]")
                        continue
                    else:
                        console.print(f"[red]All models exhausted[/red]")
                        raise
                else:
                    raise
    
    def _run_with_actions(self, goal: str, max_iters: int, debug: bool):
        """Run agent loop with ACTION-based parsing (for non-Anthropic providers)."""
        # Always rebuild the head so plan-mode instructions reflect current state
        history = make_prompt(goal, self.repo)
        for step in range(max_iters):
            if debug:
                console.rule(f"Step {step+1}")
            # Compact context if needed
            history = self.compact_context(history)
            # refresh plan header each turn
            base = make_prompt(goal, self.repo)
            # keep only the tail after the first header
            tail = history.split("</CONTEXT>", 1)[-1] if "</CONTEXT>" in history else ""
            prompt = base + tail
            # Add small delay to avoid rate limit bursts
            if step > 0:
                time.sleep(0.5)  # 500ms delay between API calls
            
            response = self._gen(prompt)
            
            if debug:
                console.print(f"[dim]Response preview: {response[:200]}...[/dim]")
            
            action = self._extract_action(response)
            if not action:
                if debug:
                    console.print("[yellow]No ACTION detected. Stopping.")
                    console.print(f"[red]Full response was:[/red]\n{response}")
                else:
                    console.print("\n" + response.strip())
                break
            
            tool_name, args = action
            
            if debug:
                console.print(f"[bold]ACTION[/bold]: {tool_name} {args}")
            else:
                # Clean output
                if tool_name == "write_file":
                    console.print(f"\n[dim]Writing {args.get('path', 'file')}...[/dim]")
                elif tool_name == "read_file":
                    console.print(f"\n[dim]Reading {args.get('path', 'file')}...[/dim]")
                elif tool_name == "apply_patch":
                    console.print(f"\n[dim]Applying changes...[/dim]")
                elif tool_name == "run":
                    cmd = args.get('cmd', '')
                    console.print(f"\n[dim]Running: {cmd[:50]}{'...' if len(cmd) > 50 else ''}[/dim]")
            
            obs = self.tools.dispatch(tool_name, args)
            
            # Truncate large observations to prevent token explosion
            MAX_OBS_LENGTH = 5000  # Characters, roughly 1250 tokens
            truncated_obs = obs
            was_truncated = False
            if isinstance(obs, str) and len(obs) > MAX_OBS_LENGTH:
                truncated_obs = obs[:MAX_OBS_LENGTH] + "\n[... Output truncated - " + str(len(obs) - MAX_OBS_LENGTH) + " characters omitted ...]"
                was_truncated = True
            
            if debug:
                console.print("[bold]OBSERVATION[/bold]:", (obs[:2000] + "…") if isinstance(obs, str) and len(obs) > 2000 else obs)
                if was_truncated:
                    console.print("[yellow]Note: Full output truncated in conversation history to prevent token limits[/yellow]")
            else:
                # Clean output - show minimal feedback
                try:
                    result = json.loads(obs) if isinstance(obs, str) else obs
                    if isinstance(result, dict) and result.get("status") == "error":
                        console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
                except:
                    pass
            
            # Add truncated observation to history
            history = prompt + f"\nOBSERVATION: {truncated_obs}\n"
            if self.tools.goal_satisfied(goal):
                if debug:
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