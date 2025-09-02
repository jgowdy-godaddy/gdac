from __future__ import annotations
import os
import json
import subprocess
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path

"""
Hooks System: Extensible command triggers for custom workflows
Integrates with external tools and custom automation workflows.
"""

@dataclass 
class Hook:
    """A hook configuration."""
    name: str
    command: str
    trigger: str  # 'before_step', 'after_step', 'on_error', 'on_goal', etc.
    description: str = ""
    enabled: bool = True

class HookManager:
    """Manage and execute hooks for extensibility."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.hooks_file = os.path.join(repo_path, ".agentic_hooks.json")
        self.hooks: Dict[str, Hook] = {}
        self.load_hooks()
    
    def load_hooks(self):
        """Load hooks from configuration file."""
        if not os.path.exists(self.hooks_file):
            self._create_default_hooks()
            return
            
        try:
            with open(self.hooks_file, 'r') as f:
                data = json.load(f)
            
            self.hooks = {}
            for hook_data in data.get('hooks', []):
                hook = Hook(**hook_data)
                self.hooks[hook.name] = hook
                
        except Exception as e:
            print(f"Warning: Failed to load hooks: {e}")
            self._create_default_hooks()
    
    def save_hooks(self):
        """Save hooks to configuration file."""
        try:
            data = {
                'hooks': [
                    {
                        'name': hook.name,
                        'command': hook.command, 
                        'trigger': hook.trigger,
                        'description': hook.description,
                        'enabled': hook.enabled
                    }
                    for hook in self.hooks.values()
                ]
            }
            
            with open(self.hooks_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save hooks: {e}")
    
    def _create_default_hooks(self):
        """Create default hook examples."""
        self.hooks = {
            'pre_commit_check': Hook(
                name='pre_commit_check',
                command='echo "Running pre-commit checks..." && git status --porcelain',
                trigger='before_commit',
                description='Check repository status before commits',
                enabled=False  # Disabled by default
            ),
            'format_on_write': Hook(
                name='format_on_write',
                command='echo "Auto-formatting code..." && python -m black {file_path} 2>/dev/null || true',
                trigger='after_write',
                description='Auto-format Python files after writing',
                enabled=False
            ),
            'test_on_change': Hook(
                name='test_on_change',
                command='echo "Running tests..." && python -m pytest -x --tb=short 2>/dev/null || echo "Tests not configured"',
                trigger='after_step',
                description='Run tests after significant changes',
                enabled=False
            )
        }
        self.save_hooks()
    
    def add_hook(self, name: str, command: str, trigger: str, description: str = "", enabled: bool = True):
        """Add a new hook."""
        self.hooks[name] = Hook(name, command, trigger, description, enabled)
        self.save_hooks()
    
    def remove_hook(self, name: str) -> bool:
        """Remove a hook by name."""
        if name in self.hooks:
            del self.hooks[name]
            self.save_hooks()
            return True
        return False
    
    def enable_hook(self, name: str) -> bool:
        """Enable a hook."""
        if name in self.hooks:
            self.hooks[name].enabled = True
            self.save_hooks()
            return True
        return False
    
    def disable_hook(self, name: str) -> bool:
        """Disable a hook."""
        if name in self.hooks:
            self.hooks[name].enabled = False
            self.save_hooks()
            return True
        return False
    
    def list_hooks(self) -> List[Hook]:
        """Get all hooks."""
        return list(self.hooks.values())
    
    def get_hooks_for_trigger(self, trigger: str) -> List[Hook]:
        """Get enabled hooks for a specific trigger."""
        return [
            hook for hook in self.hooks.values() 
            if hook.trigger == trigger and hook.enabled
        ]
    
    def execute_hooks(self, trigger: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute all hooks for a trigger."""
        hooks = self.get_hooks_for_trigger(trigger)
        results = []
        
        for hook in hooks:
            try:
                result = self._execute_hook(hook, context or {})
                results.append({
                    'hook': hook.name,
                    'status': 'success',
                    'output': result
                })
            except Exception as e:
                results.append({
                    'hook': hook.name,
                    'status': 'error', 
                    'error': str(e)
                })
        
        return results
    
    def _execute_hook(self, hook: Hook, context: Dict[str, Any]) -> str:
        """Execute a single hook command."""
        # Format command with context variables
        command = hook.command
        for key, value in context.items():
            command = command.replace(f"{{{key}}}", str(value))
        
        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            return output.strip()
            
        except subprocess.TimeoutExpired:
            raise Exception(f"Hook '{hook.name}' timed out after 30 seconds")
        except Exception as e:
            raise Exception(f"Hook '{hook.name}' failed: {e}")

# Hook trigger functions for integration with the main system
def trigger_hooks(hook_manager: HookManager, trigger: str, context: Dict[str, Any] = None):
    """Trigger hooks and handle results."""
    if not hook_manager:
        return
        
    results = hook_manager.execute_hooks(trigger, context)
    
    # Log hook results
    for result in results:
        if result['status'] == 'success':
            if result.get('output'):
                print(f"Hook {result['hook']}: {result['output'][:100]}{'...' if len(result['output']) > 100 else ''}")
        else:
            print(f"Hook {result['hook']} failed: {result['error']}")

# Integration points for the main system
class HookableSession:
    """Mixin to add hooks support to Session classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_manager = HookManager(getattr(self, 'repo', '.'))
    
    def trigger_hooks(self, trigger: str, **context):
        """Trigger hooks with context."""
        trigger_hooks(self.hook_manager, trigger, context)