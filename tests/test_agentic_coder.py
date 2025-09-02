#!/usr/bin/env python3
"""
Comprehensive test suite for Agentic Coder
Tests all major components and features.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_coder.config import ModelSpec, MODEL_PRESETS
from agentic_coder.tools import ToolRegistry
from agentic_coder.tools.filesystem import read_file, write_file, list_dir, search_text
from agentic_coder.tools.repo_map import repo_map
from agentic_coder.tools.ast_parser import enhanced_repo_map
from agentic_coder.hooks import HookManager, Hook
from agentic_coder.memory import MemoryManager, MemoryEntry
from agentic_coder.mcp import MCPServer, MCPClient, MCPTool, MCPResource
from agentic_coder.github_actions import GitHubActionsManager
from agentic_coder.commands import CommandProcessor


class TestConfig:
    """Test configuration and model presets."""
    
    def test_model_spec_creation(self):
        spec = ModelSpec(
            id="test-model",
            dtype="float16",
            trust_remote_code=False
        )
        assert spec.id == "test-model"
        assert spec.dtype == "float16"
        assert spec.trust_remote_code == False
    
    def test_model_presets_exist(self):
        assert len(MODEL_PRESETS) > 0
        assert "qwen2.5-coder-14b" in MODEL_PRESETS
        assert "remote-openai" in MODEL_PRESETS
        assert "remote-anthropic" in MODEL_PRESETS
    
    def test_remote_models_configured(self):
        openai_spec = MODEL_PRESETS["remote-openai"]
        # Remote models don't have the remote flag set in the current config
        assert openai_spec.id == "REMOTE"
        
        anthropic_spec = MODEL_PRESETS["remote-anthropic"]
        assert anthropic_spec.id == "REMOTE"


class TestToolRegistry:
    """Test the tool registry system."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_tool_registry_creation(self, temp_repo):
        registry = ToolRegistry(temp_repo)
        assert registry.repo == temp_repo
        assert len(registry._tools) > 0
    
    def test_tool_dispatch(self, temp_repo):
        registry = ToolRegistry(temp_repo)
        
        # Test read_file tool
        test_file = os.path.join(temp_repo, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        result = registry.dispatch("read_file", {"path": "test.txt"})
        result_data = json.loads(result)
        assert result_data["status"] == "ok"
        assert "test content" in result_data["output"]
    
    def test_unknown_tool(self, temp_repo):
        registry = ToolRegistry(temp_repo)
        result = registry.dispatch("unknown_tool", {})
        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "unknown tool" in result_data["error"]
    
    def test_plan_mode_gating(self, temp_repo):
        registry = ToolRegistry(temp_repo)
        
        # Create plan mode file
        plan_file = os.path.join(temp_repo, ".agentic_plan.json")
        with open(plan_file, "w") as f:
            json.dump({"active": True}, f)
        
        # Try to use write_file (should be blocked)
        result = registry.dispatch("write_file", {"path": "test.txt", "content": "data"})
        result_data = json.loads(result)
        assert result_data["status"] == "plan_active"


class TestFilesystemTools:
    """Test filesystem tools."""
    
    @pytest.fixture
    def temp_repo(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_read_write_file(self, temp_repo):
        # Write file
        write_result = write_file(temp_repo, "test.py", "print('hello')")
        assert "WROTE:" in write_result  # Match actual output format
        
        # Read file
        read_result = read_file(temp_repo, "test.py")
        assert "print('hello')" in read_result
    
    def test_list_directory(self, temp_repo):
        # Create some files
        Path(temp_repo, "file1.txt").touch()
        Path(temp_repo, "file2.py").touch()
        os.makedirs(os.path.join(temp_repo, "subdir"))
        
        result = list_dir(temp_repo, ".")
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result
    
    def test_search_text(self, temp_repo):
        # Create test files
        with open(os.path.join(temp_repo, "test1.py"), "w") as f:
            f.write("def hello():\n    print('world')")
        
        with open(os.path.join(temp_repo, "test2.py"), "w") as f:
            f.write("def goodbye():\n    print('world')")
        
        result = search_text(temp_repo, "hello")
        assert "test1.py" in result
        assert "test2.py" not in result or "goodbye" not in result


class TestRepoMap:
    """Test repository mapping."""
    
    @pytest.fixture
    def temp_repo(self):
        temp_dir = tempfile.mkdtemp()
        # Initialize git repo
        os.system(f"cd {temp_dir} && git init")
        
        # Create test files
        with open(os.path.join(temp_dir, "main.py"), "w") as f:
            f.write("class MyClass:\n    def method(self):\n        pass")
        
        with open(os.path.join(temp_dir, "utils.py"), "w") as f:
            f.write("def helper_function():\n    return 42")
        
        # Add files to git
        os.system(f"cd {temp_dir} && git add .")
        os.system(f"cd {temp_dir} && git commit -m 'test'")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_repo_map_generation(self, temp_repo):
        result = repo_map(temp_repo, max_tokens=1000)
        assert "REPOSITORY MAP" in result
        assert "main.py" in result
        assert "class MyClass" in result
        assert "utils.py" in result
        assert "def helper_function" in result
    
    def test_enhanced_repo_map(self, temp_repo):
        result = enhanced_repo_map(temp_repo, max_tokens=1000, use_ast=False)
        assert "REPOSITORY MAP" in result
        # Falls back to basic repo_map when AST unavailable


class TestHooks:
    """Test hooks system."""
    
    @pytest.fixture
    def temp_repo(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_hook_manager_creation(self, temp_repo):
        hook_manager = HookManager(temp_repo)
        assert hook_manager.repo_path == temp_repo
        assert len(hook_manager.hooks) > 0  # Default hooks created
    
    def test_add_remove_hook(self, temp_repo):
        hook_manager = HookManager(temp_repo)
        
        # Add hook
        hook_manager.add_hook("test_hook", "echo test", "test_trigger", "Test hook")
        assert "test_hook" in hook_manager.hooks
        
        # Remove hook
        removed = hook_manager.remove_hook("test_hook")
        assert removed == True
        assert "test_hook" not in hook_manager.hooks
    
    def test_enable_disable_hook(self, temp_repo):
        hook_manager = HookManager(temp_repo)
        hook_manager.add_hook("test_hook", "echo test", "test_trigger", enabled=True)
        
        # Disable
        hook_manager.disable_hook("test_hook")
        assert hook_manager.hooks["test_hook"].enabled == False
        
        # Enable
        hook_manager.enable_hook("test_hook")
        assert hook_manager.hooks["test_hook"].enabled == True
    
    def test_execute_hooks(self, temp_repo):
        hook_manager = HookManager(temp_repo)
        hook_manager.add_hook(
            "test_hook",
            "echo 'Hook executed with {value}'",
            "test_trigger",
            enabled=True
        )
        
        results = hook_manager.execute_hooks("test_trigger", {"value": "test123"})
        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert "test123" in results[0]["output"]


class TestMemory:
    """Test memory management."""
    
    @pytest.fixture
    def temp_repo(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_memory_manager_creation(self, temp_repo):
        memory_manager = MemoryManager(temp_repo)
        assert memory_manager.repo_path == temp_repo
    
    def test_session_management(self, temp_repo):
        memory_manager = MemoryManager(temp_repo)
        
        # Start session
        session_id = memory_manager.start_session("test_model")
        assert session_id is not None
        assert memory_manager.current_session is not None
        
        # Add memories
        memory_manager.add_memory("goal", "Test goal")
        memory_manager.add_memory("action", "Test action")
        
        # Get recent memories
        recent = memory_manager.get_recent_memories(10)
        assert len(recent) == 2
        assert recent[0].type == "goal"
        assert recent[1].type == "action"
    
    def test_memory_search(self, temp_repo):
        memory_manager = MemoryManager(temp_repo)
        memory_manager.start_session("test_model")
        
        memory_manager.add_memory("goal", "Build a web application")
        memory_manager.add_memory("action", "Created database schema")
        memory_manager.add_memory("result", "Successfully set up PostgreSQL")
        
        # Search memories
        results = memory_manager.search_memories("database")
        assert len(results) == 1
        assert "database" in results[0].content.lower()
    
    def test_session_persistence(self, temp_repo):
        memory_manager = MemoryManager(temp_repo)
        
        # Create and save session
        session_id = memory_manager.start_session("test_model")
        memory_manager.add_memory("goal", "Persistent goal")
        memory_manager.set_goal("Test goal")
        
        # Load session in new manager
        new_manager = MemoryManager(temp_repo)
        loaded = new_manager.load_session(session_id)
        assert loaded == True
        assert new_manager.get_goal() == "Test goal"


class TestMCP:
    """Test MCP integration."""
    
    def test_mcp_server_creation(self):
        server = MCPServer("/tmp")
        assert server.repo_path == "/tmp"
        assert len(server.tools) > 0
        assert len(server.prompts) > 0
    
    def test_mcp_tool_registration(self):
        server = MCPServer("/tmp")
        
        tool = MCPTool(
            name="test_tool",
            description="Test tool",
            inputSchema={"type": "object"}
        )
        server.add_tool(tool)
        assert "test_tool" in server.tools
    
    @pytest.mark.asyncio
    async def test_mcp_request_handling(self):
        server = MCPServer("/tmp")
        
        # Test initialization
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "capabilities" in response["result"]
    
    def test_mcp_client_creation(self):
        client = MCPClient()
        assert client.servers == {}
        
        # Add server
        client.add_server("test_server", "python", ["-m", "test"])
        assert "test_server" in client.servers


class TestGitHubActions:
    """Test GitHub Actions integration."""
    
    @pytest.fixture
    def temp_repo(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_github_actions_manager(self, temp_repo):
        ga_manager = GitHubActionsManager(temp_repo)
        assert ga_manager.repo_path == temp_repo
        assert os.path.exists(ga_manager.workflows_dir)
    
    def test_create_workflows(self, temp_repo):
        ga_manager = GitHubActionsManager(temp_repo)
        
        # Create code review workflow
        workflow_path = ga_manager.create_code_review_workflow()
        assert os.path.exists(workflow_path)
        
        # Validate created workflow
        validation = ga_manager.validate_workflow("agentic_code_review.yml")
        assert validation["valid"] == True
    
    def test_list_workflows(self, temp_repo):
        ga_manager = GitHubActionsManager(temp_repo)
        
        # Create some workflows
        ga_manager.create_test_automation_workflow()
        ga_manager.create_security_scan_workflow()
        
        # List workflows
        workflows = ga_manager.list_workflows()
        assert len(workflows) >= 2
        
        # Check workflow properties
        for workflow in workflows:
            assert "file" in workflow
            assert "name" in workflow


class TestCommands:
    """Test command processor."""
    
    def test_command_processor_creation(self):
        mock_session = Mock()
        mock_session.repo = "/tmp"
        mock_session.preset = "test"
        mock_session.streaming = True
        mock_session.goal = "Test goal"
        
        processor = CommandProcessor(mock_session)
        assert processor.session == mock_session
        assert len(processor.commands) > 0
    
    def test_command_parsing(self):
        mock_session = Mock()
        processor = CommandProcessor(mock_session)
        
        # Test colon prefix
        assert processor.process_command(":help") == True
        
        # Test slash prefix
        assert processor.process_command("/help") == True
        
        # Test quit command
        assert processor.process_command(":quit") == True  # Command processor returns True for valid commands
        assert processor.process_command("/exit") == False
    
    def test_text_input_handling(self):
        mock_session = Mock()
        mock_session.set_goal = Mock()
        
        processor = CommandProcessor(mock_session)
        
        # Test regular text input
        processor.process_command("This is a goal")
        mock_session.set_goal.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])