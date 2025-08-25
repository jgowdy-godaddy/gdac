from __future__ import annotations
import os
import json
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

"""
MCP (Model Context Protocol) Integration
Provides both server and client capabilities for connecting to external tools and data sources.
Based on the MCP specification from Anthropic.
"""

logger = logging.getLogger(__name__)

# MCP Protocol Primitives
@dataclass
class MCPResource:
    """A resource exposed by an MCP server."""
    uri: str
    name: str
    description: str
    mimeType: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    metadata: Dict[str, Any] = None

@dataclass
class MCPPrompt:
    """A prompt template exposed by an MCP server."""
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

class MCPServer:
    """
    MCP Server implementation for Agentic Coder.
    Exposes our tools, resources, and prompts to external MCP clients.
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.resources: Dict[str, MCPResource] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self._initialize_capabilities()
    
    def _initialize_capabilities(self):
        """Initialize server capabilities."""
        # Expose repository as a resource
        self.add_resource(MCPResource(
            uri=f"file://{os.path.abspath(self.repo_path)}",
            name="repository",
            description="The current repository",
            mimeType="application/x-directory"
        ))
        
        # Expose our tools as MCP tools
        self._register_tools()
        
        # Add prompt templates
        self._register_prompts()
    
    def _register_tools(self):
        """Register Agentic Coder tools as MCP tools."""
        # File operations
        self.add_tool(MCPTool(
            name="read_file",
            description="Read a file from the repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo"}
                },
                "required": ["path"]
            }
        ))
        
        self.add_tool(MCPTool(
            name="write_file",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo"},
                    "content": {"type": "string", "description": "File content"}
                },
                "required": ["path", "content"]
            }
        ))
        
        self.add_tool(MCPTool(
            name="search_text",
            description="Search for text patterns in the repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "glob": {"type": "string", "description": "File glob pattern"}
                },
                "required": ["pattern"]
            }
        ))
        
        # Code analysis
        self.add_tool(MCPTool(
            name="analyze_code",
            description="Analyze code structure using AST",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File to analyze"}
                }
            }
        ))
        
        # Git operations
        self.add_tool(MCPTool(
            name="git_status",
            description="Get git repository status",
            inputSchema={"type": "object", "properties": {}}
        ))
        
        self.add_tool(MCPTool(
            name="git_diff",
            description="Get git diff",
            inputSchema={
                "type": "object",
                "properties": {
                    "cached": {"type": "boolean", "description": "Show staged changes"}
                }
            }
        ))
    
    def _register_prompts(self):
        """Register prompt templates."""
        self.add_prompt(MCPPrompt(
            name="code_review",
            description="Review code changes for quality and issues",
            arguments=[
                {"name": "diff", "description": "The code diff to review", "required": True}
            ]
        ))
        
        self.add_prompt(MCPPrompt(
            name="bug_fix",
            description="Analyze and fix a bug",
            arguments=[
                {"name": "error", "description": "Error message or description", "required": True},
                {"name": "context", "description": "Additional context", "required": False}
            ]
        ))
        
        self.add_prompt(MCPPrompt(
            name="feature_implementation",
            description="Implement a new feature",
            arguments=[
                {"name": "description", "description": "Feature description", "required": True},
                {"name": "requirements", "description": "Technical requirements", "required": False}
            ]
        ))
    
    def add_resource(self, resource: MCPResource):
        """Add a resource to the server."""
        self.resources[resource.name] = resource
    
    def add_tool(self, tool: MCPTool):
        """Add a tool to the server."""
        self.tools[tool.name] = tool
    
    def add_prompt(self, prompt: MCPPrompt):
        """Add a prompt to the server."""
        self.prompts[prompt.name] = prompt
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return self._handle_initialize(request_id)
            elif method == "resources/list":
                return self._handle_list_resources(request_id)
            elif method == "tools/list":
                return self._handle_list_tools(request_id)
            elif method == "prompts/list":
                return self._handle_list_prompts(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(params, request_id)
            elif method == "resources/read":
                return self._handle_resource_read(params, request_id)
            elif method == "prompts/get":
                return self._handle_prompt_get(params, request_id)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
        except Exception as e:
            return self._error_response(request_id, -32603, str(e))
    
    def _handle_initialize(self, request_id: int) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "resources": {"list": True, "read": True},
                    "tools": {"list": True, "call": True},
                    "prompts": {"list": True, "get": True}
                },
                "serverInfo": {
                    "name": "agentic-coder-mcp",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_list_resources(self, request_id: int) -> Dict[str, Any]:
        """Handle resource list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": [
                    {
                        "uri": r.uri,
                        "name": r.name,
                        "description": r.description,
                        "mimeType": r.mimeType
                    }
                    for r in self.resources.values()
                ]
            }
        }
    
    def _handle_list_tools(self, request_id: int) -> Dict[str, Any]:
        """Handle tool list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema
                    }
                    for t in self.tools.values()
                ]
            }
        }
    
    def _handle_list_prompts(self, request_id: int) -> Dict[str, Any]:
        """Handle prompt list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "prompts": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "arguments": p.arguments
                    }
                    for p in self.prompts.values()
                ]
            }
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Handle tool call request."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")
        
        # Import our tools and execute
        from .tools import ToolRegistry
        tools = ToolRegistry(self.repo_path)
        
        try:
            result = tools.dispatch(tool_name.replace("_", ""), tool_args)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": result}
            }
        except Exception as e:
            return self._error_response(request_id, -32603, f"Tool execution failed: {e}")
    
    def _handle_resource_read(self, params: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Handle resource read request."""
        uri = params.get("uri")
        
        # For now, just handle file URIs
        if uri.startswith("file://"):
            path = uri[7:]  # Remove file:// prefix
            try:
                with open(path, 'r') as f:
                    content = f.read()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/plain",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return self._error_response(request_id, -32603, f"Failed to read resource: {e}")
        
        return self._error_response(request_id, -32602, f"Unsupported URI scheme: {uri}")
    
    def _handle_prompt_get(self, params: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Handle prompt get request."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if prompt_name not in self.prompts:
            return self._error_response(request_id, -32602, f"Unknown prompt: {prompt_name}")
        
        prompt = self.prompts[prompt_name]
        
        # Generate prompt content based on arguments
        if prompt_name == "code_review":
            content = f"Please review the following code changes:\n\n{arguments.get('diff', '')}"
        elif prompt_name == "bug_fix":
            content = f"Please analyze and fix this bug:\nError: {arguments.get('error', '')}\nContext: {arguments.get('context', 'None provided')}"
        elif prompt_name == "feature_implementation":
            content = f"Please implement the following feature:\nDescription: {arguments.get('description', '')}\nRequirements: {arguments.get('requirements', 'None specified')}"
        else:
            content = f"Prompt: {prompt_name}\nArguments: {json.dumps(arguments)}"
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": content}
                    }
                ]
            }
        }
    
    def _error_response(self, request_id: int, code: int, message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

class MCPClient:
    """
    MCP Client for connecting to external MCP servers.
    Allows Agentic Coder to use tools and resources from other systems.
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.config_file = os.path.expanduser("~/.agentic_mcp.json")
        self._load_config()
    
    def _load_config(self):
        """Load MCP server configurations."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                for server_name, server_config in config.get("servers", {}).items():
                    self.add_server(
                        server_name,
                        server_config.get("command"),
                        server_config.get("args", []),
                        server_config.get("env", {})
                    )
            except Exception as e:
                logger.error(f"Failed to load MCP config: {e}")
    
    def save_config(self):
        """Save MCP server configurations."""
        config = {
            "servers": {
                name: {
                    "command": conn.command,
                    "args": conn.args,
                    "env": conn.env
                }
                for name, conn in self.servers.items()
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save MCP config: {e}")
    
    def add_server(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        """Add an MCP server configuration."""
        self.servers[name] = MCPServerConnection(name, command, args or [], env or {})
    
    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration."""
        if name in self.servers:
            if self.servers[name].connected:
                asyncio.run(self.servers[name].disconnect())
            del self.servers[name]
            self.save_config()
            return True
        return False
    
    async def connect_server(self, name: str) -> bool:
        """Connect to an MCP server."""
        if name not in self.servers:
            return False
        
        server = self.servers[name]
        return await server.connect()
    
    async def disconnect_server(self, name: str) -> bool:
        """Disconnect from an MCP server."""
        if name not in self.servers:
            return False
        
        server = self.servers[name]
        return await server.disconnect()
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all configured servers."""
        return [
            {
                "name": name,
                "command": conn.command,
                "connected": conn.connected,
                "capabilities": conn.capabilities if conn.connected else None
            }
            for name, conn in self.servers.items()
        ]
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific server."""
        if server_name not in self.servers:
            raise ValueError(f"Unknown server: {server_name}")
        
        server = self.servers[server_name]
        if not server.connected:
            if not await server.connect():
                raise ConnectionError(f"Failed to connect to server: {server_name}")
        
        return await server.call_tool(tool_name, arguments)
    
    async def read_resource(self, server_name: str, uri: str) -> Any:
        """Read a resource from a specific server."""
        if server_name not in self.servers:
            raise ValueError(f"Unknown server: {server_name}")
        
        server = self.servers[server_name]
        if not server.connected:
            if not await server.connect():
                raise ConnectionError(f"Failed to connect to server: {server_name}")
        
        return await server.read_resource(uri)
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any]) -> Any:
        """Get a prompt from a specific server."""
        if server_name not in self.servers:
            raise ValueError(f"Unknown server: {server_name}")
        
        server = self.servers[server_name]
        if not server.connected:
            if not await server.connect():
                raise ConnectionError(f"Failed to connect to server: {server_name}")
        
        return await server.get_prompt(prompt_name, arguments)

class MCPServerConnection:
    """Manages a connection to an MCP server."""
    
    def __init__(self, name: str, command: str, args: List[str], env: Dict[str, str]):
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self.connected = False
        self.capabilities: Dict[str, Any] = {}
        self._request_id = 0
    
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            # Start the server process
            env = os.environ.copy()
            env.update(self.env)
            
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Send initialization request
            response = await self._send_request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "agentic-coder",
                    "version": "1.0.0"
                }
            })
            
            if response and "result" in response:
                self.capabilities = response["result"].get("capabilities", {})
                self.connected = True
                return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.name}: {e}")
        
        return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the MCP server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            
            self.process = None
            self.connected = False
            return True
        
        return False
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a request to the server."""
        if not self.process:
            return None
        
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params
        }
        
        try:
            # Send request
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str)
            self.process.stdin.flush()
            
            # Read response
            response_str = self.process.stdout.readline()
            if response_str:
                return json.loads(response_str)
            
        except Exception as e:
            logger.error(f"Failed to send request to {self.name}: {e}")
        
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server."""
        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"Tool call failed: {response['error']['message']}")
        
        return None
    
    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the server."""
        response = await self._send_request("resources/read", {"uri": uri})
        
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"Resource read failed: {response['error']['message']}")
        
        return None
    
    async def get_prompt(self, prompt_name: str, arguments: Dict[str, Any]) -> Any:
        """Get a prompt from the server."""
        response = await self._send_request("prompts/get", {
            "name": prompt_name,
            "arguments": arguments
        })
        
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"Prompt get failed: {response['error']['message']}")
        
        return None

# Integration with Agentic Coder tools
def create_mcp_tool_wrapper(client: MCPClient, server_name: str, tool_name: str) -> Callable:
    """Create a wrapper function for an MCP tool."""
    async def wrapper(**kwargs):
        return await client.call_tool(server_name, tool_name, kwargs)
    
    # Make it sync for compatibility with our tool system
    def sync_wrapper(**kwargs):
        return asyncio.run(wrapper(**kwargs))
    
    return sync_wrapper

def register_mcp_tools(tool_registry, mcp_client: MCPClient):
    """Register MCP tools with the Agentic Coder tool registry."""
    for server_name in mcp_client.servers:
        if not mcp_client.servers[server_name].connected:
            continue
        
        # Get available tools from the server
        # This would require listing tools from the server
        # For now, we'll just register known tools
        
        # Example: Register Google Drive tools if connected
        if server_name == "google_drive":
            tool_registry._tools[f"mcp_{server_name}_list_files"] = create_mcp_tool_wrapper(
                mcp_client, server_name, "list_files"
            )
            tool_registry._tools[f"mcp_{server_name}_read_file"] = create_mcp_tool_wrapper(
                mcp_client, server_name, "read_file"
            )