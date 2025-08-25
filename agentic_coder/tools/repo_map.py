from __future__ import annotations
import os
import subprocess
from typing import Dict, List, Set, Optional
from pathlib import Path
import fnmatch

"""
Tool: repo_map
Description: Generate a concise repository map showing key files and symbols, similar to Aider's approach.
Uses git ls-files for discovery and basic pattern matching for symbol extraction.
Args: {
  "max_tokens": 1000,        # Token budget for the map
  "include_tests": false,    # Include test files
  "file_patterns": ["*.py", "*.js", "*.ts", "*.go", "*.java", "*.cpp", "*.c", "*.h"]
}
"""

def _get_git_files(repo: str) -> List[str]:
    """Get tracked files from git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"], 
            cwd=repo, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError:
        return []

def _should_include_file(file_path: str, patterns: List[str], include_tests: bool) -> bool:
    """Determine if file should be included in the map."""
    # Skip test files if not requested
    if not include_tests:
        test_indicators = ['test_', '_test.', '/tests/', '/test/', 'spec_', '_spec.']
        if any(indicator in file_path.lower() for indicator in test_indicators):
            return False
    
    # Skip common exclusions
    exclusions = [
        '__pycache__', '.git/', '.venv/', 'node_modules/', 
        '.pytest_cache/', 'dist/', 'build/', '.idea/', '.vscode/',
        '*.pyc', '*.pyo', '*.egg-info', '.DS_Store'
    ]
    
    for exclusion in exclusions:
        if fnmatch.fnmatch(file_path, exclusion) or exclusion in file_path:
            return False
    
    # Check if file matches any of the patterns
    return any(fnmatch.fnmatch(file_path, pattern) for pattern in patterns)

def _extract_python_symbols(content: str) -> List[str]:
    """Extract key Python symbols (classes, functions) with basic parsing."""
    symbols = []
    lines = content.split('\n')
    
    for line in lines:
        stripped = line.strip()
        # Class definitions
        if stripped.startswith('class ') and ':' in stripped:
            class_def = stripped.split(':')[0].strip()
            symbols.append(class_def)
        # Function definitions (top-level and methods)
        elif stripped.startswith('def ') and ':' in stripped:
            func_def = stripped.split(':')[0].strip()
            symbols.append(func_def)
        # Async function definitions
        elif stripped.startswith('async def ') and ':' in stripped:
            func_def = stripped.split(':')[0].strip()
            symbols.append(func_def)
    
    return symbols

def _extract_javascript_symbols(content: str) -> List[str]:
    """Extract key JavaScript/TypeScript symbols."""
    symbols = []
    lines = content.split('\n')
    
    for line in lines:
        stripped = line.strip()
        # Class definitions
        if stripped.startswith('class '):
            if '{' in stripped:
                class_def = stripped.split('{')[0].strip()
                symbols.append(class_def)
        # Function declarations
        elif stripped.startswith('function '):
            if '(' in stripped:
                func_def = stripped.split('(')[0].strip() + '(...)'
                symbols.append(func_def)
        # Arrow functions and method definitions
        elif '=>' in stripped or stripped.endswith('{'):
            if any(keyword in stripped for keyword in ['const ', 'let ', 'var ', 'export ']):
                # Extract variable/function name
                for keyword in ['const ', 'let ', 'var ', 'export const ', 'export let ']:
                    if stripped.startswith(keyword):
                        name = stripped[len(keyword):].split('=')[0].strip()
                        if name:
                            symbols.append(f"{name} = ...")
                        break
    
    return symbols

def _extract_symbols_by_extension(file_path: str, content: str) -> List[str]:
    """Extract symbols based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.py':
        return _extract_python_symbols(content)
    elif ext in ['.js', '.ts', '.jsx', '.tsx']:
        return _extract_javascript_symbols(content)
    else:
        # For other languages, just return first few lines as context
        lines = content.split('\n')[:5]
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith('//')]

def _build_file_map(repo: str, files: List[str], patterns: List[str], include_tests: bool) -> Dict[str, List[str]]:
    """Build a map of files to their key symbols."""
    file_map = {}
    
    for file_path in files:
        if not _should_include_file(file_path, patterns, include_tests):
            continue
        
        full_path = os.path.join(repo, file_path)
        if not os.path.exists(full_path):
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            symbols = _extract_symbols_by_extension(file_path, content)
            if symbols:
                file_map[file_path] = symbols
                
        except Exception:
            continue
    
    return file_map

def _format_repo_map(file_map: Dict[str, List[str]], max_tokens: int) -> str:
    """Format the repository map as a string within token budget."""
    lines = ["REPOSITORY MAP:"]
    current_tokens = 2  # Rough estimate for header
    
    # Sort files by perceived importance (main files first, then by depth)
    def file_priority(file_path: str) -> int:
        priority = 0
        if 'main' in file_path or '__init__' in file_path:
            priority -= 100
        if file_path.count('/') == 0:  # Root level files
            priority -= 50
        if any(important in file_path for important in ['config', 'setup', 'cli', 'app']):
            priority -= 25
        return priority
    
    sorted_files = sorted(file_map.keys(), key=file_priority)
    
    for file_path in sorted_files:
        symbols = file_map[file_path]
        
        # Estimate tokens for this file section
        file_tokens = len(file_path.split()) + sum(len(sym.split()) for sym in symbols) + 2
        
        if current_tokens + file_tokens > max_tokens:
            lines.append("... (truncated due to token limit)")
            break
        
        lines.append(f"\n{file_path}:")
        for symbol in symbols[:10]:  # Limit symbols per file
            lines.append(f"  {symbol}")
        
        current_tokens += file_tokens
    
    return "\n".join(lines)

def repo_map(repo: str, max_tokens: int = 1000, include_tests: bool = False, 
             file_patterns: Optional[List[str]] = None) -> str:
    """Generate a repository map showing key files and symbols."""
    if file_patterns is None:
        file_patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", 
                        "*.java", "*.cpp", "*.c", "*.h", "*.rs", "*.rb"]
    
    # Get git-tracked files
    git_files = _get_git_files(repo)
    if not git_files:
        return "No git repository or no tracked files found."
    
    # Build map of files to symbols
    file_map = _build_file_map(repo, git_files, file_patterns, include_tests)
    
    if not file_map:
        return "No matching files found for repository mapping."
    
    # Format and return the map
    return _format_repo_map(file_map, max_tokens)