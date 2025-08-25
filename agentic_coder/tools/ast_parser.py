from __future__ import annotations
import os
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import fnmatch

# Try to import tree-sitter, but fall back gracefully if not available
AST_AVAILABLE = False
try:
    # Test if tree-sitter is properly available
    import tree_sitter
    # For now, disable AST due to compatibility issues with tree_sitter_languages
    # This can be re-enabled when a compatible version is available
    AST_AVAILABLE = False  # Temporarily disabled
    print("Tree-sitter available but disabled due to compatibility issues")
except ImportError:
    AST_AVAILABLE = False

from .repo_map import repo_map

"""
AST Parser Tool: Enhanced repository analysis using tree-sitter with repo_map fallback.
Provides accurate symbol extraction and code structure analysis.

NOTE: Currently falls back to repo_map due to compatibility issues with tree_sitter_languages.
To enable full AST functionality:
1. Fix tree_sitter_languages compatibility or use alternative bindings
2. Set AST_AVAILABLE = True after fixing imports
3. Test with: enhanced_repo_map('.', use_ast=True)

The system gracefully falls back to regex-based symbol extraction via repo_map.
"""

# Language mapping for tree-sitter
LANGUAGE_MAP = {
    '.py': 'python',
    '.js': 'javascript', 
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.go': 'go',
    '.java': 'java',
    '.rs': 'rust',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
}

def _get_language_for_file(file_path: str) -> Optional[str]:
    """Get tree-sitter language name for a file."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_MAP.get(ext)

def _extract_python_symbols_ast(node: Node, source: bytes) -> List[str]:
    """Extract Python symbols using tree-sitter AST."""
    symbols = []
    
    def traverse(node: Node):
        if node.type == 'class_definition':
            class_name = None
            for child in node.children:
                if child.type == 'identifier':
                    class_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    break
            if class_name:
                symbols.append(f"class {class_name}")
        
        elif node.type == 'function_definition':
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    break
            if func_name:
                symbols.append(f"def {func_name}")
        
        elif node.type == 'async_function_definition':
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    break
            if func_name:
                symbols.append(f"async def {func_name}")
        
        for child in node.children:
            traverse(child)
    
    traverse(node)
    return symbols

def _extract_javascript_symbols_ast(node: Node, source: bytes) -> List[str]:
    """Extract JavaScript/TypeScript symbols using tree-sitter AST."""
    symbols = []
    
    def traverse(node: Node):
        if node.type == 'class_declaration':
            class_name = None
            for child in node.children:
                if child.type == 'identifier':
                    class_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    break
            if class_name:
                symbols.append(f"class {class_name}")
        
        elif node.type == 'function_declaration':
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    break
            if func_name:
                symbols.append(f"function {func_name}")
        
        elif node.type == 'method_definition':
            method_name = None
            for child in node.children:
                if child.type == 'property_identifier':
                    method_name = source[child.start_byte:child.end_byte].decode('utf-8')
                    break
            if method_name:
                symbols.append(f"method {method_name}")
        
        elif node.type in ['variable_declaration', 'lexical_declaration']:
            # Look for arrow functions and assignments
            for child in node.children:
                if child.type == 'variable_declarator':
                    var_name = None
                    has_function = False
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            var_name = source[subchild.start_byte:subchild.end_byte].decode('utf-8')
                        elif subchild.type in ['arrow_function', 'function_expression']:
                            has_function = True
                    if var_name and has_function:
                        symbols.append(f"const {var_name} = ...")
        
        for child in node.children:
            traverse(child)
    
    traverse(node)
    return symbols

def _extract_symbols_ast(file_path: str, content: str, language: str) -> List[str]:
    """Extract symbols using tree-sitter AST parsing."""
    if not AST_AVAILABLE:
        return []
    
    try:
        parser = get_language_safe(language)
        if not parser:
            return []
        
        source_bytes = content.encode('utf-8')
        tree = parser.parse(source_bytes)
        
        if language == 'python':
            return _extract_python_symbols_ast(tree.root_node, source_bytes)
        elif language in ['javascript', 'typescript']:
            return _extract_javascript_symbols_ast(tree.root_node, source_bytes)
        else:
            # For other languages, extract basic structure
            symbols = []
            def traverse(node: Node):
                # Generic extraction for other languages
                if node.type in ['class_declaration', 'struct_item', 'impl_item']:
                    text = source_bytes[node.start_byte:min(node.start_byte + 100, node.end_byte)].decode('utf-8', errors='ignore')
                    symbols.append(text.split('\n')[0].strip())
                for child in node.children:
                    traverse(child)
            traverse(tree.root_node)
            return symbols[:10]  # Limit generic extractions
    except Exception:
        return []

def enhanced_repo_map(repo: str, max_tokens: int = 1000, include_tests: bool = False, 
                     use_ast: bool = True, file_patterns: Optional[List[str]] = None) -> str:
    """Enhanced repository mapping with AST parsing and repo_map fallback."""
    
    # If AST is not available or disabled, fall back to repo_map
    if not use_ast or not AST_AVAILABLE:
        return repo_map(repo, max_tokens, include_tests, file_patterns)
    
    # Try AST-enhanced mapping, fall back to repo_map on failure
    try:
        return _ast_repo_map(repo, max_tokens, include_tests, file_patterns)
    except Exception:
        return repo_map(repo, max_tokens, include_tests, file_patterns)

def _ast_repo_map(repo: str, max_tokens: int, include_tests: bool, 
                  file_patterns: Optional[List[str]]) -> str:
    """AST-powered repository mapping."""
    from .repo_map import _get_git_files, _should_include_file, _format_repo_map
    
    if file_patterns is None:
        file_patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", 
                        "*.java", "*.cpp", "*.c", "*.h", "*.rs", "*.rb"]
    
    # Get git-tracked files
    git_files = _get_git_files(repo)
    if not git_files:
        return "No git repository or no tracked files found."
    
    # Build enhanced file map using AST when possible
    file_map = {}
    
    for file_path in git_files:
        if not _should_include_file(file_path, file_patterns, include_tests):
            continue
        
        full_path = os.path.join(repo, file_path)
        if not os.path.exists(full_path):
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = _get_language_for_file(file_path)
            symbols = []
            
            if language:
                # Try AST extraction first
                symbols = _extract_symbols_ast(file_path, content, language)
            
            # Fall back to regex extraction if AST fails or language not supported
            if not symbols:
                from .repo_map import _extract_symbols_by_extension
                symbols = _extract_symbols_by_extension(file_path, content)
            
            if symbols:
                file_map[file_path] = symbols
                
        except Exception:
            continue
    
    if not file_map:
        return "No matching files found for repository mapping."
    
    # Format with AST indicator
    result = _format_repo_map(file_map, max_tokens)
    if AST_AVAILABLE:
        result = "AST-ENHANCED " + result
    
    return result

def analyze_code_structure(repo: str, file_path: str = "", **kwargs) -> str:
    """Analyze code structure for a specific file or the whole repository."""
    if not AST_AVAILABLE:
        return "AST analysis unavailable. Install tree-sitter dependencies."
    
    if file_path:
        # Analyze specific file
        full_path = os.path.join(repo, file_path)
        if not os.path.exists(full_path):
            return f"File not found: {file_path}"
        
        language = _get_language_for_file(file_path)
        if not language:
            return f"Language not supported for AST analysis: {file_path}"
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            symbols = _extract_symbols_ast(file_path, content, language)
            return f"Code structure for {file_path}:\n" + "\n".join(f"  {sym}" for sym in symbols)
        except Exception as e:
            return f"Failed to analyze {file_path}: {e}"
    else:
        # Analyze whole repository
        return enhanced_repo_map(repo, max_tokens=1500, use_ast=True)

def parse_ast(repo: str, file_path: str, **kwargs) -> str:
    """Parse a single file and return AST structure information."""
    if not AST_AVAILABLE:
        return "AST parsing unavailable. Install tree-sitter dependencies."
    
    full_path = os.path.join(repo, file_path)
    if not os.path.exists(full_path):
        return f"File not found: {file_path}"
    
    language = _get_language_for_file(file_path)
    if not language:
        return f"Language not supported: {file_path}"
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = get_language_safe(language)
        if not parser:
            return f"Language not supported or parser unavailable: {language}"
        
        tree = parser.parse(content.encode('utf-8'))
        
        def node_summary(node: Node, depth: int = 0) -> str:
            indent = "  " * depth
            result = f"{indent}{node.type}"
            if depth < 3:  # Limit depth to avoid overwhelming output
                for child in node.children[:5]:  # Limit children
                    result += "\n" + node_summary(child, depth + 1)
            return result
        
        return f"AST for {file_path} ({language}):\n{node_summary(tree.root_node)}"
    except Exception as e:
        return f"Failed to parse AST: {e}"