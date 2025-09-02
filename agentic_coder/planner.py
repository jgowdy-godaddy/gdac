import os, json
from .tools.ast_parser import enhanced_repo_map

USER_TEMPLATE = """<GOAL>
{goal}
</GOAL>
<CONTEXT>
repo={repo}

{repo_context}
</CONTEXT>
"""

def make_prompt(goal: str, repo: str) -> str:
    # DON'T load agent instructions here - they're now in the system prompt
    # Only create the user message with goal and context
    
    # Generate repository context using enhanced AST mapping (with repo_map fallback)
    repo_context = ""
    try:
        repo_context = enhanced_repo_map(repo, max_tokens=800, include_tests=False, use_ast=True)
    except Exception as e:
        repo_context = f"Repository context unavailable: {e}"
    
    # Simple user prompt with goal and context ONLY
    # Agent instructions are now in the system prompt
    return USER_TEMPLATE.format(goal=goal, repo=repo, repo_context=repo_context)