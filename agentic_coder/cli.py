from __future__ import annotations
import os
import sys
import typer
from rich import print
from dotenv import load_dotenv
from .runtime import Agent
from .config import MODEL_PRESETS
from .repl import repl

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(add_completion=False)

@app.command()
def models():
    for k, v in MODEL_PRESETS.items():
        print(f"[bold]{k}[/bold] -> {v.id}{' (remote)' if v.remote else ''}")

@app.command()
def run(model: str = typer.Option(None, help="Model name (e.g., claude-3-5-sonnet, gpt-4o, qwen2.5-coder-14b)"),
        repo: str = typer.Option(".", help="Path to a git repo or project root"),
        goal: str = typer.Option(None, help="High-level goal for the agent"),
        max_iters: int = typer.Option(20, help="Max reasoning/acting iterations"),
        debug: bool = typer.Option(False, help="Enable debug output"),
        base_url: str = typer.Option(None, help="API base URL (overrides env)"),
        api_key: str = typer.Option(None, help="API key (overrides env)"),
        resume: bool = typer.Option(False, help="Show session picker to resume a previous conversation"),
        continue_session: bool = typer.Option(False, "--continue", help="Automatically continue the most recent session")):
    if not os.path.isdir(repo):
        raise typer.BadParameter("repo must be a directory")
    
    # Auto-detect model based on available credentials if not specified
    if model is None:
        if os.environ.get("ANTHROPIC_API_KEY") or api_key:
            model = "claude-opus-4.1"  # Default to best Anthropic model (Opus 4.1)
            if debug:
                print("[dim]Auto-detected Anthropic credentials, using claude-4-opus[/dim]")
        elif os.environ.get("OPENAI_API_KEY"):
            model = "gpt-5"  # Default to best OpenAI model
            if debug:
                print("[dim]Auto-detected OpenAI credentials, using gpt-5[/dim]")
        else:
            # Default to local model if no API keys found
            model = "qwen2.5-coder-14b"
            if debug:
                print("[dim]No API credentials found, using local model qwen2.5-coder-14b[/dim]")
    
    # Set debug mode globally
    os.environ['AGENTIC_DEBUG'] = '1' if debug else '0'
    
    # If no goal provided, launch REPL mode
    if goal is None:
        repl(repo=repo, model=model, debug=debug, 
             base_url=base_url, api_key=api_key,
             resume=resume, continue_session=continue_session)
    else:
        agent = Agent(model=model, repo=repo,
                      base_url=base_url, api_key=api_key)
        agent.run(goal=goal, max_iters=max_iters, debug=debug)

def main():
    # If no arguments provided or no recognized command, default to 'run'
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ['models', 'run', '--help']):
        # Insert 'run' as the default command
        sys.argv.insert(1, 'run')
    app()

if __name__ == "__main__":
    main()