from __future__ import annotations
import os
import typer
from rich import print
from .runtime import Agent
from .config import MODEL_PRESETS

app = typer.Typer(add_completion=False)

@app.command()
def models():
    for k, v in MODEL_PRESETS.items():
        print(f"[bold]{k}[/bold] -> {v.id}{' (remote)' if v.remote else ''}")

@app.command()
def run(model: str = typer.Option("qwen2.5-coder-14b", help="Model preset key"),
        repo: str = typer.Option(".", help="Path to a git repo or project root"),
        goal: str = typer.Option(..., help="High-level goal for the agent"),
        max_iters: int = typer.Option(20, help="Max reasoning/acting iterations"),
        remote_model: str = typer.Option(None, help="Remote model name (for remote presets)"),
        remote_base_url: str = typer.Option(None, help="Remote base URL (overrides env)"),
        remote_api_key: str = typer.Option(None, help="Remote API key (overrides env)")):
    if not os.path.isdir(repo):
        raise typer.BadParameter("repo must be a directory")
    agent = Agent(model, repo, remote_model=remote_model,
                  remote_base_url=remote_base_url, remote_api_key=remote_api_key)
    agent.run(goal=goal, max_iters=max_iters)

if __name__ == "__main__":
    app()