from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .config import load_config
from .llm import build_planner
from .memory import JsonlMemory
from .orchestrator import BrowserAgent
from .profile import load_profile

app = typer.Typer(help="Run a modular autonomous browser agent.")
console = Console()


@app.callback()
def main() -> None:
    """Autonomous browser agent commands."""


@app.command()
def run(
    goal: str = typer.Argument(..., help="Goal for the browser agent."),
    profile: Path | None = typer.Option(None, "--profile", "-p", help="YAML profile data for forms and QA."),
    headless: bool = typer.Option(False, "--headless/--headed", help="Run browser without a visible window."),
    max_steps: int = typer.Option(20, "--max-steps", min=1, max=100, help="Maximum control-loop iterations."),
    memory: Path = typer.Option(Path("runs/latest.jsonl"), "--memory", help="JSONL run log path."),
) -> None:
    config = load_config(headless=headless, max_steps=max_steps)
    profile_data = load_profile(profile)
    planner = build_planner(config)
    agent = BrowserAgent(config=config, planner=planner, profile=profile_data, memory=JsonlMemory(memory), console=console)
    asyncio.run(agent.run(goal))


if __name__ == "__main__":
    app()
