from __future__ import annotations

from rich.console import Console

from .browser import BrowserController
from .config import AgentConfig
from .llm import Planner
from .memory import JsonlMemory
from .models import ActionType
from .safety import domain_allowed, needs_human_confirmation


class BrowserAgent:
    def __init__(
        self,
        config: AgentConfig,
        planner: Planner,
        profile: dict,
        memory: JsonlMemory,
        console: Console | None = None,
    ):
        self.config = config
        self.planner = planner
        self.profile = profile
        self.memory = memory
        self.console = console or Console()

    async def run(self, goal: str) -> None:
        async with BrowserController(headless=self.config.headless) as browser:
            for step in range(1, self.config.max_steps + 1):
                observation = await browser.observe()
                self.memory.append("observation", observation.model_dump())

                if not domain_allowed(observation.url, self.config):
                    self.console.print(f"[red]Blocked by domain allowlist:[/red] {observation.url}")
                    self.memory.append("error", {"reason": "domain_not_allowed", "url": observation.url})
                    return

                decision = self.planner.decide(goal, observation, self.profile)
                self.memory.append("decision", decision.model_dump())
                self.console.print(f"[cyan]Step {step}[/cyan] {decision.action.action.value}: {decision.action.reason}")

                if decision.action.action == ActionType.DONE:
                    self.console.print("[green]Goal marked complete.[/green]")
                    return

                if decision.action.action == ActionType.ASK_HUMAN:
                    answer = self.console.input(f"[yellow]{decision.action.question or 'Input needed'}[/yellow]\n> ")
                    self.memory.append("human", {"question": decision.action.question, "answer": answer})
                    continue

                if decision.action.url and not domain_allowed(decision.action.url, self.config):
                    self.console.print(f"[red]Refusing navigation outside allowlist:[/red] {decision.action.url}")
                    return

                if needs_human_confirmation(decision.action, self.config):
                    answer = self.console.input("[yellow]Confirm this sensitive action? Type yes to continue.[/yellow]\n> ")
                    self.memory.append("human", {"confirmation_for": decision.action.model_dump(), "answer": answer})
                    if answer.strip().lower() != "yes":
                        self.console.print("Skipped sensitive action.")
                        return

                try:
                    result = await browser.execute(decision.action)
                    self.memory.append("action", {"result": result, "action": decision.action.model_dump()})
                except Exception as exc:
                    self.memory.append("error", {"error": repr(exc), "action": decision.action.model_dump()})
                    self.console.print(f"[red]Action failed:[/red] {exc}")

            self.console.print("[yellow]Stopped at max step limit.[/yellow]")
