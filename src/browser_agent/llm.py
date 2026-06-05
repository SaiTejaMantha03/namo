from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI

from .config import AgentConfig
from .models import ActionType, AgentDecision, BrowserAction, PageObservation


class Planner(Protocol):
    def decide(self, goal: str, observation: PageObservation, profile: dict) -> AgentDecision:
        ...


class OpenAIPlanner:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OpenAI()

    def decide(self, goal: str, observation: PageObservation, profile: dict) -> AgentDecision:
        response = self.client.responses.parse(
            model=self.config.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the Planner Agent in a modular browser automation system. "
                        "Choose exactly one safe next browser action. Prefer using element refs "
                        "from the observation. Ask a human when confidence is low, credentials "
                        "are needed, CAPTCHA appears, payment is requested, or submission is irreversible."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Goal:\n{goal}\n\n"
                        f"Profile data:\n{profile}\n\n"
                        f"Current page observation:\n{observation.model_dump()}"
                    ),
                },
            ],
            text_format=AgentDecision,
        )
        return response.output_parsed


class HeuristicPlanner:
    """Tiny no-key planner for smoke tests and demos."""

    def decide(self, goal: str, observation: PageObservation, profile: dict) -> AgentDecision:
        if observation.url == "about:blank":
            maybe_url = next((word for word in goal.split() if word.startswith("http")), None)
            if maybe_url:
                action = BrowserAction(action=ActionType.GOTO, url=maybe_url, reason="Open the URL from the goal.")
            else:
                action = BrowserAction(
                    action=ActionType.ASK_HUMAN,
                    question="What URL should I open first?",
                    reason="No starting URL was provided.",
                )
            return AgentDecision(thought="Need a starting page.", confidence=0.7, action=action)

        for element in observation.elements:
            label = element.text.lower()
            role = element.role.lower()
            if role in {"input", "textarea"}:
                value = _profile_value_for_label(label, profile)
                if value:
                    return AgentDecision(
                        thought="Fill an obvious profile field.",
                        confidence=0.65,
                        action=BrowserAction(action=ActionType.FILL, ref=element.ref, value=value, reason=f"Fill {element.text}"),
                    )

        return AgentDecision(
            thought="No safe deterministic action found.",
            confidence=0.45,
            action=BrowserAction(
                action=ActionType.ASK_HUMAN,
                question="I do not see a safe next action. What should I do?",
                reason="The heuristic planner cannot continue confidently.",
            ),
        )


def build_planner(config: AgentConfig) -> Planner:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIPlanner(config)
    return HeuristicPlanner()


def _profile_value_for_label(label: str, profile: dict) -> str | None:
    label_map = {
        "name": profile.get("name"),
        "email": profile.get("email"),
        "phone": profile.get("phone"),
        "location": profile.get("location"),
    }
    for key, value in label_map.items():
        if key in label and value:
            return str(value)
    links = profile.get("links") or {}
    if "github" in label:
        return links.get("github")
    if "linkedin" in label:
        return links.get("linkedin")
    if "portfolio" in label or "website" in label:
        return links.get("portfolio")
    return None
