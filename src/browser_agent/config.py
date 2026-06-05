from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentConfig:
    model: str
    allowed_domains: tuple[str, ...]
    allow_submit: bool
    headless: bool
    max_steps: int


def load_config(headless: bool, max_steps: int) -> AgentConfig:
    load_dotenv()
    domains = tuple(
        domain.strip().lower()
        for domain in os.getenv("BROWSER_AGENT_ALLOWED_DOMAINS", "").split(",")
        if domain.strip()
    )
    return AgentConfig(
        model=os.getenv("BROWSER_AGENT_MODEL", "gpt-4.1"),
        allowed_domains=domains,
        allow_submit=os.getenv("BROWSER_AGENT_ALLOW_SUBMIT", "false").lower() == "true",
        headless=headless,
        max_steps=max_steps,
    )
