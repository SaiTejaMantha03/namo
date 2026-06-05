from __future__ import annotations

from urllib.parse import urlparse

from .config import AgentConfig
from .models import BrowserAction


SUBMIT_WORDS = ("submit", "apply", "send", "purchase", "pay", "confirm", "delete")


def domain_allowed(url: str, config: AgentConfig) -> bool:
    if not config.allowed_domains:
        return True
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in config.allowed_domains)


def needs_human_confirmation(action: BrowserAction, config: AgentConfig) -> bool:
    if config.allow_submit:
        return False
    haystack = " ".join(filter(None, [action.reason, action.text, action.value])).lower()
    return action.action.value in {"click"} and any(word in haystack for word in SUBMIT_WORDS)
