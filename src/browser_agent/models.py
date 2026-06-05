from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    GOTO = "goto"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    WAIT = "wait"
    ASK_HUMAN = "ask_human"
    DONE = "done"


class BrowserAction(BaseModel):
    action: ActionType
    reason: str
    url: str | None = None
    ref: str | None = None
    text: str | None = None
    value: str | None = None
    question: str | None = None


class AgentDecision(BaseModel):
    thought: str = Field(description="Short private rationale for the next move.")
    confidence: float = Field(ge=0, le=1)
    action: BrowserAction


class InteractiveElement(BaseModel):
    ref: str
    role: str
    text: str
    selector_hint: str | None = None


class PageObservation(BaseModel):
    url: str
    title: str
    text: str
    elements: list[InteractiveElement]


class RunEvent(BaseModel):
    kind: Literal["decision", "observation", "action", "error", "human"]
    payload: dict[str, Any]
