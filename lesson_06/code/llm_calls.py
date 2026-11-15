"""Ready-made model calls for lesson 6. You don't need to change this file.

ask_structured() -> the model fills a Pydantic form (like lesson 3).
ask_text()       -> the model writes plain text.
Both add the cost of every call to one total: total_cost().
"""

import threading
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import estimate_cost, extract_text

MODEL = "claude-haiku-5-5"


class Route(BaseModel):
    category: Literal["new_claim", "claim_status", "policy_question", "cancel_policy", "other"]


class Facts(BaseModel):
    customer_name: str | None = Field(description="Customer's name as signed, or null")
    policy_number: str | None = Field(description="Policy number exactly as written, or null if missing")
    summary: str = Field(description="One sentence in English: what happened and what the customer wants")


class Verdict(BaseModel):
    ok: bool = Field(description="True only if ALL rules pass")
    problems: list[str] = Field(description="One short line per broken rule; empty list if ok")


_total_cost = 0.0
_lock = threading.Lock()


def _add_cost(reply) -> None:
    global _total_cost
    # In step 4 several calls run at the same time; the lock makes them add one by one.
    with _lock:
        _total_cost += estimate_cost(reply.usage, MODEL)


def total_cost() -> float:
    return _total_cost


def ask_structured(client, system: str, content: str, output_format):
    reply = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        system=system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
        output_format=output_format,
    )
    _add_cost(reply)
    return reply.parsed_output


def ask_text(client, system: str, content: str) -> str:
    reply = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
    )
    _add_cost(reply)
    return extract_text(reply)
