"""Shared helpers for all course exercises.

Boilerplate lives here so lesson code can focus on the AI part.
Usage (run from the course root):
    from common.llm import get_client, extract_text, estimate_cost, warn_if_truncated
"""

import os
import re
import sys

import anthropic
from anthropic.types import Message, Usage
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-haiku-5-5"

# USD per 1M tokens: (input, output). Check https://claude.com/pricing before use.
# Note: Haiku 5.5 charges more for prompts over 100K tokens; ignored here.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
}


def get_client() -> anthropic.Anthropic:
    """Create an API client. Exits with a clear message if the key is missing."""
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not found. Copy .env.example to .env and add your key.")
    # SDK retries connection errors, 429 and 5xx automatically.
    return anthropic.Anthropic(max_retries=2, timeout=60.0)


def extract_text(message: Message) -> str:
    """Join all text blocks. Never use content[0].text: a thinking block may come first."""
    return "".join(block.text for block in message.content if block.type == "text")


def estimate_cost(usage: Usage, model: str = DEFAULT_MODEL) -> float:
    """Return the cost of one API call in USD."""
    if model not in PRICES:
        raise ValueError(f"No price for model {model!r}. Add it to PRICES in common/llm.py.")
    price_in, price_out = PRICES[model]
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000


def extract_tag(text: str, tag: str) -> str | None:
    """Return the content of the first <tag>...</tag> in text (stripped), or None if absent."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def warn_if_truncated(message: Message) -> None:
    """The API does not raise on truncation, so we have to check stop_reason ourselves."""
    if message.stop_reason == "max_tokens":
        print("\n[warning] response truncated: max_tokens reached", file=sys.stderr)
