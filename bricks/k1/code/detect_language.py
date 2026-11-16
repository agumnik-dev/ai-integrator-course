"""Worked example for brick K1: which language is a customer email written in?

Run from the course root:
    uv run python -m bricks.k1.code.detect_language "שלום, מתי תגיע הפוליסה החדשה שלי?"
"""

import sys

from common.llm import estimate_cost, extract_text, get_client

MODEL = "claude-haiku-5-5"

SYSTEM = """You work for an insurance company.
You receive a customer email.
Determine which language it is written in.
Answer with one word: English, Russian, Hebrew or Other."""


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python -m bricks.k1.code.detect_language "email text"')
    email = sys.argv[1]

    client = get_client()
    reply = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": email}],
    )

    print(extract_text(reply))
    print(f"[{reply.usage.input_tokens} in, {reply.usage.output_tokens} out, ${estimate_cost(reply.usage):.5f}]")


if __name__ == "__main__":
    main()
