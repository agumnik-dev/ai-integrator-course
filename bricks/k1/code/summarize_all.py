"""Brick K1: one-line Russian summary for every letter in bricks/k1/data

Run from the course root:
    uv run python -m bricks.k1.code.summarize_all
"""

import sys
from pathlib import Path

from common.llm import estimate_cost, extract_text, get_client

MODEL = "claude-haiku-5-5"
DATA_DIR = Path("bricks/k1/data")

SYSTEM = """You work for an insurance company.
You receive a customer email.
Summarize it in one sentence for an employee, in Russian, not more than 15 words, provide what happened and what the customer wants.
Do not add anything that is not in the email."""


def main() -> None:
    # Sorted so the output order is stable between runs
    email_files = sorted(DATA_DIR.glob("*.txt"))
    if not email_files:
        sys.exit(f"No .txt files found in {DATA_DIR}")

    client = get_client()
    total_cost = 0.0

    for email_file in email_files:
        email_content = email_file.read_text(encoding="utf-8")

        reply = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": email_content}],
        )

        print(f"{email_file.name}: {extract_text(reply)}")
        total_cost += estimate_cost(reply.usage, MODEL)

    print(f"Total cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
