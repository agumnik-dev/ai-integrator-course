"""Brick K1: summarize a customer email in one sentence in Russian.

Run from the course root:
    uv run python -m bricks.k1.code.summarize bricks/k1/data/letter.txt
"""

import sys
from pathlib import Path

from common.llm import estimate_cost, extract_text, get_client

MODEL = "claude-haiku-5-5"

SYSTEM = """You work for an insurance company.
You receive a customer email.
Summarize it in one sentence for an employee, in Russian, not more than 15 words, provide what happened and what the customer wants.
Do not add anything that is not in the email."""


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python -m bricks.k1.code.summarize "email file path"')
    email_file_path = sys.argv[1]

    email_file = Path(email_file_path)
    if not email_file.exists():
        sys.exit('File not found: ' + email_file_path)

    email_content = email_file.read_text(encoding="utf-8")

    client = get_client()
    reply = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": email_content}],
    )

    print(extract_text(reply))
    print(f"[{reply.usage.input_tokens} in, {reply.usage.output_tokens} out, ${estimate_cost(reply.usage):.5f}]")


if __name__ == "__main__":
    main()
