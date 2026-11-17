"""my test of Pydantic model.

Run from the course root:
    uv run python -m bricks.k2.code.extract_letter bricks/k2/data/letter.txt
"""

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import estimate_cost, get_client

MODEL = "claude-haiku-5-5"

SYSTEM = """You work for an insurance company.
You receive a customer email. It can be in English, Russian or Hebrew.
Extract the fields described in the schema.
If a value is not written in the email, return null. Never guess."""


class CustomerLetter(BaseModel):
    customer_name: str | None = Field(description="Customer name")
    policy_number: str | None = Field(description="The policy number. Null if not written, never guess")
    # claim_number: str | None = Field(description="Our claim number, like CL-5610. Not an airline, police or other report number. Null if none.")
    claim_number: str | None = Field(description="Our claim number, like 5610. Not an airline, police or other report number. Null if none.")
    topic: Literal["new_claim", "claim_status", "coverage_question", "other"] = Field(
    description="new_claim: the customer reports a new event (accident, theft, loss), "
    "even if they also ask about coverage; "
    "claim_status: asks about a claim that is already opened; "
    "coverage_question: asks what the policy covers, without a new event; "
    "other: anything else"
)
    language: Literal["English", "Russian", "Hebrew", "Other"] = Field(description="Language the email is written in")
    request: str = Field(description = "one short sentence in Russian: what the customer wants")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: uv run python -m bricks.k2.code.extract_letter path/to/letter.txt")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")

    client = get_client()
    reply = client.messages.parse(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": text}],
        output_format=CustomerLetter,
    )
    letter = reply.parsed_output  # an object of CustomerLetter, not text

    print(letter)

    # Strict rules are code, not the model.
    if letter.policy_number is None:
        print("-> no policy number supplied: ask the customer for the policy number")

    print(f"[${estimate_cost(reply.usage):.5f}]")


if __name__ == "__main__":
    main()
