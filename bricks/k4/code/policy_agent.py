"""Policy agent with one tool.

An accountant asks about a purchase order (PO). The model asks our code
to look the policy up, then answers from the data.

Run from the course root:
    uv run python -m bricks.k4.code.policy_agent letter_police.txt
"""

import json
import sys
from pathlib import Path

from common.llm import estimate_cost, extract_text, get_client

MODEL = "claude-haiku-5-5"
MAX_STEPS = 5
DATA_DIR = Path("bricks/k4/data")
POLICIES = json.loads((DATA_DIR / "policies.json").read_text(encoding="utf-8"))

SYSTEM = """You answer letters of insurance company's clients.
If the letters contains policy number - use get_policy tool to extract the policy.
Answer only from what the tool returns: if the case is covered and what is a deductible.
If the tool returns an error, say that the policy was not found and that our operator will call you. Never guess.
Answer in 2-3 sentences, in the language of the question."""

# 1. The tool for the model: its name, what it does, what input it needs.
GET_POLICY_TOOL = {
    "name": "get_policy",
    "description": "Look up a policy by its number. "
                   "Returns type, holder, status, coverage, not_covered, deductible in ILS.",
    "input_schema": {
        "type": "object",
        "properties": {
            "policy_number": {"type": "string", "holder": "string", "status": "active", "valid_until": "date YYYY-MM-DD", "coverage": "list of options", "not_covered": "string", "deductible_ils": "number"},
        },
        "required": ["policy_number"],
    },
}


# 2. Our function: the code that really does the work.
def get_policy(policy_number: str) -> dict | None:
    """Return the policy by its number, or None if there is no such policy."""
    return POLICIES.get(policy_number)


# 3. The agent loop.
def run_agent(client, letter: str) -> tuple[str, list[str], float]:

    """Return (answer, PO numbers the model asked for, cost)."""
    messages = [{"role": "user", "content": letter}]
    asked = []
    cost = 0.0

    for step in range(MAX_STEPS):
        reply = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM,
            tools=[GET_POLICY_TOOL],
            output_config={"effort": "low"},
            messages=messages,
        )
        cost += estimate_cost(reply.usage, MODEL)
        messages.append({"role": "assistant", "content": reply.content})

        # The model did not ask for a tool: this is the final answer.
        if reply.stop_reason != "tool_use":
            return extract_text(reply), asked, cost

        # The model asked for a tool: run it and send the results back.
        results = []
        for block in reply.content:
            if block.type != "tool_use":
                continue
            policy_number = block.input["policy_number"]
            asked.append(policy_number)
            policy = get_policy(policy_number)
            if policy is None:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Policy No. {policy_number} not found",
                    "is_error": True,
                })
            else:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(policy, ensure_ascii=False),
                })
        messages.append({"role": "user", "content": results})

    return "Stopped: step limit reached.", asked, cost


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: uv run python -m bricks.k4.code.policy_agent 'file_name'")
    path = (DATA_DIR / sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")

    # read letter content
    letter = path.read_text(encoding="utf-8")

    # get client
    client = get_client()

    # run agent return data
    answer, asked, cost = run_agent(client, letter)

    print(answer)
    print(f"tool calls: {asked}")
    print(f"cost: ${cost:.5f}")


if __name__ == "__main__":
    main()
