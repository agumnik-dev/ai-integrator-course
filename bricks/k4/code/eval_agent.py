"""Check which policies the agent requests against known correct answers.

Run from the course root:
    uv run python -m bricks.k4.code.eval_agent
"""

import json

from bricks.k4.code.policy_agent import DATA_DIR, run_agent
from common.llm import get_client

# The test set: letter file -> policy numbers the agent must request with get_policy.
EXPECTED = json.loads((DATA_DIR / "expected_agent.json").read_text(encoding="utf-8"))


def main() -> None:
    client = get_client()

    correct = 0
    errors = []
    cost = 0.0

    for file_name, expected in EXPECTED.items():
        letter = (DATA_DIR / file_name).read_text(encoding="utf-8")
        answer, asked, letter_cost = run_agent(client, letter)
        cost += letter_cost

        print(f"=== {file_name} ===")
        print(answer)
        print(f"tool calls: {asked}")
        print()

        # Order does not matter, so compare sorted lists.
        if sorted(asked) == sorted(expected):
            correct += 1
        else:
            errors.append(f"{file_name}: expected {expected}, got {asked}")

    print(f"tool calls correct: {correct} of {len(EXPECTED)}")
    for error in errors:
        print(f"  ✗ {error}")
    print(f"cost: ${cost:.5f}")


if __name__ == "__main__":
    main()
