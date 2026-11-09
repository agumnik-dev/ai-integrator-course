"""Lesson 5: the agent from lesson 4, made reliable.

Fill in TODO A (step 2) and TODO B (step 3). Until then, `raise` lets errors crash the program.

Run from the course root:
    uv run python -m lesson_05.code.agent "your question"
"""

import json
import sys

import anthropic

from common.llm import estimate_cost, extract_text, get_client
from lesson_05.code.tools import GET_CLAIMS_TOOL, GET_POLICY_TOOL, get_claims, get_policy

MODEL = "claude-haiku-5-5"
MAX_STEPS = 5

SYSTEM = """You are a support assistant of "Shomer" insurance company.
Use the tools to look up policy data. Answer only from what the tools return.
If a tool returns an error, tell the customer which information is temporarily
unavailable and that an agent will follow up. Never guess missing data.
Answer in the customer's language, in 2-3 sentences."""

TOOLS = [GET_POLICY_TOOL, GET_CLAIMS_TOOL]
FUNCTIONS = {"get_policy": get_policy, "get_claims": get_claims}

FALLBACK = "Sorry, the assistant is temporarily unavailable. Please try again later or call *2800."


def ask(client, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    cost = 0.0

    for step in range(MAX_STEPS):
        try:
            reply = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM,
                tools=TOOLS,
                output_config={"effort": "low"},
                messages=messages,
            )
        except anthropic.APIError as e:
            # TODO B (step 3): the API call failed even after the SDK's automatic retries.
            #   Replace `raise` with ONE line that returns the polite message for the customer:
            #   (Optional: before it, print the error for yourself: print(f"  [api error] {e}"))
            print(f"  [API error] {e}")
            return FALLBACK
            # raise

        cost += estimate_cost(reply.usage, MODEL)
        messages.append({"role": "assistant", "content": reply.content})

        if reply.stop_reason != "tool_use":
            print(f"  [{step + 1} model calls, ${cost:.4f}]")
            return extract_text(reply)

        results = []
        for block in reply.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")
                try:
                    function = FUNCTIONS[block.name]
                    output = function(**block.input)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    })
                except Exception as e:
                    # TODO A (step 2): our function failed (e.g. the database is down).
                    #   Replace `raise` with a tool_result that tells the MODEL about the error.
                    #   It is the same dict as in the `try` block above, with two differences:
                    print(f"[tool error] {block.name}: {e}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {e}",      # the error text instead of the data
                        "is_error": True,              # NEW: marks this result as an error
                    })
                    # raise

        messages.append({"role": "user", "content": results})

    return "Stopped: too many tool calls."


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python -m lesson_05.code.agent "your question"')
    client = get_client()
    print(ask(client, sys.argv[1]))


if __name__ == "__main__":
    main()
