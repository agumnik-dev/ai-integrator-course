"""Step 3: the model may call tools many times; we loop until it answers.

Fill in the TODOs. Each one is 1-3 lines.
Look at one_call.py: every TODO here repeats one of its steps.

Run from the course root:
    uv run python -m lesson_04.code.agent "your question"
"""

import json
import sys

from common.llm import estimate_cost, extract_text, get_client
from lesson_04.code.tools import GET_CLAIMS_TOOL, GET_POLICY_TOOL, get_claims, get_policy

MODEL = "claude-haiku-5-5"
MAX_STEPS = 5  # safety limit: never more than 5 model calls per question

SYSTEM = """You are a support assistant of "Shomer" insurance company.
Use the tools to look up policy data. Answer only from what the tools return.
Answer in the customer's language, in 2-3 sentences."""

TOOLS = [GET_POLICY_TOOL, GET_CLAIMS_TOOL]
FUNCTIONS = {"get_policy": get_policy, "get_claims": get_claims}  # tool name -> our function


def ask(client, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    cost = 0.0

    for step in range(MAX_STEPS):
        # 1. Call the model with the whole history (messages) and ALL tools (TOOLS).
        #    Same call as step 1 in one_call.py, only tools=TOOLS instead of tools=[GET_POLICY_TOOL].
        reply = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            tools=TOOLS,
            output_config={"effort": "low"},
            messages=messages,
        )
        cost += estimate_cost(reply.usage, MODEL)

        # 2. Add the model's answer to the history: role "assistant", content = reply.content (whole).
        #    Same as the first messages.append(...) in step 4 of one_call.py.
        # TODO (1 line)
        messages.append({"role": "assistant", "content": reply.content})

        # 3. If the model did NOT ask for a tool, this is the final answer.
        #    Check: reply.stop_reason != "tool_use"
        #    Then: print(f"  [{step + 1} model calls, ${cost:.4f}]") and return extract_text(reply)
        # TODO (3 lines)
        if reply.stop_reason != "tool_use":
            print(f"  [{step + 1} model calls, ${cost:.4f}]")
            return extract_text(reply)

        # 4. The model asked for one or more tools. Run each one and collect the results.
        results = []
        for block in reply.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")

                # 4a. Take our function from FUNCTIONS by the tool name (block.name)
                #     and call it with the model's data (**block.input).
                #     In one_call.py it was: get_policy(**tool_call.input)
                tool_name = FUNCTIONS[block.name]
                output = tool_name(**block.input)

                # 4b. Add a tool_result block for this request to the results list.
                #     Same dict as in step 4 of one_call.py, but with block.id and output.
                # TODO (results.append({...}))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                })

        # 5. Send all results back to the model as ONE message with role "user".
        #    Same as the second messages.append(...) in step 4 of one_call.py, but content=results.
        # TODO (1 line)
        messages.append({"role": "user", "content": results})


    return "Stopped: too many tool calls."


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python -m lesson_04.code.agent "your question"')
    client = get_client()
    print(ask(client, sys.argv[1]))


if __name__ == "__main__":
    main()
