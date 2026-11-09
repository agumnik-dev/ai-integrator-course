import json
import sys

from common.llm import estimate_cost, extract_text, get_client
from lesson_04.code.tools import GET_CLAIMS_TOOL, GET_POLICY_TOOL, get_claims, get_policy

MODEL = "claude-haiku-5-5"
MAX_STEPS = 5   # safety limit: never more than 5 rounds of tool calls per question

SYSTEM = """You are a support assistant of "Shomer" insurance company.
Use the tools to look up policy data. Answer only from what the tools return.
Answer in the customer's language, in 2-3 sentences."""

QUESTION = "Does my policy 4471-22 cover windshield damage? And until when is it valid?"


# TOOLS = [GET_POLICY_TOOL, GET_CLAIMS_TOOL]
# FUNCTIONS = {"get_policy": get_policy, "get_claims": get_claims}   # tool name -> our function



def main() -> None:
    client = get_client()
    messages = [{"role": "user", "content": QUESTION}]

    # 1. First request: the model sees the question AND the tool description.
    reply = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        tools=[GET_POLICY_TOOL],
        output_config={"effort": "low"},
        messages=messages,
    )
    print("stop_reason:", reply.stop_reason)
    for block in reply.content:
        print("block:", block.type)

    # 2. Find the tool request in the answer.
    for block in reply.content:
        if block.type == "tool_use":
            tool_call = block
    print("model asks:", tool_call.name, tool_call.input)

    # 3. WE run the function. The model never runs code itself.
    result = get_policy(**tool_call.input)

    # # 4. Send back: the model's request (whole) + our result with the same id.
    messages.append({"role": "assistant", "content": reply.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False),
        }],
    })

    # # 5. Second request: now the model has the data and writes the answer.
    reply = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        tools=[GET_POLICY_TOOL],
        output_config={"effort": "low"},
        messages=messages,
    )
    print("stop_reason:", reply.stop_reason)
    print("answer:", extract_text(reply))


if __name__ == "__main__":
    main()