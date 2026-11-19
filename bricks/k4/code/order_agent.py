"""Worked example for brick K4: an agent with one tool.

An accountant asks about a purchase order (PO). The model asks our code
to look the order up, then answers from the data.

Run from the course root:
    uv run python -m bricks.k4.code.order_agent "Who approved PO 4500012874?"
"""

import json
import sys
from pathlib import Path

from common.llm import estimate_cost, extract_text, get_client

MODEL = "claude-haiku-5-5"
MAX_STEPS = 5
DATA_DIR = Path("bricks/k4/data")
ORDERS = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))

SYSTEM = """You help the accounting department of a trading company.
Use the get_order tool to look up purchase orders. Answer only from what the tool returns.
If the tool returns an error, say that the order was not found. Never guess.
Answer in 1-2 sentences, in the language of the question."""

# 1. The tool for the model: its name, what it does, what input it needs.
GET_ORDER_TOOL = {
    "name": "get_order",
    "description": "Look up a purchase order (PO) by its number. "
                   "Returns the supplier, status, who approved it and the total in ILS.",
    "input_schema": {
        "type": "object",
        "properties": {
            "po_number": {"type": "string", "description": "PO number, 10 digits, e.g. 4500012874"},
        },
        "required": ["po_number"],
    },
}


# 2. Our function: the code that really does the work.
def get_order(po_number: str) -> dict | None:
    """Return the order, or None if there is no such PO."""
    return ORDERS.get(po_number)


# 3. The agent loop.
def run_agent(client, question: str) -> tuple[str, list[str], float]:
    """Return (answer, PO numbers the model asked for, cost)."""
    messages = [{"role": "user", "content": question}]
    asked = []
    cost = 0.0

    for step in range(MAX_STEPS):
        reply = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM,
            tools=[GET_ORDER_TOOL],
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
            po_number = block.input["po_number"]
            asked.append(po_number)
            order = get_order(po_number)
            if order is None:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Order {po_number} not found",
                    "is_error": True,
                })
            else:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(order, ensure_ascii=False),
                })
        messages.append({"role": "user", "content": results})

    return "Stopped: step limit reached.", asked, cost


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python -m bricks.k4.code.order_agent "your question"')
    client = get_client()
    answer, asked, cost = run_agent(client, sys.argv[1])
    print(answer)
    print(f"tool calls: {asked}")
    print(f"cost: ${cost:.5f}")


if __name__ == "__main__":
    main()
