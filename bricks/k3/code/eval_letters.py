"""check letter's content extraction against known correct answers.

Run from the course root:
    uv run python -m bricks.k3.code.eval_letters
"""

from pathlib import Path
import json

from bricks.k2.code.extract_letter import MODEL, SYSTEM, CustomerLetter
from common.llm import estimate_cost, get_client

DATA_DIR = Path("bricks/k3/data")
JSON_FILE = Path("expected_letters.json")

# The test set: file -> correct answers. Written by a person, NOT by the model.
file_path = (DATA_DIR / JSON_FILE)
EXPECTED = json.loads(file_path.read_text(encoding="utf-8"))


FIELDS = ["policy_number", "claim_number", "topic", "language"]

def main() -> None:
    client = get_client()

    # Счётчик верных ответов для каждого поля # {"policy_number": 0, "claim_number": 0, "topic": 0}
    correct = {field: 0 for field in FIELDS}  
    errors = []
    cost = 0.0

    for file_name, right in EXPECTED.items():
        text = (DATA_DIR / file_name).read_text(encoding="utf-8")
        reply = client.messages.parse(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": text}],
            output_format=CustomerLetter,
        )
        cost += estimate_cost(reply.usage, MODEL)
        model_output = reply.parsed_output

        # Модель ничего не вернула
        if model_output is None:
            errors.append(f"{file_name}: the model returned no data")
            continue

        for field in FIELDS:
            got = getattr(model_output, field)
            if got == right[field]:
                correct[field] += 1
            else:
                errors.append(f"{file_name} {field}: expected {right[field]!r}, got {got!r}")

    total = len(EXPECTED)
    for field in FIELDS:
        print(f"{field:<12} {correct[field]}/{total}")
    for error in errors:
        print(f"  ✗ {error}")
    print(f"cost: ${cost:.5f}")


if __name__ == "__main__":
    main()
