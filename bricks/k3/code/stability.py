"""Brick K3: check how stable the letter extraction is across repeated runs.

Run from the course root:
    uv run python -m bricks.k3.code.stability
"""

import json
from pathlib import Path

from bricks.k2.code.extract_letter import MODEL, SYSTEM, CustomerLetter
from common.llm import estimate_cost, get_client

DATA_DIR = Path("bricks/k3/data")
EXPECTED_FILE = DATA_DIR / "expected_letters.json"
FIELDS = ["policy_number", "claim_number", "topic", "language"]
RUNS = 3


def main() -> None:
    # Only the file names are used here; the expected answers are for eval_letters.py
    file_names = list(json.loads(EXPECTED_FILE.read_text(encoding="utf-8")))

    client = get_client()
    cost = 0.0
    stable_count = 0

    for file_name in file_names:
        text = (DATA_DIR / file_name).read_text(encoding="utf-8")

        # answers[field] -> the values from each run
        answers = {field: [] for field in FIELDS}
        for _ in range(RUNS):
            reply = client.messages.parse(
                model=MODEL,
                max_tokens=1000,
                system=SYSTEM,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": text}],
                output_format=CustomerLetter,
            )
            cost += estimate_cost(reply.usage, MODEL)
            letter = reply.parsed_output

            for field in FIELDS:
                # A run with no data counts as a different answer, not a skipped one
                answers[field].append(getattr(letter, field) if letter else "<no data>")

        unstable = [field for field in FIELDS if len(set(answers[field])) > 1]
        if not unstable:
            stable_count += 1
            continue

        print(f"{file_name}:")
        for field in unstable:
            print(f"  {field}: {answers[field]}")

    print(f"{stable_count} of {len(file_names)} letters stable")
    print(f"Total cost: ${cost:.5f}")


if __name__ == "__main__":
    main()
