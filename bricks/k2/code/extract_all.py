"""Brick K2: extract CustomerLetter fields from every letter and save them to CSV

Run from the course root:
    uv run python -m bricks.k2.code.extract_all
"""

import csv
import sys
from pathlib import Path

from bricks.k2.code.extract_letter import MODEL, SYSTEM, CustomerLetter
from common.llm import estimate_cost, get_client

DATA_DIR = Path("bricks/k2/data")
CSV_PATH = Path("bricks/k2/letters.csv")
COLUMNS = ["file", "customer_name", "policy_number", "claim_number", "topic", "language", "request"]


def main() -> None:
    # Sorted so the row order is stable between runs
    letter_files = sorted(DATA_DIR.glob("letter*.txt"))
    if not letter_files:
        sys.exit(f"No letter*.txt files found in {DATA_DIR}")

    client = get_client()
    total_cost = 0.0
    rows = []

    for letter_file in letter_files:
        text = letter_file.read_text(encoding="utf-8")

        reply = client.messages.parse(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": text}],
            output_format=CustomerLetter,
        )
        total_cost += estimate_cost(reply.usage, MODEL)

        letter = reply.parsed_output
        if letter is None:
            # Do not skip silently: say which file failed and why
            print(f"{letter_file.name}: could not parse (stop_reason={reply.stop_reason})", file=sys.stderr)
            continue

        rows.append({"file": letter_file.name, **letter.model_dump()})
        print(f"{letter_file.name}: {letter.topic}, {letter.language}")

    # utf-8-sig adds a BOM so Excel shows Hebrew and Russian correctly
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} of {len(letter_files)} letters to {CSV_PATH}")
    print(f"Total cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
