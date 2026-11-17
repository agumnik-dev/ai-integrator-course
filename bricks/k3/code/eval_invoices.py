"""Worked example for brick K3: check invoice extraction against known correct answers.

Run from the course root:
    uv run python -m bricks.k3.code.eval_invoices
"""

from pathlib import Path

from bricks.k2.code.supplier_invoice import MODEL, SYSTEM, SupplierInvoice
from common.llm import estimate_cost, get_client

DATA_DIR = Path("bricks/k2/data")

# The test set: file -> correct answers. Written by a person, NOT by the model.
EXPECTED = {
    "invoice_delta.txt": {"po_number": "4500012874", "total_ils": 9369.2, "language": "English"},
    "invoice_tavor.txt": {"po_number": None, "total_ils": 3776.0, "language": "Hebrew"},
}
FIELDS = ["po_number", "total_ils", "language"]


def main() -> None:
    client = get_client()

    # Счётчик верных ответов для каждого поля # {"po_number": 0, "total_ils": 0, "language": 0}
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
            output_format=SupplierInvoice,
        )
        cost += estimate_cost(reply.usage, MODEL)
        invoice = reply.parsed_output

        # Модель ничего не вернула
        if invoice is None:
            errors.append(f"{file_name}: the model returned no data")
            continue

        for field in FIELDS:
            got = getattr(invoice, field)  # like invoice[field] in JS
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
