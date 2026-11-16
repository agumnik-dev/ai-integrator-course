"""Worked example for brick K2: read a supplier invoice into a Pydantic model.

Run from the course root:
    uv run python -m bricks.k2.code.supplier_invoice bricks/k2/data/invoice_delta.txt
"""

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import estimate_cost, get_client

MODEL = "claude-haiku-5-5"

SYSTEM = """You work in the accounting department of a company.
You receive invoices from suppliers. An invoice can be in English or Hebrew.
Extract the fields described in the schema.
If a value is not written in the invoice, return null. Never guess."""


class SupplierInvoice(BaseModel):
    supplier: str = Field(description="Supplier company name, as written")
    invoice_number: str = Field(description="The supplier's own invoice number")
    po_number: str | None = Field(
        description="Our SAP purchase order number: 10 digits, starts with 45. "
        "Not the invoice number. Null if not written."
    )
    total_ils: float = Field(description="Final total INCLUDING VAT, in ILS, as a number. Not the subtotal.")
    language: Literal["English", "Hebrew", "Other"] = Field(description="Language the invoice is written in")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: uv run python -m bricks.k2.code.supplier_invoice path/to/invoice.txt")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")

    client = get_client()
    reply = client.messages.parse(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": text}],
        output_format=SupplierInvoice,
    )
    invoice = reply.parsed_output  # an object of SupplierInvoice, not text

    print(invoice)

    # Strict rules are code, not the model.
    if invoice.po_number is None:
        print("-> no PO number: ask the supplier before payment")
    else:
        print(f"-> check PO {invoice.po_number} in SAP: {invoice.total_ils:,.2f} ILS")

    print(f"[${estimate_cost(reply.usage):.5f}]")


if __name__ == "__main__":
    main()
