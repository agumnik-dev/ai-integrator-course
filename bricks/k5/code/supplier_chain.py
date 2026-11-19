"""Worked example for brick K5: a chain extract -> decide -> write.

Supplier invoices for a trading company:
  1. the model reads the invoice into a schema   (extract)
  2. plain code checks it against purchase orders (decide)
  3. the model writes a short email to the supplier (write)

Run from the course root:
    uv run python -m bricks.k5.code.supplier_chain
"""

import json
from pathlib import Path

from bricks.k2.code.supplier_invoice import MODEL, SupplierInvoice
from bricks.k2.code.supplier_invoice import SYSTEM as EXTRACT_SYSTEM
from common.llm import estimate_cost, extract_text, get_client

DATA_DIR = Path("bricks/k5/data")
INVOICES = json.loads((DATA_DIR / "supplier_invoices.json").read_text(encoding="utf-8"))
ORDERS = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))

WRITE_SYSTEM = """You write a short email from the accounting department to a supplier.
Use only the decision and the reason you are given. Do not change any number.
Write in the language of the invoice, 2-3 sentences. Sign as "Accounting, Shibolet Trading"."""


# Link 1. The model reads the invoice.
def extract(client, text: str) -> tuple[SupplierInvoice | None, float]:
    reply = client.messages.parse(
        model=MODEL,
        max_tokens=1000,
        system=EXTRACT_SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": text}],
        output_format=SupplierInvoice,
    )
    return reply.parsed_output, estimate_cost(reply.usage, MODEL)


# Link 2. Plain code decides. No model here: the rules must be exact.
def decide(invoice: SupplierInvoice) -> tuple[str, str]:
    """Return (decision, reason). decision: pay | ask_po | hold."""
    if invoice.po_number is None:
        return "ask_po", "The invoice has no purchase order (PO) number."
    order = ORDERS.get(invoice.po_number)
    if order is None:
        return "hold", f"PO {invoice.po_number} was not found."
    if order["status"] != "approved":
        return "hold", f"PO {invoice.po_number} is {order['status']}."
    if abs(order["total_ils"] - invoice.total_ils) > 1:
        return "hold", (f"Invoice total {invoice.total_ils:,.2f} ILS differs "
                        f"from PO total {order['total_ils']:,.2f} ILS.")
    return "pay", f"Matches PO {invoice.po_number}: {invoice.total_ils:,.2f} ILS."


# Link 3. The model writes the email. Decision and numbers come from code.
def write(client, invoice: SupplierInvoice, decision: str, reason: str) -> tuple[str, float]:
    content = (f"Invoice language: {invoice.language}\n"
               f"Supplier: {invoice.supplier}\n"
               f"Invoice number: {invoice.invoice_number}\n"
               f"Decision: {decision}\n"
               f"Reason: {reason}")
    reply = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=WRITE_SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
    )
    return extract_text(reply), estimate_cost(reply.usage, MODEL)


# The whole chain for one invoice.
def run_chain(client, text: str) -> dict:
    invoice, cost = extract(client, text)
    if invoice is None:  # link 1 failed: stop, do not guess
        return {"invoice": None, "decision": "error", "reason": "could not read the invoice",
                "email": "", "cost": cost}
    decision, reason = decide(invoice)
    email, write_cost = write(client, invoice, decision, reason)
    return {"invoice": invoice, "decision": decision, "reason": reason,
            "email": email, "cost": cost + write_cost}


def main() -> None:
    client = get_client()
    total_cost = 0.0
    for item in INVOICES:
        result = run_chain(client, item["text"])
        total_cost += result["cost"]
        print(f"=== {item['id']}: {result['decision']} — {result['reason']}")
        print(result["email"], "\n")
    print(f"cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
