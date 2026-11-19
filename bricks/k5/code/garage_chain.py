"""Garage chain: extract -> decide -> write.

Garage invoices for a trading company:
  1. the model reads the invoice into a schema   (extract)
  2. plain code checks it against purchase orders (decide)
  3. the model writes a short email to the supplier (write)

Run from the course root:
    uv run python -m bricks.k5.code.garage_chain
"""

import sys
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

from bricks.k2.code.supplier_invoice import MODEL
from common.llm import estimate_cost, extract_text, get_client

EXTRACT_SYSTEM = """You work in the claim department of a company.
You receive invoices from garages. An invoice can be in English, Russian or Hebrew.
Extract the fields described in the schema.
If a value is not written in the invoice, return null. Never guess."""

DATA_DIR = Path("bricks/k5/data")
GARAGE_INVOICES = json.loads((DATA_DIR / "garage_invoices.json").read_text(encoding="utf-8"))
POLICIES = json.loads((DATA_DIR / "policies.json").read_text(encoding="utf-8"))

ADJUSTER_LIMIT_ILS = 10_000

WRITE_SYSTEM = """You write a short email from the claim department to a garage.
Use only the decision and the reason you are given. Use only the amount we pay that you are given.
Write in the language of the invoice, 2-3 sentences. Sign as "Claim department, Shibolet"."""

class GarageInvoice(BaseModel):
    garage: str = Field(description="Supplier company name, as written")
    invoice_number: str = Field(description="The supplier's own invoice number")
    customer_name: str = Field(description="The customer name")
    policy_number: str | None = Field(
        description="policy number: 5 or 6 digits, Formats: XXXXX or XXXX-XX"
        "Not the invoice number. Null if not written."
    )
    work_type: Literal["collision", "glass", "theft", "other"] = Field(
    description="collision: body or engine repair after an accident; glass: windshield or windows only; theft: repair after a theft or break-in; other: anything else")
    total_ils: float = Field(description="Final total, i.e. Subtotal + VAT, in ILS, as a number.")
    language: Literal["English", "Hebrew", "Russian", "Other"] = Field(description="Language the invoice is written in")


# Link 1. The model reads the invoice.
def extract(client, text: str) -> tuple[GarageInvoice | None, float]:
    reply = client.messages.parse(
        model=MODEL,
        max_tokens=1000,
        system=EXTRACT_SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": text}],
        output_format=GarageInvoice,
    )
    return reply.parsed_output, estimate_cost(reply.usage, MODEL)


# Link 2. Plain code decides. No model here: the rules must be exact.
def decide(invoice: GarageInvoice) -> tuple[str, float | None, str]:
    # """Return (decision, reason). decision: ask_policy | not_covered | adjuster | accept."""
    if invoice.policy_number is None:
        return "ask_policy", None, "The invoice has no policy number. Ask garage to provide it"
    policy = POLICIES.get(invoice.policy_number)
    if policy is None: # policy not found
        return "ask_policy", None, f"Policy {invoice.policy_number} was not found."
    if invoice.work_type not in policy["coverage"]:
        return "not_covered", None, f"The work is not covered by policy PO {invoice.policy_number}."
    if invoice.total_ils > ADJUSTER_LIMIT_ILS:
        return "adjuster", None, f"Invoice is sent to adjuster to be checked"

    payout = invoice.total_ils - policy["deductible_ils"]
    return "accept", payout, (f"Total {invoice.total_ils:,.2f} ILS minus the customer's deductible "
                          f"{policy['deductible_ils']:,} ILS under policy {invoice.policy_number}.")


# Link 3. The model writes the email. Decision and numbers come from code.
def write(client, invoice: GarageInvoice, decision: str, payout: float | None, reason: str) -> tuple[str, float]:
    amount = f"{payout:,.2f} ILS" if payout is not None else "none"
    content = (f"Invoice language: {invoice.language}\n"
               f"garage: {invoice.garage}\n"
               f"Invoice number: {invoice.invoice_number}\n"
               f"Decision: {decision}\n"
               f"Amount we pay: {amount}\n"
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
        return {"invoice": None, "decision": "error", "payout": None, "reason": "could not read the invoice",
                "email": "", "cost": cost}
    decision, payout, reason = decide(invoice)
    email, write_cost = write(client, invoice, decision, payout, reason)
    return {"invoice": invoice, "decision": decision, "payout": payout, "reason": reason,
            "email": email, "cost": cost + write_cost}


def main() -> None:
    client = get_client()
    total_cost = 0.0
    for garage_invoice in GARAGE_INVOICES:
        result = run_chain(client, garage_invoice["text"])
        print(f"=== {garage_invoice['id']}: {result['decision']} {result['payout']} — {result['reason']}")
        total_cost += result["cost"]
        print(result["email"], "\n")
    print(f"cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
