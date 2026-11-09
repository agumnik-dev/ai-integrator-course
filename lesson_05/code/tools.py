"""Tools for lesson 5. Same as lesson 4, plus a switch that breaks the claims database."""

import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
POLICIES = json.loads((DATA / "policies.json").read_text(encoding="utf-8"))
CLAIMS = json.loads((DATA / "claims.json").read_text(encoding="utf-8"))

# Step 1: set to False to simulate "the claims database is down".
CLAIMS_DB_ONLINE = True


def get_policy(policy_number: str) -> dict:
    """Return the policy by its number, or an error message if not found."""
    return POLICIES.get(policy_number, {"error": f"policy {policy_number} not found"})


def get_claims(policy_number: str) -> list:
    """Return all claims opened on this policy (empty list if none)."""
    if not CLAIMS_DB_ONLINE:
        # In real life this would be a timeout or a connection error from the database.
        raise ConnectionError("claims database is not responding")
    return [claim for claim in CLAIMS if claim["policy_number"] == policy_number]


GET_POLICY_TOOL = {
    "name": "get_policy",
    "description": "Look up an insurance policy by its number. Returns the holder, status, "
                   "expiry date, what is covered and what is not, and the deductible.",
    "input_schema": {
        "type": "object",
        "properties": {
            "policy_number": {"type": "string", "description": "Policy number, e.g. 4471-22"}
        },
        "required": ["policy_number"],
    },
}

GET_CLAIMS_TOOL = {
    "name": "get_claims",
    "description": "List all claims opened on a policy, with their status and amount.",
    "input_schema": {
        "type": "object",
        "properties": {
            "policy_number": {"type": "string", "description": "Policy number, e.g. 88213"}
        },
        "required": ["policy_number"],
    },
}
