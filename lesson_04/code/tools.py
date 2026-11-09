"""Tools the model can use: look up policies and claims in our "database"."""

import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
POLICIES = json.loads((DATA / "policies.json").read_text(encoding="utf-8"))
CLAIMS = json.loads((DATA / "claims.json").read_text(encoding="utf-8"))


# TODO (step 1): write get_policy(policy_number) -> dict
# It returns the policy with this number from POLICIES.
# POLICIES is a dict: the key is the policy number, the value is the policy.

def get_policy(policy_number) -> dict:
    """Return the policy by its number, or an error message if not found."""
    return POLICIES.get(policy_number, {"Error": f"policy {policy_number} not found"})


def get_claims(policy_number: str) -> list:
    """Return all claims opened on this policy (empty list if none)."""
    return [claim for claim in CLAIMS if claim["policy_number"] == policy_number]


# TODO (step 2): write GET_POLICY_TOOL by analogy with GET_CLAIMS_TOOL below.
GET_POLICY_TOOL = {
    "name": "get_policy",
    "description": "Find a policy by its number, return holder, status, valid_until, coverage, not_covered, deductible_ils",
    "input_schema": {
        "type": "object",
        "properties": {
            "policy_number": {"type": "string", "description": "Policy number, e.g. 88213"}
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
