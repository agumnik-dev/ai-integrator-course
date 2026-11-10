"""Lesson 6: four patterns for working with the model.

Routing -> chain (with a gate) -> check -> parallel.

Run from the course root (one mode per step):
    uv run python -m lesson_06.code.pipeline route              # step 1
    uv run python -m lesson_06.code.pipeline chain              # step 2
    uv run python -m lesson_06.code.pipeline check              # step 3
    uv run python -m lesson_06.code.pipeline all                # step 4: one email after another
    uv run python -m lesson_06.code.pipeline all --parallel     # step 4: all emails at once
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.llm import get_client
from lesson_06.code.llm_calls import Facts, Route, Verdict, ask_structured, ask_text, total_cost

EMAILS = json.loads((Path(__file__).parent.parent / "data" / "emails.json").read_text(encoding="utf-8"))

# ---------- Step 1. Routing ----------

ROUTER_SYSTEM = """Classify a customer email sent to an insurance company:
- new_claim: the customer reports a new incident (accident, damage, theft, flood)
- claim_status: the customer asks about a claim that is already open
- policy_question: the customer asks about their policy (coverage, dates, price)
- cancel_policy: the customer asks to cancel his policy (coverage, dates, price)
- other: anything else (advertising, spam, unrelated)"""

# Department rules: category -> what the reply must contain.
REPLY_RULES = {
    "new_claim": "Claims department. Confirm the claim was received. Ask for photos of the damage "
                 "(and a police report for theft or hit-and-run). Say an adjuster will call within 2 business days.",
    "claim_status": "Claims department. Say the claims team will check the status and answer within 1 business day.",
    "policy_question": "Policy department. Say a policy agent will answer the question within 1 business day. "
                       "Do not answer the question yourself.",
    "cancel_policy": "Your request is received, our agent will call you in two work days",
    "other": "No automatic reply. Forward to a person.",
}


def route(client, text: str) -> str:
    """The model only names the category. Ready."""
    return ask_structured(client, ROUTER_SYSTEM, text, Route).category


def pick_rules(category: str) -> str:
    return REPLY_RULES[category]


# ---------- Step 2. Chain: extract -> (gate) -> draft ----------

EXTRACT_SYSTEM = "Extract facts from a customer email sent to an insurance company."

DRAFT_SYSTEM = """You write a short reply email on behalf of "Shomer" insurance company.
Write in the customer's language, 3-5 sentences. Sign as "Shomer customer service".
Use only facts from the email. Never promise payment, claim approval or any amount of money.
Follow the department rules."""


def extract(client, text: str) -> Facts:
    """Link 1 of the chain. Ready."""
    return ask_structured(client, EXTRACT_SYSTEM, text, Facts)


def gate(facts: Facts, rules: str) -> str:
    """Plain code between the links: check the facts and add a rule if needed."""
    if facts.policy_number is None:
        print("  [gate] no policy number -> asking the customer for it")
        rules = rules + "\nAsk the customer to send their policy number."
    return rules


def draft(client, text: str, facts: Facts, rules: str, problems: list[str] | None = None) -> str:
    """Link 2 of the chain: write the reply."""
    # build the text for the model from three parts: the email, the facts (from link 1), the rules.
    facts_as_json = facts.model_dump_json(indent=2) #turns the Facts object into JSON text
    content = f"Customer email:\n{text}\n\nFacts:\n{facts_as_json}\n\nDepartment rules:\n{rules}"
    if problems:  # step 3: the checker rejected the previous reply
        content += "\n\nYour previous reply was rejected. Fix these problems:\n" + "\n".join(problems)
    return ask_text(client, DRAFT_SYSTEM, content)


def chain(client, text: str, rules: str) -> tuple[Facts, str, str]:
    facts = extract(client, text)              # link 1
    rules = gate(facts, rules)                 # code between the links
    reply = draft(client, text, facts, rules)  # link 2
    return facts, rules, reply


# ---------- Step 3. Check before sending ----------

CHECK_SYSTEM = """You review a reply email of an insurance company before it is sent.
Rules:
1. The reply is in the same language as the customer email.
2. It does not promise payment, claim approval or any amount of money.
3. It does not state facts that are not in the customer email.
4. It follows the department rules.
Return ok=true only if all rules pass. Otherwise list each broken rule in problems."""

# A deliberately bad reply to email_1: wrong language and promises money.
BAD_DRAFT = "Шалом! Ваша жалоба одобрена, завтра мы переведём вам 12 000 шекелей."


def check(client, text: str, reply: str, rules: str) -> Verdict:
    """A separate model call with a different job: not "write", but "check". Ready."""
    content = f"Customer email:\n{text}\n\nDepartment rules:\n{rules}\n\nReply to check:\n{reply}"
    return ask_structured(client, CHECK_SYSTEM, content, Verdict)


def review(client, text: str, facts: Facts, rules: str, reply: str) -> tuple[str, str]:
    """Check the reply. If the checker rejects it, write it again ONCE and check again."""
    verdict = check(client, text, reply, rules)
    print(f"  [check] ok={verdict.ok} {verdict.problems}")
    if verdict.ok:
        return reply, "ok"

    # The first line writes the reply again, now with the list of problems;
    reply = draft(client, text, facts, rules, problems=verdict.problems)

    # The second checks the new reply.
    verdict = check(client, text, reply, rules)

    print(f"  [check again] ok={verdict.ok} {verdict.problems}")
    return reply, "fixed" if verdict.ok else "failed"


# ---------- Step 4. Parallel ----------

def process_email(client, email: dict) -> dict:
    """The whole pipeline for ONE email. Ready."""
    category = route(client, email["text"])
    if category == "other":  # routing to a person: no automatic reply
        return {"id": email["id"], "category": category, "status": "to a person", "reply": ""}
    facts, rules, reply = chain(client, email["text"], pick_rules(category))
    reply, status = review(client, email["text"], facts, rules, reply)
    return {"id": email["id"], "category": category, "status": status, "reply": reply}


def run_all(client, parallel: bool) -> list[dict]:
    if not parallel:
        return [process_email(client, email) for email in EMAILS]
    
    # run process_email for all EMAILS at the same time (like Promise.all in JS).
    with ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(lambda email: process_email(client, email), EMAILS))


# ---------- Modes: one per step. Ready. ----------

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    client = get_client()

    if mode == "route":
        for email in EMAILS:
            category = route(client, email["text"])
            print(f"{email['id']}  {category:<16} -> {pick_rules(category)[:60]}...")

    elif mode == "chain":
        for email in EMAILS[:4]:  # email_5 is advertising, no reply needed
            print(f"=== {email['id']}")
            category = route(client, email["text"])
            facts, rules, reply = chain(client, email["text"], pick_rules(category))
            print(f"  facts: {facts}")
            print(reply, "\n")

    elif mode == "check":
        email = EMAILS[0]
        rules = pick_rules("new_claim")
        facts = extract(client, email["text"])
        print("=== bad reply:\n" + BAD_DRAFT)
        reply, status = review(client, email["text"], facts, rules, BAD_DRAFT)
        print(f"=== final reply [{status}]:\n{reply}")

    elif mode == "all":
        parallel = "--parallel" in sys.argv
        start = time.perf_counter()
        rows = run_all(client, parallel)
        seconds = time.perf_counter() - start
        for row in rows:
            print(f"=== {row['id']}  {row['category']}  [{row['status']}]")
            if row["reply"]:
                print(row["reply"], "\n")
        print(f"{'parallel' if parallel else 'one by one'}: {seconds:.1f} sec, ${total_cost():.4f}")

    else:
        sys.exit("Usage: uv run python -m lesson_06.code.pipeline route|chain|check|all [--parallel]")
    if mode != "all":
        print(f"cost: ${total_cost():.4f}")


if __name__ == "__main__":
    main()
