"""LangSmith evaluate() harness for the full agent pipeline: router accuracy,
answer faithfulness, abstention/escalation correctness, and ticket quality.

Unlike evaluate.py (retrieval-only: recall@k / MRR), this runs the actual
LangGraph pipeline and uploads results as LangSmith experiments.

Safety invariants (do not violate when editing):
- Never call build_graph().invoke() on a query that could route to "status"
  -- status_check_node makes a real call to jira_utils.get_tickets_by_email()
  against the live Jira tenant. Router-accuracy testing of the "status"
  label calls router_node() directly instead, bypassing the graph entirely.
- Never resume past the first "collect_ticket_info" interrupt -- escalate_node
  requires two interrupt() resumes before jira_utils.create_ticket() is ever
  called. Every target function here invokes the graph exactly once with a
  fresh thread_id and never issues a Command(resume=...) call, so a real
  Jira ticket is never created by running these evals.

Run with:  cd src && uv run python evaluate_agent.py [router|faithfulness|abstention|ticket-quality ...]
(no args runs all four)
"""

import json
import re
import sys
import uuid

from langsmith import Client
from langsmith.evaluation import evaluate

from config import get_llm
from graph import (
    CATEGORY_BY_ROUTE,
    build_graph,
    is_insufficient,
    router_node,
    _hr_context,
    _it_context,
)

client = Client()
judge_llm = get_llm()  # same model that answers -- see self-grading note below

SAMPLE_SIZE = 10  # per-corpus sample size for the "should answer" query set

# Hand-authored hard cases, found by probing the live pipeline. The sampled
# fixture queries are near-verbatim copies of corpus content, so they all
# pass trivially; these exist to show non-perfect scores in LangSmith.
#
# Blends two unrelated incident families (VPN cert disconnect, family 9 +
# Intune compliance, family 3): the model tends to fabricate a causal link
# between them and invent remediation steps found in neither family.
HARD_FAITHFULNESS_QUERIES = [
    "My VPN keeps disconnecting seconds after Okta MFA succeeds, and separately "
    "Intune shows my device as noncompliant with a missing encryption signal -- "
    "could these two issues be caused by the same underlying problem?",
]

# Rambling multi-topic query: escalation defaults stay structurally valid
# (ticket_structure 1.0) while the drafted description is too disorganized
# for a human agent to act on (ticket_description_quality ~0.5).
HARD_TICKET_QUERIES = [
    "So I have been meaning to ask about a few things -- my manager mentioned "
    "something about a new expense system, also my badge sometimes does not work "
    "on Mondays, and I think there was an email about updating something but I "
    "forget what, and also is there a stipend for home internet, and one more "
    "thing, does the pet-friendly office policy apply to the Denver location or "
    "just headquarters?",
]


def _load(name: str) -> list:
    with open(name) as f:
        return json.load(f)


def _answerable_sample() -> list[str]:
    hr = [item["query"] for item in _load("eval_hr.json")[:SAMPLE_SIZE]]
    it = [item["query"] for item in _load("eval_it.json")[:SAMPLE_SIZE]]
    return hr + it


def _ensure_dataset(name: str, description: str, examples: list[dict]) -> None:
    """Sync a LangSmith dataset with the local fixtures, keyed on
    inputs["query"]: create it if missing, otherwise add new queries, update
    ones whose expected outputs changed, and remove ones no longer in the
    fixture -- so editing eval_*.json is reflected without duplicating rows
    that haven't changed."""
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(dataset_name=name, description=description)
        client.create_examples(dataset_name=name, examples=examples)
        return

    existing_by_query = {ex.inputs["query"]: ex for ex in client.list_examples(dataset_name=name)}
    new_by_query = {e["inputs"]["query"]: e for e in examples}

    to_create = [e for q, e in new_by_query.items() if q not in existing_by_query]
    to_update = [
        {"id": existing_by_query[q].id, "inputs": e["inputs"], "outputs": e["outputs"]}
        for q, e in new_by_query.items()
        if q in existing_by_query and existing_by_query[q].outputs != e["outputs"]
    ]
    to_delete = [ex.id for q, ex in existing_by_query.items() if q not in new_by_query]

    if to_create:
        client.create_examples(dataset_name=name, examples=to_create)
    if to_update:
        client.update_examples(dataset_name=name, updates=to_update)
    if to_delete:
        client.delete_examples(example_ids=to_delete)


def _parse_judge_json(raw: str) -> dict:
    """Parse a judge LLM's JSON response, tolerating stray prose around it.
    Falls back to a hard 0.0 score so a malformed judge response fails loud
    (visible in the UI) instead of crashing the whole evaluate() run."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"score": 0.0, "reason": f"unparseable judge output: {raw[:200]}"}


def _invoke(query: str) -> dict:
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return build_graph().invoke({"query": query}, config=cfg)


# --- 1. Router accuracy -------------------------------------------------------

def target_router(inputs: dict) -> dict:
    result = router_node({"query": inputs["query"]})
    return {"route": result["route"]}


def router_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    correct = outputs.get("route") == reference_outputs.get("expected_route")
    return {"key": "router_accuracy", "score": int(correct)}


def run_router() -> None:
    examples = [
        {"inputs": {"query": e["query"]}, "outputs": {"expected_route": e["expected_route"]}}
        for e in _load("eval_router.json")
    ]
    _ensure_dataset("omnidesk-router", "Router classification accuracy (hr/it/status/both)", examples)
    evaluate(target_router, data="omnidesk-router", evaluators=[router_accuracy], experiment_prefix="router")


# --- 2. Answer faithfulness ----------------------------------------------------

FAITHFULNESS_JUDGE_PROMPT = """You are grading whether an AI support assistant's \
answer is fully grounded in the context it was given, with no fabricated \
details (no invented policy numbers, dates, thresholds, steps, tool names, \
or resolutions that don't appear in the context).

Question: {query}

Context the assistant was given:
{context}

Assistant's answer:
{answer}

Score the answer's faithfulness to the context on this scale:
- 1.0: Every claim in the answer is directly supported by the context (or \
  the answer is the exact phrase "I don't have enough information to answer \
  that." and the context genuinely does not cover the question).
- 0.5: The answer is mostly grounded but adds at least one minor unsupported \
  detail or overgeneralizes slightly beyond what the context states.
- 0.0: The answer contains a significant fabrication -- a specific fact, \
  number, step, or policy claim not present anywhere in the context -- OR \
  the answer says "I don't have enough information" despite the context \
  clearly containing the answer.

Respond with ONLY a JSON object: {{"score": <0.0|0.5|1.0>, "reason": "<one sentence>"}}"""


def target_faithfulness(inputs: dict) -> dict:
    result = _invoke(inputs["query"])
    context = "\n\n".join(
        part for part in (
            _hr_context(result["hr_hits"]) if result.get("hr_hits") else "",
            _it_context(result["it_hits"]) if result.get("it_hits") else "",
        ) if part
    )
    return {"answer": result.get("answer", ""), "route": result.get("route"), "context": context}


def faithfulness(inputs: dict, outputs: dict) -> dict:
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        query=inputs["query"], context=outputs.get("context", ""), answer=outputs.get("answer", ""),
    )
    parsed = _parse_judge_json(judge_llm.invoke(prompt).content)
    return {"key": "faithfulness", "score": parsed.get("score", 0.0), "comment": parsed.get("reason", "")}


def run_faithfulness() -> None:
    queries = _answerable_sample() + HARD_FAITHFULNESS_QUERIES
    examples = [{"inputs": {"query": q}, "outputs": {}} for q in queries]
    _ensure_dataset("omnidesk-faithfulness", "Answer groundedness in retrieved context", examples)
    evaluate(target_faithfulness, data="omnidesk-faithfulness", evaluators=[faithfulness],
             experiment_prefix="faithfulness")


# --- 3. Abstention / escalation correctness ------------------------------------

def target_abstention(inputs: dict) -> dict:
    result = _invoke(inputs["query"])
    interrupts = result.get("__interrupt__") or []
    reached_escalate = bool(interrupts) and interrupts[0].value.get("type") == "collect_ticket_info"
    answer = result.get("answer", "")
    return {
        "answer": answer,
        "route": result.get("route"),
        "abstained": is_insufficient(answer),
        "reached_escalate_interrupt": reached_escalate,
    }


def abstention_correctness(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["expected_abstain"]
    actual = outputs.get("abstained", False)
    interrupt_consistent = actual == outputs.get("reached_escalate_interrupt", False)
    correct = (actual == expected) and interrupt_consistent
    comment = None if correct else (
        f"expected_abstain={expected}, actual_abstain={actual}, "
        f"reached_escalate_interrupt={outputs.get('reached_escalate_interrupt')}"
    )
    return {"key": "abstention_correctness", "score": int(correct), "comment": comment}


def run_abstention() -> None:
    examples = [{"inputs": {"query": e["query"]}, "outputs": {"expected_abstain": True}}
                for e in _load("eval_hr_negative.json")]
    examples += [{"inputs": {"query": q}, "outputs": {"expected_abstain": False}}
                 for q in _answerable_sample()]
    _ensure_dataset("omnidesk-abstention", "Pipeline-level abstain/escalate correctness", examples)
    evaluate(target_abstention, data="omnidesk-abstention", evaluators=[abstention_correctness],
             experiment_prefix="abstention")


# --- 4. Ticket quality -----------------------------------------------------------

TICKET_QUALITY_JUDGE_PROMPT = """You are grading the quality of an auto-drafted \
support ticket description, written when an AI assistant could not answer an \
employee's question and is escalating to a human.

Original employee question: {query}

Auto-drafted ticket summary: {summary}
Auto-drafted ticket description: {description}

A human support agent should be able to read only the summary+description \
(without the original question) and understand what the employee needs. \
Score on this scale:
- 1.0: Clear, specific, and would let a human agent act without needing to \
  ask the employee anything further.
- 0.5: Understandable but vague, generic, or missing a detail a human agent \
  would likely need to ask for.
- 0.0: Garbled, missing key context, or effectively useless to a human \
  reading it cold.

Respond with ONLY a JSON object: {{"score": <0.0|0.5|1.0>, "reason": "<one sentence>"}}"""


def target_ticket_quality(inputs: dict) -> dict:
    result = _invoke(inputs["query"])
    interrupts = result.get("__interrupt__") or []
    if not interrupts or interrupts[0].value.get("type") != "collect_ticket_info":
        return {"reached_escalate": False, "defaults": {}, "route": result.get("route")}
    return {
        "reached_escalate": True,
        "defaults": interrupts[0].value["defaults"],
        "route": result.get("route"),
    }


def ticket_structure(outputs: dict, reference_outputs: dict) -> dict:
    if not outputs.get("reached_escalate"):
        return {"key": "ticket_structure", "score": 0, "comment": "did not reach escalate_node/interrupt"}
    d = outputs["defaults"]
    checks = {
        "summary_nonempty": bool(d.get("summary", "").strip()),
        "summary_len_ok": len(d.get("summary", "")) <= 120,
        "description_nonempty": bool(d.get("description", "").strip()),
        "category_matches_route": d.get("category") == CATEGORY_BY_ROUTE.get(outputs.get("route"), "IT"),
        "priority_present": d.get("priority") in ("Low", "Medium", "High", "Critical"),
    }
    score = sum(checks.values()) / len(checks)
    failed = [k for k, v in checks.items() if not v]
    return {"key": "ticket_structure", "score": score, "comment": f"failed: {failed}" if failed else None}


def ticket_description_quality(inputs: dict, outputs: dict) -> dict:
    if not outputs.get("reached_escalate"):
        return {"key": "ticket_description_quality", "score": 0, "comment": "no ticket drafted"}
    d = outputs["defaults"]
    prompt = TICKET_QUALITY_JUDGE_PROMPT.format(
        query=inputs["query"], summary=d.get("summary", ""), description=d.get("description", ""),
    )
    parsed = _parse_judge_json(judge_llm.invoke(prompt).content)
    return {"key": "ticket_description_quality", "score": parsed.get("score", 0.0), "comment": parsed.get("reason", "")}


def run_ticket_quality() -> None:
    examples = [{"inputs": {"query": e["query"]}, "outputs": {"expected_category": "HR"}}
                for e in _load("eval_hr_negative.json")]
    # HARD_TICKET_QUERIES route "both" (mixed HR/IT topics), unlike the HR-only
    # negative fixture above -- category expectation differs accordingly.
    examples += [{"inputs": {"query": q}, "outputs": {"expected_category": "HR/IT"}}
                 for q in HARD_TICKET_QUERIES]
    _ensure_dataset("omnidesk-ticket-quality", "Auto-drafted escalation ticket quality", examples)
    evaluate(target_ticket_quality, data="omnidesk-ticket-quality",
             evaluators=[ticket_structure, ticket_description_quality], experiment_prefix="ticket-quality")


# --- CLI -----------------------------------------------------------------------

RUNNERS = {
    "router": run_router,
    "faithfulness": run_faithfulness,
    "abstention": run_abstention,
    "ticket-quality": run_ticket_quality,
}


def main() -> None:
    names = sys.argv[1:] or list(RUNNERS)
    for name in names:
        print(f"\n=== running {name} eval ===")
        RUNNERS[name]()


if __name__ == "__main__":
    main()
