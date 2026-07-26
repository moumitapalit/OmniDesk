"""LangGraph pipeline: route -> retrieve (HR | IT | both) -> answer.

    query -> router --hr---> hr_retrieve   -> hr_answer   -> END
                     --it---> it_retrieve   -> it_answer   -> END
                     --both-> both_retrieve -> both_answer -> END

Router is an LLM classification node with a low-confidence fallback to
"both". Answer nodes use different strategies:
- HR: direct grounded answer from policy Q/A pairs.
- IT: synthesized diagnosis from root causes/resolutions of the top
  distinct issue families (never naive top-k -- see retrieval.py).
"""

from typing import Literal, TypedDict


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

import jira_utils
from config import HR_COLLECTION, get_llm
from retrieval import search, search_it_deduped

llm = get_llm()

# Exact phrase answer nodes must use verbatim when the retrieved context
# doesn't cover the question. The UI matches on this string (see app.py) to
# decide when to offer filing a Jira ticket, so it must stay word-for-word
# consistent across all three answer prompts below.
INSUFFICIENT_INFO_PHRASE = "I don't have enough information to answer that."

CATEGORY_BY_ROUTE = {"hr": "HR", "it": "IT", "both": "HR/IT"}


def is_insufficient(answer: str) -> bool:
    return INSUFFICIENT_INFO_PHRASE.lower() in answer.lower()


class RAGState(TypedDict, total=False):
    query: str
    route: str                 # "hr" | "it" | "both" | "status"
    hr_hits: list
    it_hits: list
    answer: str
    email: str                  # known employee email, carried in from the UI session if any
    ticket_result: dict        # set by escalate_node once resolved (created/declined/error)


# --- Router -----------------------------------------------------------------

ROUTER_PROMPT = """Classify this employee query into exactly one word.

hr     -> HR policy questions (leave, comp-off, overtime, conduct, ethics,
          anti-bribery, whistleblower, policy administration)
it     -> IT problems (login, passwords, VPN, email, devices, software errors,
          access issues, account lockouts)
status -> asking about the status or updates of a support ticket they already
          raised (e.g. "what's the status of my ticket", "did anyone look at
          my VPN issue", "any update on ticket OMNI-12")
both   -> genuinely spans HR and IT, or you are unsure

Query: {query}

Answer with only: hr, it, status, or both."""


def router_node(state: RAGState) -> RAGState:
    out = llm.invoke(ROUTER_PROMPT.format(query=state["query"])).content.strip().lower()
    route = out if out in ("hr", "it", "both", "status") else "both"  # fallback on garbage
    return {"route": route}


def route_decision(state: RAGState) -> Literal["hr", "it", "both", "status"]:
    return state["route"]


# --- Retrieval nodes ----------------------------------------------------------

def hr_retrieve(state: RAGState) -> RAGState:
    return {"hr_hits": search(HR_COLLECTION, state["query"], mode="hybrid", top_k=5)}


def it_retrieve(state: RAGState) -> RAGState:
    return {"it_hits": search_it_deduped(state["query"], mode="hybrid")}


def both_retrieve(state: RAGState) -> RAGState:
    return {**hr_retrieve(state), **it_retrieve(state)}


# --- Answer nodes -------------------------------------------------------------

HR_ANSWER_PROMPT = """You are an HR assistant for Kreeda Labs. Answer the
employee's question using ONLY the policy excerpts below. Do not guess and
do not use outside knowledge. If the excerpts don't cover the question,
respond with EXACTLY this sentence and nothing else: "{insufficient}"

Policy excerpts:
{{context}}

Question: {{query}}

Answer concisely.""".format(insufficient=INSUFFICIENT_INFO_PHRASE)

IT_ANSWER_PROMPT = """You are an IT support assistant. Based on similar past
incidents below, give: (1) the most likely root cause, (2) recommended
resolution steps. If multiple root causes are plausible, present the top
candidates and how to distinguish them. Use ONLY the incidents below -- do
not guess and do not use outside knowledge. If none of the incidents are
relevant to the new ticket, respond with EXACTLY this sentence and nothing
else: "{insufficient}"

Similar past incidents:
{{context}}

New ticket: {{query}}

Diagnosis and resolution:""".format(insufficient=INSUFFICIENT_INFO_PHRASE)


def _hr_context(hits) -> str:
    return "\n\n".join(
        f"[{h.payload['topic']}] Q: {h.payload['question']}\nA: {h.payload['answer']}"
        for h in hits
    )


def _it_context(hits) -> str:
    return "\n\n".join(
        f"[family {h.payload['family_id']}] {h.payload['title']}\n"
        f"Root cause: {h.payload['root_cause']}\n"
        f"Resolution:\n{h.payload['resolution']}"
        for h in hits
    )


def hr_answer(state: RAGState) -> RAGState:
    prompt = HR_ANSWER_PROMPT.format(context=_hr_context(state["hr_hits"]),
                                     query=state["query"])
    return {"answer": llm.invoke(prompt).content}


def it_answer(state: RAGState) -> RAGState:
    prompt = IT_ANSWER_PROMPT.format(context=_it_context(state["it_hits"]),
                                     query=state["query"])
    return {"answer": llm.invoke(prompt).content}


def both_answer(state: RAGState) -> RAGState:
    context = (
        "HR POLICIES:\n" + _hr_context(state.get("hr_hits", []))
        + "\n\nIT INCIDENTS:\n" + _it_context(state.get("it_hits", []))
    )
    prompt = (
        "Answer the employee's question using only the context below. "
        "It may involve HR policy, an IT issue, or both. Do not guess and "
        "do not use outside knowledge. If the context doesn't cover the "
        f'question, respond with EXACTLY this sentence and nothing else: '
        f'"{INSUFFICIENT_INFO_PHRASE}"\n\n'
        f"{context}\n\nQuestion: {state['query']}\n\nAnswer:"
    )
    return {"answer": llm.invoke(prompt).content}


# --- Escalation (Jira ticket) ------------------------------------------------
#
# Runs only when an answer node emitted INSUFFICIENT_INFO_PHRASE. Two
# interrupts give the human-in-the-loop checkpoints required before any
# ticket is filed:
#   1. collect_ticket_info -- the "offer": UI shows a prefilled form and the
#      employee can fill it in or resume with {"decline": True} to opt out.
#   2. confirm_ticket -- a mandatory recap the employee must explicitly
#      accept before the Jira API is called; resuming with
#      {"confirm": False} cancels without creating anything.
def escalate_node(state: RAGState) -> RAGState:
    defaults = {
        "email": state.get("email", ""),
        "summary": state["query"][:120],
        "description": (
            f"Employee question: {state['query']}\n\n"
            "The agent could not find a matching answer in the knowledge base."
        ),
        "category": CATEGORY_BY_ROUTE.get(state.get("route"), "IT"),
        "priority": "Medium",
    }
    ticket_info = interrupt({"type": "collect_ticket_info", "defaults": defaults})
    if ticket_info.get("decline"):
        return {"ticket_result": {"ok": False, "error": "Employee declined to file a ticket."}}

    decision = interrupt({"type": "confirm_ticket", "ticket": ticket_info})
    if not decision.get("confirm"):
        return {"ticket_result": {"ok": False, "error": "Cancelled by employee."}, "email": ticket_info["email"]}

    outcome = jira_utils.create_ticket(
        email=ticket_info["email"],
        summary=ticket_info["summary"],
        description=ticket_info["description"],
        category=ticket_info["category"],
        priority=ticket_info["priority"],
    )
    return {"ticket_result": outcome, "email": ticket_info["email"]}


def escalate_decision(state: RAGState) -> Literal["escalate", "end"]:
    return "escalate" if is_insufficient(state["answer"]) else "end"


# --- Ticket status check -------------------------------------------------------
#
# Read-only flow, so it needs no confirmation step -- just the two lookups
# the spec calls for: ask for email if not already known, then let the
# employee pick a ticket if they have more than one.
def status_check_node(state: RAGState) -> RAGState:
    email = state.get("email") or interrupt({"type": "collect_status_email"})

    lookup = jira_utils.get_tickets_by_email(email)
    if not lookup["ok"]:
        return {"answer": f"I couldn't look up your tickets: {lookup['error']}", "email": email}

    tickets = lookup["tickets"]
    if not tickets:
        return {"answer": "I didn't find any tickets raised under that email.", "email": email}

    if len(tickets) == 1:
        chosen_key = tickets[0]["key"]
    else:
        chosen_key = interrupt({"type": "choose_ticket", "tickets": tickets})

    chosen = next(t for t in tickets if t["key"] == chosen_key)
    comments = jira_utils.get_ticket_comments(chosen_key)

    lines = [f"**{chosen['key']}** — {chosen['summary']} — Status: {chosen['status']}"]
    if comments["ok"] and comments["comments"]:
        lines.append("\nUpdates:")
        lines += [f"- {c['created']} — {c['author']}: {c['text']}" for c in comments["comments"]]
    elif comments["ok"]:
        lines.append("\nNo updates yet from the support team.")
    return {"answer": "\n".join(lines), "email": email}


# --- Graph ---------------------------------------------------------------------

def build_graph(checkpointer=None):
    g = StateGraph(RAGState)
    g.add_node("router", router_node)
    g.add_node("hr_retrieve", hr_retrieve)
    g.add_node("it_retrieve", it_retrieve)
    g.add_node("both_retrieve", both_retrieve)
    g.add_node("hr_answer", hr_answer)
    g.add_node("it_answer", it_answer)
    g.add_node("both_answer", both_answer)
    g.add_node("escalate", escalate_node)
    g.add_node("status_check", status_check_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_decision, {
        "hr": "hr_retrieve",
        "it": "it_retrieve",
        "both": "both_retrieve",
        "status": "status_check",
    })
    g.add_edge("hr_retrieve", "hr_answer")
    g.add_edge("it_retrieve", "it_answer")
    g.add_edge("both_retrieve", "both_answer")
    for answer_node in ("hr_answer", "it_answer", "both_answer"):
        g.add_conditional_edges(answer_node, escalate_decision, {
            "escalate": "escalate",
            "end": END,
        })
    g.add_edge("escalate", END)
    g.add_edge("status_check", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def resume(value) -> Command:
    """Wrap a value to resume a paused graph, so callers (e.g. the UI) don't
    need to import langgraph themselves."""
    return Command(resume=value)


if __name__ == "__main__":
    import uuid

    graph = build_graph()
    for q in [
        "Can I carry forward my comp off to next quarter?",
        "My AD account keeps locking after I reset my password",
        "Am I allowed to accept a gift from a vendor?",
    ]:
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = graph.invoke({"query": q}, config=cfg)
        print(f"\n=== {q}\n[route: {result['route']}]\n{result['answer']}")
