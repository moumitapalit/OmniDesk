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


from langgraph.graph import END, StateGraph

from config import HR_COLLECTION, get_llm
from retrieval import search, search_it_deduped

llm = get_llm()  

class RAGState(TypedDict, total=False):
    query: str
    route: str                 # "hr" | "it" | "both"
    hr_hits: list
    it_hits: list
    answer: str


# --- Router -----------------------------------------------------------------

ROUTER_PROMPT = """Classify this employee query into exactly one word.

hr   -> HR policy questions (leave, comp-off, overtime, conduct, ethics,
        anti-bribery, whistleblower, policy administration)
it   -> IT problems (login, passwords, VPN, email, devices, software errors,
        access issues, account lockouts)
both -> genuinely spans both, or you are unsure

Query: {query}

Answer with only: hr, it, or both."""


def router_node(state: RAGState) -> RAGState:
    out = llm.invoke(ROUTER_PROMPT.format(query=state["query"])).content.strip().lower()
    route = out if out in ("hr", "it", "both") else "both"  # fallback on garbage
    return {"route": route}


def route_decision(state: RAGState) -> Literal["hr", "it", "both"]:
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
employee's question using ONLY the policy excerpts below. If they don't
cover it, say you don't know and suggest contacting HR.

Policy excerpts:
{context}

Question: {query}

Answer concisely."""

IT_ANSWER_PROMPT = """You are an IT support assistant. Based on similar past
incidents below, give: (1) the most likely root cause, (2) recommended
resolution steps. If multiple root causes are plausible, present the top
candidates and how to distinguish them. Use ONLY the incidents below.

Similar past incidents:
{context}

New ticket: {query}

Diagnosis and resolution:"""


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
        "It may involve HR policy, an IT issue, or both.\n\n"
        f"{context}\n\nQuestion: {state['query']}\n\nAnswer:"
    )
    return {"answer": llm.invoke(prompt).content}


# --- Graph ---------------------------------------------------------------------

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("router", router_node)
    g.add_node("hr_retrieve", hr_retrieve)
    g.add_node("it_retrieve", it_retrieve)
    g.add_node("both_retrieve", both_retrieve)
    g.add_node("hr_answer", hr_answer)
    g.add_node("it_answer", it_answer)
    g.add_node("both_answer", both_answer)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_decision, {
        "hr": "hr_retrieve",
        "it": "it_retrieve",
        "both": "both_retrieve",
    })
    g.add_edge("hr_retrieve", "hr_answer")
    g.add_edge("it_retrieve", "it_answer")
    g.add_edge("both_retrieve", "both_answer")
    g.add_edge("hr_answer", END)
    g.add_edge("it_answer", END)
    g.add_edge("both_answer", END)
    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    for q in [
        "Can I carry forward my comp off to next quarter?",
        "My AD account keeps locking after I reset my password",
        "Am I allowed to accept a gift from a vendor?",
    ]:
        result = graph.invoke({"query": q})
        print(f"\n=== {q}\n[route: {result['route']}]\n{result['answer']}")
