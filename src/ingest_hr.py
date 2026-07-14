"""Ingest strova-ai/hr-policies-qa-dataset into Qdrant.

Design decisions (see README):
- One row = one point. No chunking: answers are 1-3 sentences.
- Embed "question + answer" (Q carries the vocabulary employees actually use).
- Derive a coarse `topic` label with keyword rules for filtered retrieval
  and per-topic eval breakdowns.
- Hold out ~50 rows as an eval set (they are still ingested; eval measures
  whether the *right* point is retrieved for a held-out question).
"""

import json
import random

from datasets import load_dataset

from config import HR_COLLECTION, get_qdrant
from qdrant_utils import recreate_hybrid_collection, upsert_hybrid
# Load environment variables (API keys, model names)
from dotenv import load_dotenv
load_dotenv()

EVAL_HOLDOUT = 50
EVAL_FILE = "eval_hr.json"

TOPIC_RULES = {
    "comp_off": ["comp off", "comp-off", "compensatory"],
    "overtime": ["overtime", "extra hours"],
    "leave": ["leave", "vacation", "pto", "holiday"],
    "anti_bribery": ["bribe", "bribery", "gift", "kickback", "corruption"],
    "whistleblower": ["whistleblow", "retaliat", "report a violation"],
    "policy_admin": ["policy review", "policy update", "revised", "approval of the policy"],
    "ethics_conduct": ["ethic", "conduct", "harass", "conflict of interest"],
}


def derive_topic(question: str, answer: str) -> str:
    text = f"{question} {answer}".lower()
    for topic, keywords in TOPIC_RULES.items():
        if any(k in text for k in keywords):
            return topic
    return "other"


def flatten(row: dict) -> dict | None:
    msgs = {m["role"]: m["content"] for m in row["messages"]}
    q, a = msgs.get("user"), msgs.get("assistant")
    if not q or not a:
        return None
    return {"question": q.strip(), "answer": a.strip()}


def main() -> None:
    ds = load_dataset("strova-ai/hr-policies-qa-dataset", split="train")
    pairs = [p for p in (flatten(r) for r in ds) if p]
    print(f"Loaded {len(pairs)} Q/A pairs")

    for i, p in enumerate(pairs):
        p["id"] = i
        p["topic"] = derive_topic(p["question"], p["answer"])
        p["source"] = "hr_policy"

    # Embed Q + A together; return the answer from payload at query time.
    texts = [f"{p['question']}\n{p['answer']}" for p in pairs]

    client = get_qdrant()
    recreate_hybrid_collection(client, HR_COLLECTION)
    upsert_hybrid(client, HR_COLLECTION, texts, pairs)
    print(f"Upserted {len(pairs)} points into '{HR_COLLECTION}'")

    # Eval set: held-out questions whose gold doc is their own point id.
    random.seed(42)
    holdout = random.sample(pairs, k=min(EVAL_HOLDOUT, len(pairs)))
    eval_set = [{"query": p["question"], "gold_id": p["id"], "topic": p["topic"]} for p in holdout]
    with open(EVAL_FILE, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"Wrote {len(eval_set)} eval queries to {EVAL_FILE}")

    from collections import Counter

    print("Topic distribution:", Counter(p["topic"] for p in pairs))


if __name__ == "__main__":
    main()
