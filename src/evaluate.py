"""Eval harness: dense vs sparse vs hybrid on both corpora.

HR  metric: recall@k + MRR -- did the held-out question retrieve its own
            Q/A point? (gold_id from eval_hr.json)
IT  metric: family-match@k + MRR -- did the query retrieve tickets from
            its own issue family? (gold_family from eval_it.json)
            Individual-ticket recall would be misleading here because of
            the ~50 near-duplicates per family.

Run after both ingestion scripts. Prints a comparison table.
"""

import json

from config import HR_COLLECTION, IT_COLLECTION
from retrieval import search

MODES = ["dense", "sparse", "hybrid"]
TOP_K = 5


def eval_hr(mode: str) -> tuple[float, float]:
    with open("eval_hr.json") as f:
        eval_set = json.load(f)
    hits_at_k, rr_sum = 0, 0.0
    for item in eval_set:
        results = search(HR_COLLECTION, item["query"], mode=mode, top_k=TOP_K)
        ids = [r.id for r in results]
        if item["gold_id"] in ids:
            hits_at_k += 1
            rr_sum += 1.0 / (ids.index(item["gold_id"]) + 1)
    n = len(eval_set)
    return hits_at_k / n, rr_sum / n


def eval_it(mode: str) -> tuple[float, float]:
    with open("eval_it.json") as f:
        eval_set = json.load(f)
    hits_at_k, rr_sum = 0, 0.0
    for item in eval_set:
        results = search(IT_COLLECTION, item["query"], mode=mode, top_k=TOP_K)
        fams = [r.payload.get("family_id") for r in results]
        if item["gold_family"] in fams:
            hits_at_k += 1
            rr_sum += 1.0 / (fams.index(item["gold_family"]) + 1)
    n = len(eval_set)
    return hits_at_k / n, rr_sum / n


def main() -> None:
    print(f"{'corpus':<12}{'mode':<10}{'recall@'+str(TOP_K):<12}{'MRR':<8}")
    print("-" * 42)
    for mode in MODES:
        r, mrr = eval_hr(mode)
        print(f"{'hr':<12}{mode:<10}{r:<12.3f}{mrr:<8.3f}")
    for mode in MODES:
        r, mrr = eval_it(mode)
        print(f"{'it':<12}{mode:<10}{r:<12.3f}{mrr:<8.3f}")


if __name__ == "__main__":
    main()
