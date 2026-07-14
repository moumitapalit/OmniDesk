"""Ingest ameau01/synthetic-it-support-tickets into Qdrant.

Design decisions (see README):
- Embed ONLY title + submitted description -- that's what a new incoming
  ticket looks like, so we match like with like. root_cause / resolution
  live in the payload and are used for generation, never for matching.
- The 745 records derive from ~15 diagnostic instruction sets (heavy
  near-duplication). We cluster dense embeddings into N_IT_FAMILIES
  KMeans clusters and store `family_id` so retrieval can dedupe/diversify
  by family and eval can score family-match.
- Eval: hold out a few tickets per family; query = their description,
  gold = their family_id.

NOTE on field names: the dataset is nested. `extract_record` below tries
the expected paths and falls back gracefully; run with --inspect first to
print one raw record and adjust if the schema differs.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from sklearn.cluster import KMeans
# Load environment variables (API keys, model names)
from dotenv import load_dotenv
load_dotenv()
from config import IT_COLLECTION, N_IT_FAMILIES, get_qdrant
from qdrant_utils import embed_texts, recreate_hybrid_collection, upsert_hybrid

EVAL_HOLDOUT_PER_FAMILY = 3
EVAL_FILE = "eval_it.json"


def _get(d: dict, *paths, default=""):
    """Return the first present value among dotted paths like 'ticket.title'."""
    for path in paths:
        cur = d
        ok = True
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur and cur[key] is not None:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur:
            return cur
    return default


def extract_record(row: dict) -> dict | None:
    title = _get(row, "ticket.title", "title", "subject")
    desc = _get(row, "ticket.submitted_description", "ticket.description",
                "submitted_description", "description")
    root_cause = _get(row, "root_cause", "diagnosis.root_cause")
    resolution = _get(row, "resolution", default={})

    if isinstance(resolution, dict):
        steps = resolution.get("steps", [])
        resolution_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) \
            if steps else json.dumps(resolution)
    elif isinstance(resolution, list):
        resolution_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(resolution))
    else:
        resolution_text = str(resolution)

    if not (title or desc):
        return None
    return {
        "title": str(title).strip(),
        "description": str(desc).strip(),
        "root_cause": str(root_cause).strip(),
        "resolution": resolution_text.strip(),
        "source": "it_ticket",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                        help="print one raw record and exit (schema check)")
    args = parser.parse_args()

    ds = load_dataset("ameau01/synthetic-it-support-tickets", split="train")

    if args.inspect:
        print(json.dumps(ds[0], indent=2, default=str)[:4000])
        return

    records = [r for r in (extract_record(row) for row in ds) if r]
    print(f"Extracted {len(records)} tickets")

    # What we embed = what an incoming ticket looks like.
    texts = [f"{r['title']}\n{r['description']}" for r in records]

    # --- Derive issue families via KMeans over dense embeddings ------------
    dense, sparse = embed_texts(texts)
    X = np.array([v for v in dense])
    km = KMeans(n_clusters=N_IT_FAMILIES, random_state=42, n_init=10)
    families = km.fit_predict(X)

    for i, r in enumerate(records):
        r["id"] = i
        r["family_id"] = int(families[i])

    print("Family sizes:", dict(sorted(Counter(families).items())))

    # --- Upsert -------------------------------------------------------------
    client = get_qdrant()
    recreate_hybrid_collection(client, IT_COLLECTION)
    upsert_hybrid(client, IT_COLLECTION, texts, records)
    print(f"Upserted {len(records)} points into '{IT_COLLECTION}'")

    # --- Eval set: N held-out tickets per family ----------------------------
    random.seed(42)
    by_family = defaultdict(list)
    for r in records:
        by_family[r["family_id"]].append(r)

    eval_set = []
    for fam, members in by_family.items():
        for r in random.sample(members, k=min(EVAL_HOLDOUT_PER_FAMILY, len(members))):
            eval_set.append({
                "query": r["description"],
                "gold_family": fam,
                "gold_root_cause": r["root_cause"],
            })
    with open(EVAL_FILE, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"Wrote {len(eval_set)} eval queries to {EVAL_FILE}")


if __name__ == "__main__":
    main()
