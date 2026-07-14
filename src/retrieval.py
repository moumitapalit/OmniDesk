"""Retrieval layer: dense / sparse / hybrid (RRF) over a hybrid collection.

`mode` exists so the eval harness can compare all three retrieval modes
with identical code paths.
"""

from qdrant_client import models

from config import QUERY_PREFIX, dense_model, get_qdrant, sparse_model

PREFETCH_LIMIT = 20


def _dense_query(query: str) -> list[float]:
    # bge models want this prefix on queries (not on passages).
    return next(dense_model().embed([QUERY_PREFIX + query])).tolist()


def _sparse_query(query: str) -> models.SparseVector:
    emb = next(sparse_model().embed([query]))
    return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def search(
    collection: str,
    query: str,
    mode: str = "hybrid",  # "dense" | "sparse" | "hybrid"
    top_k: int = 5,
    query_filter: models.Filter | None = None,
):
    client = get_qdrant()

    if mode == "dense":
        res = client.query_points(
            collection_name=collection,
            query=_dense_query(query),
            using="dense",
            limit=top_k,
            query_filter=query_filter,
        )
    elif mode == "sparse":
        res = client.query_points(
            collection_name=collection,
            query=_sparse_query(query),
            using="sparse",
            limit=top_k,
            query_filter=query_filter,
        )
    elif mode == "hybrid":
        res = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(query=_dense_query(query), using="dense",
                                limit=PREFETCH_LIMIT, filter=query_filter),
                models.Prefetch(query=_sparse_query(query), using="sparse",
                                limit=PREFETCH_LIMIT, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    return res.points


def search_it_deduped(query: str, mode: str = "hybrid",
                      n_families: int = 3, per_family: int = 2):
    """IT-specific retrieval: the corpus has ~50 near-duplicates per issue
    family, so naive top-k returns 5 paraphrases of one incident. Retrieve
    wide, then keep the top `per_family` hits from the top `n_families`
    distinct families.
    """
    from config import IT_COLLECTION

    hits = search(IT_COLLECTION, query, mode=mode, top_k=25)
    kept, family_counts, family_order = [], {}, []
    for h in hits:
        fam = h.payload.get("family_id")
        if fam not in family_counts:
            if len(family_order) >= n_families:
                continue
            family_order.append(fam)
            family_counts[fam] = 0
        if family_counts[fam] < per_family:
            kept.append(h)
            family_counts[fam] += 1
    return kept
