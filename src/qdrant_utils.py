"""Qdrant helpers shared by both ingestion scripts."""

from typing import Iterable

from qdrant_client import QdrantClient, models

from config import DENSE_DIM, dense_model, sparse_model


def recreate_hybrid_collection(client: QdrantClient, name: str) -> None:
    """Create a collection with named dense + sparse vectors (drops existing)."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            # IDF modifier -> server-side IDF, required for proper BM25 behavior.
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def embed_texts(texts: list[str]):
    """Return (dense, sparse) embeddings for a list of passages."""
    dense = list(dense_model().embed(texts, batch_size=64))
    sparse = list(sparse_model().embed(texts, batch_size=64))
    return dense, sparse


def upsert_hybrid(
    client: QdrantClient,
    collection: str,
    texts: list[str],
    payloads: list[dict],
    batch_size: int = 128,
) -> None:
    """Embed `texts` and upsert with `payloads` in batches."""
    assert len(texts) == len(payloads)
    dense, sparse = embed_texts(texts)

    points = [
        models.PointStruct(
            id=i,
            vector={
                "dense": dense[i].tolist(),
                "sparse": models.SparseVector(
                    indices=sparse[i].indices.tolist(),
                    values=sparse[i].values.tolist(),
                ),
            },
            payload=payloads[i],
        )
        for i in range(len(texts))
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[start : start + batch_size])


def batched(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
