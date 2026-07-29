"""Qdrant helpers shared by both ingestion scripts."""

from typing import Iterable

from langsmith import traceable
from qdrant_client import QdrantClient, models

from config import DENSE_DIM, dense_model, sparse_model


def recreate_hybrid_collection(client: QdrantClient, name: str, force: bool = False) -> None:
    """Create a collection with named dense + sparse vectors (drops existing).

    If the existing collection holds points, it's snapshotted before being
    dropped (so a bad ingest run is recoverable via client.recover_snapshot),
    and a confirmation prompt is shown unless `force` is set (e.g. via a
    --force CLI flag for automation/CI).
    """
    if client.collection_exists(name):
        count = client.count(name).count
        if count > 0:
            if not force:
                answer = input(
                    f"Collection '{name}' exists with {count} points. "
                    "Type YES to drop and recreate it: "
                )
                if answer.strip() != "YES":
                    raise SystemExit("Aborted: collection was not recreated.")
            snapshot = client.create_snapshot(collection_name=name)
            print(f"Snapshotted '{name}' -> {snapshot.name} before dropping "
                  "(restore with client.recover_snapshot(...) if needed).")
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


@traceable(run_type="embedding", name="embed_texts")
def embed_texts(texts: list[str]):
    """Return (dense, sparse) embeddings for a list of passages."""
    dense = list(dense_model().embed(texts, batch_size=64))
    sparse = list(sparse_model().embed(texts, batch_size=64))
    return dense, sparse


@traceable(run_type="tool", name="upsert_hybrid")
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
