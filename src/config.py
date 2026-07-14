"""Shared configuration for the hybrid RAG pipeline."""

import os

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
# Load environment variables (API keys, model names)
from dotenv import load_dotenv
load_dotenv()
# --- Qdrant ---------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
HR_COLLECTION = "hr_policies"
IT_COLLECTION = "it_tickets"

# --- Dense embeddings: local (fastembed) or OpenAI -------------------------
# These three MUST change together (model ↔ dim ↔ prefix).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # "local" | "openai"

if EMBEDDING_PROVIDER == "openai":
    DENSE_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    DENSE_DIM = 1536
    QUERY_PREFIX = ""                       # OpenAI models: no prefix
else:
    DENSE_MODEL = "BAAI/bge-small-en-v1.5"
    DENSE_DIM = 384
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SPARSE_MODEL = "Qdrant/bm25"
N_IT_FAMILIES = 15

# --- LLM: OpenAI if key present, else Ollama --------------------------------
_llm = None


def get_llm():
    """Lazy LLM factory. Returns a LangChain chat model either way, so
    consumers (graph.py) never care which provider is behind it."""
    global _llm
    if _llm is None:
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                temperature=0, timeout=120, max_retries=3,
            )
        else:
            from langchain_ollama import ChatOllama
            _llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0
            )
    return _llm


def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


_dense = None
_sparse = None


def dense_model():
    """Returns an object with .embed(texts) regardless of provider."""
    global _dense
    if _dense is None:
        if EMBEDDING_PROVIDER == "openai":
            _dense = _OpenAIEmbedder(DENSE_MODEL)
        else:
            _dense = TextEmbedding(DENSE_MODEL)
    return _dense


def sparse_model() -> SparseTextEmbedding:
    global _sparse
    if _sparse is None:
        _sparse = SparseTextEmbedding(SPARSE_MODEL)
    return _sparse


class _OpenAIEmbedder:
    """Adapter so OpenAI embeddings look like fastembed to the rest of the code."""

    def __init__(self, model: str):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def embed(self, texts, batch_size: int = 512):
        import numpy as np
        texts = list(texts)
        for start in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(
                model=self.model, input=texts[start:start + batch_size]
            )
            for item in resp.data:
                yield np.array(item.embedding)