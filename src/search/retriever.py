"""Stage 1: Vector Retriever using dense embeddings and Qdrant."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.indexing.embedder import DenseEmbedder
from src.indexing.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VectorRetriever:
    """Stage 1 Candidate Generator: Performs sub-linear ANN search over dense vector index."""

    def __init__(self, embedder: DenseEmbedder, vector_store: QdrantVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Encodes query and returns top candidate items along with retrieval latency."""
        start_time = time.perf_counter()

        query_cleaned = query.strip()
        if not query_cleaned:
            return {"candidates": [], "retrieval_ms": 0.0}

        # 1. Dense query vector
        query_vector = self.embedder.embed_text(query_cleaned)

        # 2. Approximate Nearest Neighbor search
        candidates = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            category=category,
            max_price=max_price,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.debug("Retrieved %d candidates for query '%s' in %.2f ms", len(candidates), query_cleaned, elapsed_ms)

        return {
            "candidates": candidates,
            "retrieval_ms": elapsed_ms,
        }
