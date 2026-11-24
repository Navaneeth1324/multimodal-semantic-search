"""Two-stage retrieval and reranking search pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.indexing.embedder import DenseEmbedder
from src.indexing.vector_store import QdrantVectorStore
from src.search.reranker import CrossEncoderReranker
from src.search.retriever import VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Structured response returned by the search pipeline."""
    query: str
    results: List[Dict[str, Any]]
    total_candidates: int
    retrieval_ms: float
    rerank_ms: float
    total_ms: float
    reranked: bool


class TwoStageSearchPipeline:
    """Orchestrates candidate retrieval (Stage 1) followed by contextual reranking (Stage 2)."""

    def __init__(
        self,
        embedder: Optional[DenseEmbedder] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.embedder = embedder or DenseEmbedder()
        self.vector_store = vector_store or QdrantVectorStore()
        self.retriever = VectorRetriever(embedder=self.embedder, vector_store=self.vector_store)
        self.reranker = reranker or CrossEncoderReranker()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        enable_rerank: bool = True,
    ) -> SearchResult:
        """Executes full search query lifecycle with latency profiling."""
        pipeline_start = time.perf_counter()

        # Stage 1: ANN Candidate Retrieval
        retrieval_response = self.retriever.retrieve(
            query=query,
            top_k=candidate_pool,
            category=category,
            max_price=max_price,
        )
        candidates = retrieval_response["candidates"]
        retrieval_ms = retrieval_response["retrieval_ms"]

        # Stage 2: Cross-Encoder Reranking (optional)
        if enable_rerank and candidates:
            rerank_response = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
            final_results = rerank_response["results"]
            rerank_ms = rerank_response["rerank_ms"]
            was_reranked = True
        else:
            final_results = candidates[:top_k]
            rerank_ms = 0.0
            was_reranked = False

        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

        return SearchResult(
            query=query,
            results=final_results,
            total_candidates=len(candidates),
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
            reranked=was_reranked,
        )
