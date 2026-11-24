"""Stage 2: Cross-Encoder Reranker for high-precision semantic re-ranking."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.indexing.embedder import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Computes stable sigmoid mapping unbounded logits to [0, 1] confidence intervals."""
    return 1 / (1 + np.exp(-np.clip(x, -25.0, 25.0)))


class CrossEncoderReranker:
    """Stage 2 Contextual Reranker: Evaluates full query-document cross-attention pairs."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        self.device = device or resolve_device()
        self.model_name = model_name
        logger.info("Initializing CrossEncoderReranker (%s) on device: %s", self.model_name, self.device)
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Reranks candidate documents using deep cross-encoder token interactions."""
        start_time = time.perf_counter()

        if not candidates or not query.strip():
            return {"results": [], "rerank_ms": 0.0}

        # Format input pairs: (query, candidate_text)
        pairs = []
        for c in candidates:
            title = c.get("title", "")
            desc = c.get("description", "")
            tags = ", ".join(c.get("tags", []))
            candidate_text = f"Title: {title}. Description: {desc}. Tags: {tags}"
            pairs.append((query, candidate_text))

        # Compute raw logit scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        normalized_scores = sigmoid(np.array(scores))

        # Attach rerank score and sort
        reranked_items = []
        for candidate, raw_score, norm_score in zip(candidates, scores, normalized_scores):
            item = dict(candidate)
            item["vector_score"] = item.get("score", 0.0)
            item["rerank_score"] = round(float(norm_score), 4)
            item["rerank_logit"] = round(float(raw_score), 4)
            reranked_items.append(item)

        # Sort descending by rerank score
        reranked_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = reranked_items[:top_k]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.debug("Reranked %d candidates down to %d in %.2f ms", len(candidates), len(top_results), elapsed_ms)

        return {
            "results": top_results,
            "rerank_ms": elapsed_ms,
        }
