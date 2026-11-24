"""Dense vector embedding generator using SentenceTransformers."""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_device() -> str:
    """Detects available hardware accelerator (CUDA, Apple MPS, or fallback CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseEmbedder:
    """Generates normalized dense vector embeddings for catalog items and search queries."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.device = device or resolve_device()
        self.model_name = model_name
        logger.info("Initializing DenseEmbedder (%s) on device: %s", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dim = self.model.get_embedding_dimension()
        else:
            self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality (e.g., 384)."""
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        """Encodes a single text string into a normalized unit vector."""
        embedding = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encodes a batch of text strings into normalized vectors."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 20,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings
