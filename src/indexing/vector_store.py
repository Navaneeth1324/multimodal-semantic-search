"""Qdrant vector store manager for indexing and similarity search."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams

from src.ingestion.schemas import ProductItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "products"


class QdrantVectorStore:
    """Manages collection lifecycle, vector upserts, and payload-filtered ANN search."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        storage_path: Optional[str] = "qdrant_storage",
        in_memory: bool = False,
    ):
        self.collection_name = collection_name
        if in_memory:
            self.client = QdrantClient(location=":memory:")
            logger.info("Initialized Qdrant in-memory instance")
        elif storage_path:
            self.client = QdrantClient(path=storage_path)
            logger.info("Initialized local persistent Qdrant at %s", storage_path)
        else:
            self.client = QdrantClient(location=":memory:")

    def create_collection(self, vector_size: int, distance: Distance = Distance.COSINE, recreate: bool = False) -> None:
        """Initializes the vector collection with specified dimension and distance metric."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            if recreate:
                logger.info("Recreating collection '%s'", self.collection_name)
                self.client.delete_collection(self.collection_name)
            else:
                logger.info("Collection '%s' already exists.", self.collection_name)
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        logger.info("Created collection '%s' (dim=%d, metric=%s)", self.collection_name, vector_size, distance)

    def upsert_products(self, products: List[ProductItem], embeddings: np.ndarray | List[List[float]]) -> int:
        """Upserts a list of products with corresponding dense vectors into the vector store."""
        if len(products) != len(embeddings):
            raise ValueError(f"Mismatch between products count ({len(products)}) and embeddings ({len(embeddings)})")

        points = []
        for idx, (product, vec) in enumerate(zip(products, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, product.id))
            vector = vec.tolist() if isinstance(vec, np.ndarray) else vec
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=product.model_dump(),
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Successfully upserted %d points to collection '%s'", len(points), self.collection_name)
        return len(points)

    def search(
        self,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        query_vec: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Performs Approximate Nearest Neighbors (ANN) vector search with optional payload filters."""
        vector = query_vector if query_vector is not None else query_vec
        if vector is None:
            raise ValueError("Either query_vector or query_vec must be supplied.")

        must_filters: List[rest.Condition] = []

        if category:
            must_filters.append(
                rest.FieldCondition(
                    key="category",
                    match=rest.MatchValue(value=category),
                )
            )

        if max_price is not None:
            must_filters.append(
                rest.FieldCondition(
                    key="price",
                    range=rest.Range(lte=max_price),
                )
            )

        query_filter = rest.Filter(must=must_filters) if must_filters else None

        if hasattr(self.client, "query_points"):
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
            )
            results = query_response.points
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
            )

        hits: List[Dict[str, Any]] = []
        for res in results:
            item = dict(res.payload or {})
            item["score"] = round(float(res.score), 4)
            hits.append(item)

        return hits

    def count(self) -> int:
        """Returns total vector count inside the collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count
