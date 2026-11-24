"""Orchestrates embedding generation and indexing of catalog products."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from src.indexing.embedder import DenseEmbedder
from src.indexing.vector_store import QdrantVectorStore
from src.ingestion.schemas import ProductItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_index(
    data_path: str = "data/products.json",
    storage_path: str = "qdrant_storage",
    recreate: bool = True,
) -> int:
    """Loads products, computes embeddings, and indexes them in Qdrant."""
    file_path = Path(data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Product catalog not found at {data_path}. Run `python -m src.ingestion.load_data` first.")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    products = [ProductItem(**item) for item in raw_items]
    logger.info("Loaded %d products from %s", len(products), data_path)

    # 1. Initialize embedder
    embedder = DenseEmbedder()

    # 2. Generate rich searchable texts
    search_texts = [p.searchable_text for p in products]

    start_time = time.time()
    embeddings = embedder.embed_batch(search_texts)
    embedding_duration = time.time() - start_time
    logger.info("Generated %d embeddings in %.2fs (dim=%d)", len(embeddings), embedding_duration, embedder.dimension)

    # 3. Initialize and populate vector store
    store = QdrantVectorStore(storage_path=storage_path)
    store.create_collection(vector_size=embedder.dimension, recreate=recreate)
    inserted = store.upsert_products(products, embeddings)

    logger.info("Indexed %d items successfully into Qdrant store at '%s'", inserted, storage_path)
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Index catalog into Qdrant vector database.")
    parser.add_argument("--data-path", type=str, default="data/products.json", help="Path to products JSON file.")
    parser.add_argument("--storage-path", type=str, default="qdrant_storage", help="Local Qdrant directory.")
    parser.add_argument("--recreate", action="store_true", default=True, help="Recreate collection if it exists.")
    args = parser.parse_args()

    count = build_index(data_path=args.data_path, storage_path=args.storage_path, recreate=args.recreate)
    print(f"\n[DONE] Successfully indexed {count} items!")


if __name__ == "__main__":
    main()
