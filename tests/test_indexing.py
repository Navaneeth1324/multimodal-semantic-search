"""Unit and integration tests for Day 2: Vector embedding and Qdrant indexing."""

import numpy as np
import pytest

from src.indexing.embedder import DenseEmbedder
from src.indexing.vector_store import QdrantVectorStore
from src.ingestion.schemas import ProductItem


@pytest.fixture(scope="module")
def shared_embedder():
    """Shared embedder instance to avoid re-downloading/re-initializing across tests."""
    return DenseEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def sample_products():
    return [
        ProductItem(
            id="TEST-001",
            title="AeroStride Marathon Carbon Running Shoes",
            description="Ultra lightweight carbon plate racing shoes built for long distance road running.",
            category="Footwear",
            brand="AeroStride",
            price=150.00,
            rating=4.8,
            tags=["running", "marathon", "shoes"],
        ),
        ProductItem(
            id="TEST-002",
            title="SoundSilence Wireless ANC Headphones",
            description="Active noise cancelling over-ear headphones with 40 hour battery life.",
            category="Audio",
            brand="SoundSilence",
            price=250.00,
            rating=4.9,
            tags=["headphones", "audio", "anc"],
        ),
        ProductItem(
            id="TEST-003",
            title="Budget Sport In-Ear Wireless Earbuds",
            description="Affordable sweatproof workout earphones with deep bass.",
            category="Audio",
            brand="BassPro",
            price=45.00,
            rating=4.2,
            tags=["earbuds", "budget", "gym"],
        ),
    ]


def test_embedder_dimensions_and_normalization(shared_embedder):
    text = "High performance athletic running footwear"
    vec = shared_embedder.embed_text(text)

    assert len(vec) == 384
    # Unit length check (L2 norm should be ~1.0)
    norm = np.linalg.norm(np.array(vec))
    assert abs(norm - 1.0) < 1e-4


def test_embedder_batch(shared_embedder):
    texts = [
        "First product description",
        "Second product description",
    ]
    matrix = shared_embedder.embed_batch(texts)
    assert matrix.shape == (2, 384)


def test_qdrant_vector_store_workflow(shared_embedder, sample_products):
    store = QdrantVectorStore(in_memory=True)
    store.create_collection(vector_size=shared_embedder.dimension, recreate=True)

    # Compute embeddings
    texts = [p.searchable_text for p in sample_products]
    embeddings = shared_embedder.embed_batch(texts)

    # Upsert
    inserted = store.upsert_products(sample_products, embeddings)
    assert inserted == len(sample_products)
    assert store.count() == len(sample_products)

    # Semantic search query for running shoes
    query_vec = shared_embedder.embed_text("running shoes for distance athlete")
    hits = store.search(query_vec=query_vec, top_k=2)

    assert len(hits) > 0
    # Top result must be the footwear item
    assert hits[0]["category"] == "Footwear"
    assert "AeroStride" in hits[0]["title"]
    assert hits[0]["score"] > 0.4


def test_qdrant_payload_filtering(shared_embedder, sample_products):
    store = QdrantVectorStore(in_memory=True)
    store.create_collection(vector_size=shared_embedder.dimension, recreate=True)

    texts = [p.searchable_text for p in sample_products]
    embeddings = shared_embedder.embed_batch(texts)
    store.upsert_products(sample_products, embeddings)

    # Query audio with max_price filter
    query_vec = shared_embedder.embed_text("music listening headphones")

    # Filter: category=Audio and price <= 100
    filtered_hits = store.search(query_vec=query_vec, top_k=5, category="Audio", max_price=100.0)

    assert len(filtered_hits) == 1
    assert filtered_hits[0]["id"] == "TEST-003"  # Budget Earbuds ($45), not $250 headphones
    assert filtered_hits[0]["price"] <= 100.0
