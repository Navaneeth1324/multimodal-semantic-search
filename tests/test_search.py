"""Unit and integration tests for Day 3: Two-stage retrieval and reranking."""

import pytest

from src.indexing.embedder import DenseEmbedder
from src.indexing.vector_store import QdrantVectorStore
from src.ingestion.schemas import ProductItem
from src.search.pipeline import TwoStageSearchPipeline
from src.search.reranker import CrossEncoderReranker, sigmoid
from src.search.retriever import VectorRetriever


@pytest.fixture(scope="module")
def shared_pipeline():
    """Initializes an in-memory two-stage search pipeline with populated test data."""
    embedder = DenseEmbedder()
    store = QdrantVectorStore(in_memory=True)
    store.create_collection(vector_size=embedder.dimension, recreate=True)

    items = [
        ProductItem(
            id="TEST-001",
            title="AeroStride Carbon Marathon Shoes",
            description="Elite marathon racing shoes with carbon plate and responsive foam.",
            category="Footwear",
            brand="AeroStride",
            price=160.00,
            rating=4.8,
            tags=["running", "marathon", "shoes"],
        ),
        ProductItem(
            id="TEST-002",
            title="Casual Walking Loafers",
            description="Everyday slip-on loafers for comfortable walking and casual wear.",
            category="Footwear",
            brand="CloudWalk",
            price=70.00,
            rating=4.3,
            tags=["walking", "loafers", "casual"],
        ),
        ProductItem(
            id="TEST-003",
            title="SoundSilence Wireless Noise-Cancelling Headphones",
            description="Over-ear active noise cancelling headphones for audiophiles.",
            category="Audio",
            brand="SoundSilence",
            price=250.00,
            rating=4.9,
            tags=["headphones", "audio", "anc"],
        ),
    ]

    texts = [item.searchable_text for item in items]
    vectors = embedder.embed_batch(texts)
    store.upsert_products(items, vectors)

    reranker = CrossEncoderReranker()
    pipeline = TwoStageSearchPipeline(embedder=embedder, vector_store=store, reranker=reranker)
    return pipeline


def test_sigmoid_function():
    import numpy as np
    logits = np.array([-10.0, 0.0, 10.0])
    probs = sigmoid(logits)
    assert abs(probs[1] - 0.5) < 1e-4
    assert probs[0] < 0.01
    assert probs[2] > 0.99


def test_retriever_alone(shared_pipeline):
    retriever = shared_pipeline.retriever
    res = retriever.retrieve("running shoes", top_k=2)

    assert "candidates" in res
    assert "retrieval_ms" in res
    assert len(res["candidates"]) == 2
    assert res["candidates"][0]["category"] == "Footwear"


def test_retriever_empty_query(shared_pipeline):
    retriever = shared_pipeline.retriever
    res = retriever.retrieve("   ", top_k=2)
    assert res["candidates"] == []
    assert res["retrieval_ms"] == 0.0


def test_reranker_alone(shared_pipeline):
    reranker = shared_pipeline.reranker
    candidates = [
        {"id": "A", "title": "Casual Walking Loafers", "description": "Comfort loafers.", "tags": ["shoes"]},
        {"id": "B", "title": "AeroStride Carbon Marathon Shoes", "description": "Carbon racing shoes.", "tags": ["racing"]},
    ]
    query = "carbon plate racing marathon sneakers"
    out = reranker.rerank(query, candidates, top_k=2)

    assert "results" in out
    assert len(out["results"]) == 2
    # Candidate B is far more relevant to carbon plate racing
    assert out["results"][0]["id"] == "B"
    assert out["results"][0]["rerank_score"] > out["results"][1]["rerank_score"]


def test_pipeline_end_to_end_reranked(shared_pipeline):
    query = "marathon racing footwear"
    result = shared_pipeline.search(query=query, top_k=2, enable_rerank=True)

    assert result.reranked is True
    assert result.total_candidates > 0
    assert len(result.results) <= 2
    assert result.results[0]["id"] == "TEST-001"
    assert result.retrieval_ms > 0
    assert result.rerank_ms > 0
    assert result.total_ms >= (result.retrieval_ms + result.rerank_ms)


def test_pipeline_filtering(shared_pipeline):
    query = "audio device"
    # Filter by price <= 100 on audio device (should return empty since Headphones are $250)
    res = shared_pipeline.search(query=query, category="Audio", max_price=100.0)
    assert len(res.results) == 0

    # Without max_price filter, should return headphones
    res_valid = shared_pipeline.search(query=query, category="Audio")
    assert len(res_valid.results) == 1
    assert res_valid.results[0]["id"] == "TEST-003"
