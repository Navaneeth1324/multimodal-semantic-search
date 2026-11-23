"""Unit tests for Day 1: Data schemas and ingestion pipeline."""
import json
import pytest
from pydantic import ValidationError

from src.ingestion.schemas import ProductItem, DatasetManifest
from src.ingestion.load_data import load_and_validate_catalog, export_dataset


def test_product_item_valid():
    item = ProductItem(
        id="TEST-001",
        title="  Wireless Headphones  ",
        description="High fidelity noise cancelling bluetooth headphones with mic.",
        category="Audio",
        brand="SoundCo",
        price=199.99,
        rating=4.7,
        tags=["wireless", "audio", "anc"],
    )
    # Whitespace stripping
    assert item.title == "Wireless Headphones"
    assert item.price == 199.99
    # Rich searchable text formulation
    assert "Title: Wireless Headphones" in item.searchable_text
    assert "Category: Audio" in item.searchable_text
    assert "wireless, audio, anc" in item.searchable_text


def test_product_item_invalid_price():
    with pytest.raises(ValidationError):
        ProductItem(
            id="TEST-002",
            title="Bad Product",
            description="Testing negative price boundary condition.",
            category="Test",
            brand="Test",
            price=-10.0,  # Negative price should fail gt=0
        )


def test_product_item_invalid_rating():
    with pytest.raises(ValidationError):
        ProductItem(
            id="TEST-003",
            title="Overrated Product",
            description="Testing out-of-range rating boundary condition.",
            category="Test",
            brand="Test",
            price=20.0,
            rating=6.5,  # Max allowed is 5.0
        )


def test_product_item_empty_title():
    with pytest.raises(ValidationError):
        ProductItem(
            id="TEST-004",
            title="   ",  # Blank should fail strip_whitespace
            description="Testing empty title rejection.",
            category="Test",
            brand="Test",
            price=20.0,
        )


def test_export_dataset_pipeline(tmp_path):
    catalog = load_and_validate_catalog()
    assert len(catalog) > 0

    manifest = export_dataset(catalog, tmp_path)

    # Validate output files exist
    json_file = tmp_path / "products.json"
    csv_file = tmp_path / "products.csv"
    manifest_file = tmp_path / "manifest.json"

    assert json_file.exists()
    assert csv_file.exists()
    assert manifest_file.exists()

    # Validate manifest values
    assert manifest.total_items == len(catalog)
    assert len(manifest.categories) > 0
    assert manifest.avg_price > 0

    with open(json_file, "r") as f:
        loaded_json = json.load(f)
    assert len(loaded_json) == len(catalog)
