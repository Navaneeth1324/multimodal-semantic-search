"""Data schemas for catalog items and ingestion manifests."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ProductItem(BaseModel):
    """Schema for a single product catalog item."""
    id: str = Field(..., description="Unique product identifier (e.g., PROD-001)")
    title: str = Field(..., min_length=3, max_length=200, description="Product display name")
    description: str = Field(..., min_length=10, description="Detailed product description")
    category: str = Field(..., description="Top-level category (e.g., Electronics, Footwear)")
    brand: str = Field(..., description="Manufacturer / brand name")
    price: float = Field(..., gt=0, description="Product price in USD")
    rating: float = Field(default=4.5, ge=0.0, le=5.0, description="Average customer rating")
    tags: List[str] = Field(default_factory=list, description="Descriptive keyword tags")
    image_url: Optional[str] = Field(default=None, description="External or local path to product image")

    @field_validator("title", "description", "category", "brand")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or only whitespace")
        return trimmed

    @property
    def searchable_text(self) -> str:
        """Combines attributes into a rich text block optimized for dense semantic embedding."""
        tag_str = ", ".join(self.tags)
        return (
            f"Title: {self.title}. "
            f"Category: {self.category}. "
            f"Brand: {self.brand}. "
            f"Tags: {tag_str}. "
            f"Description: {self.description}"
        )


class DatasetManifest(BaseModel):
    """Summary statistics for the ingested dataset."""
    total_items: int
    categories: List[str]
    avg_price: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)
