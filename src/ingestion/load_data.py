"""Data ingestion and dataset generator module.

Builds a validated, diverse e-commerce dataset for semantic search
and vector retrieval experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.ingestion.schemas import DatasetManifest, ProductItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Realistic seed items spanning major categories
SEED_CATALOG = [
    {
        "id": "PROD-001",
        "title": "AeroStride Pro Ultra Marathon Running Shoes",
        "category": "Footwear",
        "brand": "AeroStride",
        "price": 159.99,
        "rating": 4.8,
        "tags": ["running", "marathon", "breathable", "cushioning", "lightweight", "carbon-plate"],
        "description": "High-performance marathon running shoes with embedded carbon fiber plate and responsive foam cushioning. Engineered for long-distance road racing and maximum energy return.",
        "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600",
    },
    {
        "id": "PROD-002",
        "title": "CloudStep Slip-On Ergonomic Walking Loafers",
        "category": "Footwear",
        "brand": "CloudStep",
        "price": 79.50,
        "rating": 4.5,
        "tags": ["walking", "slip-on", "orthopedic", "comfort", "casual", "memory-foam"],
        "description": "Everyday slip-on loafers featuring dual-density memory foam insoles, breathable canvas upper, and anti-fatigue arch support for all-day walking comfort.",
        "image_url": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=600",
    },
    {
        "id": "PROD-003",
        "title": "TerraTrek Waterproof All-Terrain Hiking Boots",
        "category": "Footwear",
        "brand": "TerraTrek",
        "price": 189.00,
        "rating": 4.7,
        "tags": ["hiking", "waterproof", "trail", "vibram-sole", "outdoor", "ankle-support"],
        "description": "Rugged mid-cut hiking boots with Gore-Tex waterproof membrane and Vibram deep-lug traction outsoles for muddy backcountry trails and rocky terrain.",
        "image_url": "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?w=600",
    },
    {
        "id": "PROD-004",
        "title": "SoundSilence QuietWave Wireless ANC Headphones",
        "category": "Audio",
        "brand": "SoundSilence",
        "price": 249.99,
        "rating": 4.9,
        "tags": ["anc", "wireless", "bluetooth", "audiophile", "travel", "40h-battery"],
        "description": "Premium over-ear wireless headphones with active noise cancellation, transparency mode, custom 40mm beryllium drivers, and 40-hour rapid-charge battery.",
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600",
    },
    {
        "id": "PROD-005",
        "title": "BassPulse True Wireless Sport Earbuds",
        "category": "Audio",
        "brand": "BassPulse",
        "price": 89.95,
        "rating": 4.3,
        "tags": ["earbuds", "sweatproof", "ipx7", "gym", "wireless", "punchy-bass"],
        "description": "Secure-fit wireless earbuds with IPX7 sweat resistance, ergonomic ear hooks, touch controls, and enhanced sub-bass profile built for intense gym workouts.",
        "image_url": "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?w=600",
    },
    {
        "id": "PROD-006",
        "title": "RetroVibe Hi-Fi Turntable with Built-in Phono Preamp",
        "category": "Audio",
        "brand": "RetroVibe",
        "price": 299.00,
        "rating": 4.6,
        "tags": ["vinyl", "turntable", "vintage", "analog", "hifi", "walnut-wood"],
        "description": "Belt-drive vinyl record player crafted in natural walnut veneer, fitted with an Audio-Technica magnetic cartridge, anti-vibration feet, and switchable preamp.",
        "image_url": "https://images.unsplash.com/photo-1539185441755-769473a23570?w=600",
    },
    {
        "id": "PROD-007",
        "title": "Chronos Apex Titanium Multisport GPS Smartwatch",
        "category": "Wearables",
        "brand": "Chronos",
        "price": 399.00,
        "rating": 4.8,
        "tags": ["smartwatch", "gps", "heart-rate", "titanium", "sapphire-glass", "triathlon"],
        "description": "Rugged titanium adventure smartwatch with scratch-resistant sapphire crystal, dual-frequency GPS navigation, ECG sensor, and 14-day battery life.",
        "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600",
    },
    {
        "id": "PROD-008",
        "title": "AuraBand Minimalist Sleep & Recovery Fitness Tracker",
        "category": "Wearables",
        "brand": "Aura",
        "price": 149.00,
        "rating": 4.4,
        "tags": ["fitness-tracker", "sleep-tracking", "hrv", "screenless", "waterproof"],
        "description": "Screen-free health tracker band monitoring continuous HRV, deep sleep stages, skin temperature trends, and daily recovery scores.",
        "image_url": "https://images.unsplash.com/photo-1576243345690-4e4b79b63288?w=600",
    },
    {
        "id": "PROD-009",
        "title": "LumixCraft 4K Creator Vlogging Mirrorless Camera",
        "category": "Photography",
        "brand": "LumixCraft",
        "price": 799.00,
        "rating": 4.7,
        "tags": ["camera", "4k-video", "mirrorless", "vlogging", "flip-screen", "content-creator"],
        "description": "Compact interchangeable lens mirrorless camera featuring 24.2MP APS-C sensor, fully articulating flip touchscreen, 4K 60p uncropped video, and eye-tracking AF.",
        "image_url": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?w=600",
    },
    {
        "id": "PROD-010",
        "title": "GimbalPro 3-Axis Smartphone Video Stabilizer",
        "category": "Photography",
        "brand": "GimbalPro",
        "price": 119.00,
        "rating": 4.5,
        "tags": ["gimbal", "stabilizer", "mobile-video", "timelapse", "magnetic-clamp"],
        "description": "Foldable motorized 3-axis handheld smartphone gimbal with AI subject tracking, built-in extension rod, and intuitive joystick control for cinematic mobile video.",
        "image_url": "https://images.unsplash.com/photo-1588854337236-6889d631faa8?w=600",
    },
    {
        "id": "PROD-011",
        "title": "ThermalPeak 800-Fill Goose Down Winter Parka",
        "category": "Apparel",
        "brand": "ThermalPeak",
        "price": 320.00,
        "rating": 4.8,
        "tags": ["jacket", "down-parka", "winter", "thermal", "windproof", "sub-zero"],
        "description": "Expedition-grade insulated winter jacket stuffed with ethically sourced 800-fill goose down, featuring storm-flap closure, fleece-lined hand pockets, and DWR shell.",
        "image_url": "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=600",
    },
    {
        "id": "PROD-012",
        "title": "BreezeFlex Merino Wool Everyday Crewneck Tee",
        "category": "Apparel",
        "brand": "BreezeFlex",
        "price": 68.00,
        "rating": 4.6,
        "tags": ["tshirt", "merino-wool", "odor-resistant", "travel", "minimalist", "breathable"],
        "description": "100% superfine merino wool t-shirt providing natural temperature regulation, moisture-wicking comfort, and multi-day odor resistance for travel and daily wear.",
        "image_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=600",
    },
    {
        "id": "PROD-013",
        "title": "BaristaPro Dual-Boiler Espresso Machine with PID",
        "category": "Home & Kitchen",
        "brand": "BaristaPro",
        "price": 649.99,
        "rating": 4.9,
        "tags": ["espresso", "coffee-maker", "dual-boiler", "pid-temp", "latte-art", "stainless-steel"],
        "description": "Commercial-inspired stainless steel home espresso machine with independent brew and steam boilers, PID temperature precision, and professional 58mm portafilter.",
        "image_url": "https://images.unsplash.com/photo-1517668808822-9ebb02f2a0e6?w=600",
    },
    {
        "id": "PROD-014",
        "title": "AeroPress Go Portable Travel Coffee Maker",
        "category": "Home & Kitchen",
        "brand": "AeroPress",
        "price": 39.95,
        "rating": 4.8,
        "tags": ["coffee", "travel", "camping", "aeropress", "portable", "smooth-brew"],
        "description": "Compact all-in-one immersion coffee press that nests neatly inside its own drinking mug. Brews rich, low-acidity espresso-style concentrate in 1 minute.",
        "image_url": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600",
    },
    {
        "id": "PROD-015",
        "title": "HydroChamber Vacuum Insulated Stainless Steel Tumbler",
        "category": "Home & Kitchen",
        "brand": "HydroChamber",
        "price": 34.50,
        "rating": 4.7,
        "tags": ["water-bottle", "insulated", "stainless-steel", "bpa-free", "cold-24h", "straw-lid"],
        "description": "32oz double-wall vacuum insulated tumbler keeping liquids cold for 24 hours or piping hot for 12 hours. Powder-coated grip and leak-resistant lid.",
        "image_url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=600",
    },
    {
        "id": "PROD-016",
        "title": "MechMaster RGB Hot-Swappable Mechanical Keyboard",
        "category": "Computing",
        "brand": "MechMaster",
        "price": 129.99,
        "rating": 4.7,
        "tags": ["keyboard", "mechanical", "hot-swap", "rgb", "gasket-mount", "wireless"],
        "description": "75% compact mechanical keyboard with custom pre-lubed linear switches, gasket-mounted sound dampening, CNC aluminum frame, and tri-mode wireless connectivity.",
        "image_url": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=600",
    },
    {
        "id": "PROD-017",
        "title": "ErgoGlide Vertical Wireless Ergonomic Mouse",
        "category": "Computing",
        "brand": "ErgoGlide",
        "price": 64.95,
        "rating": 4.5,
        "tags": ["mouse", "ergonomic", "vertical", "wrist-pain", "bluetooth", "silent-click"],
        "description": "Scientifically contoured 57-degree vertical ergonomic mouse designed to alleviate forearm strain and carpal tunnel pressure during long desk hours.",
        "image_url": "https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?w=600",
    },
    {
        "id": "PROD-018",
        "title": "NomadShield Rugged Waterproof 28L Travel Backpack",
        "category": "Accessories",
        "brand": "NomadShield",
        "price": 139.00,
        "rating": 4.8,
        "tags": ["backpack", "travel", "waterproof", "laptop-sleeve", "cordura", "carry-on"],
        "description": "TSA-friendly 28-liter commuter backpack made from indestructible Cordura ballistic nylon. Features dedicated 16-inch padded laptop pocket and luggage passthrough.",
        "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=600",
    },
    {
        "id": "PROD-019",
        "title": "OptiVision Polarized Acetate Classic Sunglasses",
        "category": "Accessories",
        "brand": "OptiVision",
        "price": 95.00,
        "rating": 4.4,
        "tags": ["sunglasses", "polarized", "uv400", "classic", "acetate", "eyewear"],
        "description": "Handcrafted Italian acetate sunglasses fitted with category-3 polarized lenses delivering 100% UVA/UVB protection and crystal-clear glare reduction.",
        "image_url": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=600",
    },
    {
        "id": "PROD-020",
        "title": "ZenMat High-Density Eco Non-Slip Yoga Mat",
        "category": "Fitness",
        "brand": "ZenMat",
        "price": 54.00,
        "rating": 4.6,
        "tags": ["yoga", "fitness", "eco-friendly", "non-slip", "cushion", "pilates"],
        "description": "6mm extra-thick biodegradable natural tree rubber yoga mat offering superior joint cushioning and laser-etched alignment guides for non-slip grip.",
        "image_url": "https://images.unsplash.com/photo-1545205597-3d9d02c29597?w=600",
    },
]


def load_and_validate_catalog(seed_data: List[dict] = SEED_CATALOG) -> List[ProductItem]:
    """Validates raw dictionary records against the ProductItem Pydantic model."""
    valid_products: List[ProductItem] = []
    for idx, raw_record in enumerate(seed_data):
        try:
            item = ProductItem(**raw_record)
            valid_products.append(item)
        except Exception as exc:
            logger.warning("Skipping invalid product at index %d: %s", idx, exc)

    logger.info("Validated %d / %d records successfully.", len(valid_products), len(seed_data))
    return valid_products


def export_dataset(products: List[ProductItem], output_dir: Path) -> DatasetManifest:
    """Exports validated records to JSON, CSV, and writes a dataset summary manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "products.json"
    csv_path = output_dir / "products.csv"
    manifest_path = output_dir / "manifest.json"

    # Export JSON
    dict_records = [p.model_dump() for p in products]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dict_records, f, indent=2)

    # Export CSV
    df = pd.DataFrame(dict_records)
    # Add computed searchable text column for quick inspection
    df["searchable_text"] = [p.searchable_text for p in products]
    df.to_csv(csv_path, index=False)

    # Compute manifest stats
    categories = sorted(list(set(p.category for p in products)))
    avg_price = round(float(sum(p.price for p in products) / len(products)), 2)

    manifest = DatasetManifest(
        total_items=len(products),
        categories=categories,
        avg_price=avg_price,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2)

    logger.info("Saved %d products to %s and %s", len(products), json_path, csv_path)
    logger.info("Dataset Manifest: %s", manifest.model_dump_json())
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Ingest and prepare semantic search dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Target directory for processed dataset files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    products = load_and_validate_catalog()
    manifest = export_dataset(products, out_dir)
    print(f"\n[DONE] Ingestion Complete! {manifest.total_items} items ready in {out_dir}")


if __name__ == "__main__":
    main()
