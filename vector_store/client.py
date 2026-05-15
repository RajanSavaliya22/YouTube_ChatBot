"""
Stage 5a: Qdrant Client + Collection Setup
============================================
Manages the Qdrant connection and creates the collection with
production-grade HNSW and quantization settings.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationConfig,
    PayloadSchemaType,
)

from config import QDRANT, COLLECTION
from utils.logger import get_logger

logger = get_logger("vector_store.client")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a singleton Qdrant client (gRPC preferred for speed)."""
    global _client
    if _client is not None:
        return _client

    
    _client = QdrantClient(
        url=QDRANT.url,
        api_key=QDRANT.api_key,
        cloud_inference=True,
        timeout=60
    )
    logger.info(f"Qdrant client connected → {QDRANT.url}")

    return _client


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create the Qdrant collection with production settings:
      - HNSW index (fast ANN search)
      - INT8 scalar quantization (4x RAM reduction, <5% accuracy loss)
      - On-disk payload storage (saves RAM for large collections)
      - Payload indexes on filterable fields

    Args:
        client:   Active QdrantClient
        recreate: If True, drop and recreate existing collection
    """
    exists = client.collection_exists(COLLECTION.name)

    if exists and not recreate:
        logger.info(f"Collection '{COLLECTION.name}' already exists — skipping creation.")
        return

    if exists and recreate:
        client.delete_collection(COLLECTION.name)
        logger.warning(f"Dropped existing collection '{COLLECTION.name}'.")

    # Scalar quantization: compress float32 → int8
    # quant_config = None
    # if COLLECTION.quantization:
    #     quant_config = QuantizationConfig(
    #         scalar=ScalarQuantizationConfig(
    #             type=ScalarType.INT8,
    #             quantile=0.99,      # Clip top/bottom 1% of values before quantizing
    #             always_ram=True,    # Keep quantized vectors in RAM for fast search
    #         )
    #     )

    client.create_collection(
        collection_name=COLLECTION.name,
        vectors_config=VectorParams(
            size=COLLECTION.vector_size,
            distance=Distance.COSINE,
            on_disk=False,           # Full-precision vectors stay in RAM
        ),
        hnsw_config=HnswConfigDiff(
            m=COLLECTION.hnsw_m,                     # Edges per node (16 = good default)
            ef_construct=COLLECTION.hnsw_ef_construct, # Build accuracy (100 = good default)
            full_scan_threshold=10_000,              # Below this, brute-force beats HNSW
            on_disk=False,
        ),
        optimizers_config=OptimizersConfigDiff(
            memmap_threshold=20_000,  # Segments larger than this use mmap (disk-backed)
            indexing_threshold=10_000,
        ),
        # quantization_config=quant_config,
        on_disk_payload=COLLECTION.on_disk_payload,
    )

    _create_payload_indexes(client)
    logger.info(f"Collection '{COLLECTION.name}' created successfully.")


def _create_payload_indexes(client: QdrantClient) -> None:
    """
    Create indexes on payload fields used for filtering.
    Without these, every filter requires a full scan — O(n) not O(1).
    """
    indexed_fields = {
        "video_id":       PayloadSchemaType.KEYWORD,
        "channel":        PayloadSchemaType.KEYWORD,
        "language":       PayloadSchemaType.KEYWORD,
        "published_date": PayloadSchemaType.KEYWORD,
        "timestamp_start": PayloadSchemaType.FLOAT,
    }

    for field_name, schema_type in indexed_fields.items():
        client.create_payload_index(
            collection_name=COLLECTION.name,
            field_name=field_name,
            field_schema=schema_type,
        )

    logger.info(f"Payload indexes created on: {list(indexed_fields.keys())}")


def get_collection_stats(client: QdrantClient) -> dict:
    """Return key collection metrics for monitoring."""
    info = client.get_collection(COLLECTION.name)
    return {
        "total_vectors":   info.points_count,
        "indexed_vectors": info.indexed_vectors_count,
        "status":          str(info.status),
        "segments":        info.segments_count,
    }
