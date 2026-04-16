import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Qdrant Cloud configuration is provided via environment variables.
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

# Collection name for multi-modal search (DINOv2 + Nomic)
COLLECTION_NAME = "erp_multimodal"


def get_client():
    """
    Get a configured Qdrant client instance.

    Returns:
        QdrantClient instance connected to Qdrant Cloud
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError(
            "Missing Qdrant configuration. Set QDRANT_URL and QDRANT_API_KEY "
            "before starting the app."
        )

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120  # Allow slower startup/network conditions in hosted environments.
    )


def recreate_dual_vector_collection(
    client,
    image_vector_size=768,  # DINOv2 dimension
    text_vector_size=768,   # Nomic dimension
    collection_name=COLLECTION_NAME
):
    """
    Recreate a collection in Qdrant with dual vector support.

    This creates a collection that stores both:
    - image_vector: DINOv2 visual embeddings (768-dim)
    - text_vector: Nomic text embeddings (768-dim)

    Args:
        client: Qdrant client instance
        image_vector_size: Dimension of image vectors (default: 768 for DINOv2)
        text_vector_size: Dimension of text vectors (default: 768 for Nomic)
        collection_name: Name of collection to recreate
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "image_vector": VectorParams(
                size=image_vector_size,
                distance=Distance.COSINE
            ),
            "text_vector": VectorParams(
                size=text_vector_size,
                distance=Distance.COSINE
            )
        }
    )
    print(f"Created collection '{collection_name}' with dual vectors:")
    print(f"  - image_vector: {image_vector_size}-dim (DINOv2)")
    print(f"  - text_vector: {text_vector_size}-dim (Nomic)")
