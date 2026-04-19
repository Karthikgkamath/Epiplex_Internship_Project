"""
Nomic Text Embedding Model

This module uses Nomic's embedding model for generating text embeddings.
The model is loaded lazily so the web app can start before large ML weights are
initialized.
"""

from functools import lru_cache

import numpy as np

# Using nomic-embed-text-v1 for text embeddings
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"


@lru_cache(maxsize=1)
def get_model():
    """
    Load the Nomic embedding model once, on first use.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Nomic text embedding model: {MODEL_NAME}")
    print(f"Device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=device,
    )

    print("Nomic model loaded successfully")
    return model


def get_text_embedding(text: str) -> np.ndarray:
    """
    Generate normalized Nomic embedding for text.

    Args:
        text: Input text string

    Returns:
        Normalized embedding as numpy array (768-dim for nomic-embed-text-v1)
    """
    try:
        # Handle empty text
        if not text or text.strip() == "":
            print("Empty text detected, returning zero vector")
            return np.zeros(768, dtype=np.float32)

        model = get_model()

        # Add task prefix for better retrieval performance.
        # Nomic recommends this for search tasks.
        prefixed_text = f"search_query: {text}"

        # Generate embedding
        embedding = model.encode(
            prefixed_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding

    except Exception as e:
        print(f"Nomic Embedding Error: {e}")
        # Return zero vector on error
        return np.zeros(768, dtype=np.float32)


def get_embedding_dimension() -> int:
    """
    Get the dimension of Nomic embeddings.

    Returns:
        Embedding dimension (768 for nomic-embed-text-v1)
    """
    return 768


def get_batch_embeddings(texts: list) -> np.ndarray:
    """
    Generate embeddings for multiple texts in batch.

    Args:
        texts: List of text strings

    Returns:
        Array of embeddings with shape (num_texts, 768)
    """
    embeddings = []

    for i, text in enumerate(texts):
        print(f"Processing text {i + 1}/{len(texts)} with Nomic...")
        embedding = get_text_embedding(text)
        embeddings.append(embedding)

    return np.array(embeddings)


def get_document_embedding(text: str) -> np.ndarray:
    """
    Generate embedding for a document (indexed content).
    Uses different prefix than queries for asymmetric search.

    Args:
        text: Document text

    Returns:
        Normalized embedding
    """
    try:
        if not text or text.strip() == "":
            return np.zeros(768, dtype=np.float32)

        model = get_model()

        # Use document prefix for indexing
        prefixed_text = f"search_document: {text}"

        embedding = model.encode(
            prefixed_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding

    except Exception as e:
        print(f"Nomic Document Embedding Error: {e}")
        return np.zeros(768, dtype=np.float32)
