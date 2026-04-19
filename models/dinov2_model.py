"""
DINOv2 Visual Embedding Model

This module uses Meta's DINOv2 for generating visual embeddings from images.
DINOv2 is loaded lazily so hosted Streamlit deployments can boot before the
large model weights are initialized.
"""

from functools import lru_cache

import numpy as np
from PIL import Image

# Using dinov2-base which produces 768-dimensional embeddings
MODEL_NAME = "facebook/dinov2-base"


@lru_cache(maxsize=1)
def get_model_components():
    """
    Load DINOv2 once, on first use.

    Keeping model loading lazy lets Streamlit start quickly in hosted
    environments, so health checks and /_stcore endpoints respond before the
    large ML weights are downloaded into memory.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading DINOv2 model: {MODEL_NAME}")
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("DINOv2 model loaded successfully")
    return processor, model, device, torch


def get_visual_embedding(image: Image.Image) -> np.ndarray:
    """
    Generate normalized DINOv2 embedding for an image.

    Args:
        image: PIL Image object

    Returns:
        Normalized embedding as numpy array (768-dim for dinov2-base)
    """
    try:
        processor, model, device, torch = get_model_components()

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            # Get model output
            outputs = model(**inputs)

            # Use [CLS] token embedding as the image representation
            # Shape: (batch_size, hidden_size)
            embedding = outputs.last_hidden_state[:, 0, :]

            # L2 normalization for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        # Convert to numpy and remove batch dimension
        return embedding.cpu().numpy()[0]

    except Exception as e:
        print(f"DINOv2 Embedding Error: {e}")
        # Return zero vector on error
        return np.zeros(768, dtype=np.float32)


def get_embedding_dimension() -> int:
    """
    Get the dimension of DINOv2 embeddings.

    Returns:
        Embedding dimension (768 for dinov2-base)
    """
    return 768


def get_batch_embeddings(images: list) -> np.ndarray:
    """
    Generate embeddings for multiple images in batch.

    Args:
        images: List of PIL Image objects

    Returns:
        Array of embeddings with shape (num_images, 768)
    """
    embeddings = []

    for i, image in enumerate(images):
        print(f"Processing image {i + 1}/{len(images)} with DINOv2...")
        embedding = get_visual_embedding(image)
        embeddings.append(embedding)

    return np.array(embeddings)
