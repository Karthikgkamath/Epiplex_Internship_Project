"""
DINOv2 Visual Embedding Model

This module uses Meta's DINOv2 for generating visual embeddings from images.
DINOv2 is a self-supervised vision transformer that produces high-quality visual features.
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINOv2 model
# Using dinov2-base which produces 768-dimensional embeddings
MODEL_NAME = "facebook/dinov2-base"

print(f"🔧 Loading DINOv2 model: {MODEL_NAME}")
print(f"🖥️  Device: {DEVICE}")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

print("✅ DINOv2 model loaded successfully")


def get_visual_embedding(image: Image.Image) -> np.ndarray:
    """
    Generate normalized DINOv2 embedding for an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Normalized embedding as numpy array (768-dim for dinov2-base)
    """
    try:
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
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
        print(f"⚠️ DINOv2 Embedding Error: {e}")
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
        print(f"  🖼️  Processing image {i+1}/{len(images)} with DINOv2...")
        embedding = get_visual_embedding(image)
        embeddings.append(embedding)
    
    return np.array(embeddings)

