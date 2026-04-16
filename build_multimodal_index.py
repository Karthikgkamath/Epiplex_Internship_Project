"""
Multi-Modal Index Builder

This script builds the Qdrant index with dual embeddings:
1. Visual embeddings using DINOv2
2. Text embeddings using Nomic (from OCR-extracted text)
"""

from pathlib import Path

from PIL import Image
from tqdm import tqdm

from qdrant_client.models import PointStruct
from qdrant_client_helper import get_client, recreate_dual_vector_collection, COLLECTION_NAME
from models.dinov2_model import get_visual_embedding, get_embedding_dimension as get_visual_dim
from models.nomic_model import get_document_embedding, get_embedding_dimension as get_text_dim
from utils.ocr_extractor import extract_text_from_image


def index_dataset(dataset_dir="dataset/images"):
    """
    Index all images in the dataset directory with dual embeddings.
    """
    print("=" * 80)
    print("MULTI-MODAL INDEX BUILDER")
    print("=" * 80)

    print("\nConnecting to Qdrant Cloud...")
    client = get_client()

    visual_dim = get_visual_dim()
    text_dim = get_text_dim()

    print(f"\nRecreating collection '{COLLECTION_NAME}'...")
    recreate_dual_vector_collection(client, visual_dim, text_dim)

    dataset_path = Path(dataset_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = [f for f in dataset_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"\nNo images found in {dataset_dir}")
        return

    print(f"\nFound {len(image_files)} images to index")
    print("=" * 80)

    points = []
    for idx, image_path in enumerate(tqdm(image_files, desc="Indexing images")):
        try:
            image = Image.open(image_path).convert("RGB")

            print(f"\nProcessing: {image_path.name}")
            print(f"   Size: {image.size[0]} x {image.size[1]} pixels")

            print("   Extracting text with Tesseract OCR...")
            extracted_text = extract_text_from_image(image)

            if extracted_text:
                if len(extracted_text) > 100:
                    print(f'   Extracted: "{extracted_text[:100]}..."')
                else:
                    print(f'   Extracted: "{extracted_text}"')
            else:
                print("   No text found in image")

            print("   Generating visual embedding with DINOv2...")
            visual_embedding = get_visual_embedding(image)

            print("   Generating text embedding with Nomic...")
            text_embedding = get_document_embedding(extracted_text)

            point = PointStruct(
                id=idx,
                vector={
                    "image_vector": visual_embedding.tolist(),
                    "text_vector": text_embedding.tolist()
                },
                payload={
                    "image_path": str(image_path),
                    "filename": image_path.name,
                    "extracted_text": extracted_text,
                    "image_width": image.size[0],
                    "image_height": image.size[1]
                }
            )

            points.append(point)
            print(f"   Point created (ID: {idx})")

        except Exception as e:
            print(f"   Error processing {image_path.name}: {e}")
            continue

    if points:
        print(f"\nUploading {len(points)} points to Qdrant...")
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Successfully indexed {len(points)} images")
    else:
        print("\nNo points to upload")

    print("=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    index_dataset()
