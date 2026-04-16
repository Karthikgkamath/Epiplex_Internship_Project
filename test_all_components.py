"""
Comprehensive test to verify all components of the multi-modal system.
"""

import sys

import numpy as np
from PIL import Image, ImageDraw

print("=" * 80)
print("MULTI-MODAL SYSTEM - COMPONENT TEST")
print("=" * 80)
print()

print("1. Testing Tesseract OCR...")
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"   Tesseract OCR v{version} - OK")
except Exception as e:
    print(f"   Tesseract OCR - FAILED: {e}")
    sys.exit(1)

print("\n2. Testing DINOv2 model...")
try:
    from models.dinov2_model import get_visual_embedding, get_embedding_dimension

    test_img = Image.new("RGB", (224, 224), color="blue")
    embedding = get_visual_embedding(test_img)
    dim = get_embedding_dimension()

    assert embedding.shape[0] == 768, f"Expected 768-dim, got {embedding.shape[0]}"
    assert dim == 768, f"Expected dimension 768, got {dim}"
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 0.01, "Embedding not normalized"

    print("   DINOv2 (768-dim, normalized) - OK")
except Exception as e:
    print(f"   DINOv2 - FAILED: {e}")
    sys.exit(1)

print("\n3. Testing Nomic text embedding model...")
try:
    from models.nomic_model import get_text_embedding, get_document_embedding, get_embedding_dimension

    test_text = "Login Dashboard Settings"
    query_embedding = get_text_embedding(test_text)
    doc_embedding = get_document_embedding(test_text)
    dim = get_embedding_dimension()

    assert query_embedding.shape[0] == 768, f"Expected 768-dim, got {query_embedding.shape[0]}"
    assert doc_embedding.shape[0] == 768, f"Expected 768-dim, got {doc_embedding.shape[0]}"
    assert dim == 768, f"Expected dimension 768, got {dim}"
    assert np.abs(np.linalg.norm(query_embedding) - 1.0) < 0.01, "Query embedding not normalized"
    assert np.abs(np.linalg.norm(doc_embedding) - 1.0) < 0.01, "Doc embedding not normalized"

    print("   Nomic (768-dim, normalized) - OK")
except Exception as e:
    print(f"   Nomic - FAILED: {e}")
    sys.exit(1)

print("\n4. Testing OCR text extraction...")
try:
    from utils.ocr_extractor import extract_text_from_image, clean_text

    test_img = Image.new("RGB", (300, 100), color="white")
    extracted = extract_text_from_image(test_img)
    assert isinstance(extracted, str)

    dirty_text = "  Hello    World!!!  "
    cleaned = clean_text(dirty_text)
    assert "Hello" in cleaned and "World" in cleaned, f"Text cleaning failed: '{cleaned}'"

    print("   OCR extractor - OK")
except Exception as e:
    print(f"   OCR extractor - FAILED: {e}")
    sys.exit(1)

print("\n5. Testing Qdrant Cloud connection...")
try:
    from qdrant_client_helper import get_client, COLLECTION_NAME

    client = get_client()
    collections = client.get_collections()

    print("   Qdrant Cloud connection - OK")
    print(f"   Target collection: {COLLECTION_NAME}")

    collection_names = [c.name for c in collections.collections]
    if COLLECTION_NAME in collection_names:
        print(f"   Collection '{COLLECTION_NAME}' already exists")
        info = client.get_collection(COLLECTION_NAME)
        print(f"   Points in collection: {info.points_count}")
    else:
        print(f"   Collection '{COLLECTION_NAME}' not yet created")
except Exception as e:
    print(f"   Qdrant Cloud - FAILED: {e}")
    print("   Check your credentials in qdrant_client_helper.py")
    sys.exit(1)

print("\n6. Testing end-to-end pipeline...")
try:
    from utils.ocr_extractor import extract_text_from_image
    from models.dinov2_model import get_visual_embedding
    from models.nomic_model import get_document_embedding

    test_img = Image.new("RGB", (400, 300), color="lightblue")
    draw = ImageDraw.Draw(test_img)
    draw.text((50, 50), "Test Screenshot", fill="black")

    text = extract_text_from_image(test_img)
    visual_emb = get_visual_embedding(test_img)
    text_emb = get_document_embedding(text if text else "")

    assert visual_emb.shape[0] == 768
    assert text_emb.shape[0] == 768

    print("   End-to-end pipeline - OK")
    print(f"   Extracted text: '{text[:50]}'" if text else "   Extracted text: ''")
    print(f"   Visual embedding: {visual_emb.shape}")
    print(f"   Text embedding: {text_emb.shape}")
except Exception as e:
    print(f"   End-to-end pipeline - FAILED: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("ALL TESTS PASSED! System is ready for indexing and search!")
print("=" * 80)
print()
print("Next steps:")
print("   1. Run: python build_multimodal_index.py")
print("   2. Run: streamlit run app_multimodal.py")
