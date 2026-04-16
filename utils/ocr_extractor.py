"""
OCR Text Extraction Module using Tesseract.

This module extracts text from images using Tesseract OCR.
"""

import re

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


# If Tesseract is not in PATH, specify the path here.
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _prepare_image_for_ocr(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)

    width, height = image.size
    if min(width, height) < 64:
        scale = 64 / max(min(width, height), 1)
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

    return image



def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text from an image using Tesseract OCR.
    """
    try:
        processed = _prepare_image_for_ocr(image)
        text = pytesseract.image_to_string(processed, config="--psm 6 --oem 3")
        return clean_text(text)
    except Exception as e:
        print(f"Warning: OCR error: {e}")
        return ""



def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()



def extract_text_batch(images: list) -> list:
    texts = []
    for i, image in enumerate(images):
        print(f"  Extracting text from image {i + 1}/{len(images)}...")
        texts.append(extract_text_from_image(image))

    return texts
