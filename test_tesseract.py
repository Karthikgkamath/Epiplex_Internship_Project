"""
Quick test to check if Tesseract OCR is installed and accessible.
"""

import sys

try:
    import pytesseract
    print("pytesseract package imported successfully")
except ImportError as e:
    print(f"Failed to import pytesseract: {e}")
    sys.exit(1)

try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract OCR found. Version: {version}")
except Exception as e:
    print(f"Tesseract OCR binary not found: {e}")
    print("Install Tesseract and make sure tesseract.exe is on PATH or configured in utils/ocr_extractor.py")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw
    from utils.ocr_extractor import extract_text_from_image

    test_img = Image.new("RGB", (1200, 300), color="white")
    draw = ImageDraw.Draw(test_img)
    draw.text((40, 100), "Login Dashboard Settings", fill="black")

    result = extract_text_from_image(test_img)
    print(f"OCR test result: '{result}'")
except Exception as e:
    print(f"OCR test failed: {e}")
    sys.exit(1)

print("Tesseract OCR is ready to use")
