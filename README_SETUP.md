# Multi-Modal Screenshot Search - Setup Guide

## Quick Start

This project uses:
- DINOv2 for visual embeddings
- Nomic for text embeddings
- Tesseract OCR for text extraction
- Qdrant for vector search

## Prerequisites

### 1. Python
- Python 3.8+
- pip

### 2. Tesseract OCR

#### Windows
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to the default location: `C:\Program Files\Tesseract-OCR`
3. Add Tesseract to PATH, or set this in `utils/ocr_extractor.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

#### Linux
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS
```bash
brew install tesseract
```

### 3. Qdrant Cloud
- Create an account at https://cloud.qdrant.io
- Put your cluster URL and API key in `qdrant_client_helper.py`

## Install

```bash
pip install -r requirements.txt
```

Verify Tesseract:

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

## Run

Build the multimodal index:

```bash
python build_multimodal_index.py
```

Launch the app:

```bash
streamlit run app_multimodal.py
```

## Deploy On Render

This repo now includes:
- `render.yaml` for the Render service configuration
- `Dockerfile` so Render can install the OS-level `tesseract-ocr` package
- `.env.example` showing the required environment variables

Before deploying:
1. Rotate your current Qdrant API key because it was previously stored in source code.
2. Make sure your Qdrant collection already contains indexed data, or run `build_multimodal_index.py` after setting the environment variables.
3. Push this repository to GitHub.

In Render:
1. Create a new `Blueprint` deployment from your GitHub repo, or create a Docker web service manually.
2. Set these environment variables:
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
3. Deploy the service.

## Notes

- OCR is handled by `utils/ocr_extractor.py`
- Visual embeddings come from `models/dinov2_model.py`
- Text embeddings come from `models/nomic_model.py`
- Query and document OCR text are stored in Qdrant payloads

## Troubleshooting

### Tesseract not found
Install Tesseract and ensure `tesseract.exe` is on PATH or configured in `utils/ocr_extractor.py`.

### No images found
Put screenshots in `dataset/images/`.

### Qdrant connection issues
Check your credentials in `qdrant_client_helper.py`.
