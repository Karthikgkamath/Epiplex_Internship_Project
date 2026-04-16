# Multi-Modal Dual-Vector Retrieval System

## Overview

This project implements screenshot search using:
- DINOv2 image embeddings
- OCR text extraction with Tesseract
- Nomic text embeddings
- Weighted score fusion in Qdrant

## Indexing Flow

For each dataset image:
1. Extract text using Tesseract OCR
2. Generate a visual embedding with DINOv2
3. Generate a text embedding with Nomic
4. Store both vectors and metadata in Qdrant

## Query Flow

For each query image:
1. Extract text using Tesseract OCR
2. Generate the visual embedding
3. Generate the query text embedding
4. Search Qdrant with both vectors
5. Combine visual and text scores with configurable weights

## Models

### DINOv2
- Model: `facebook/dinov2-base`
- Output: 768-dim normalized image embedding

### Nomic
- Model: `nomic-ai/nomic-embed-text-v1`
- Output: 768-dim normalized text embedding

### Tesseract OCR
- Engine: Tesseract OCR
- Config: `--psm 6 --oem 3`
- Output: cleaned OCR text from screenshots

## Project Structure

- `utils/ocr_extractor.py`: Tesseract OCR helper
- `models/dinov2_model.py`: visual embeddings
- `models/nomic_model.py`: text embeddings
- `build_multimodal_index.py`: offline indexing
- `app_multimodal.py`: search UI
- `qdrant_client_helper.py`: Qdrant client and collection helpers

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Tesseract separately:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Usage

Build the index:

```bash
python build_multimodal_index.py
```

Run the app:

```bash
streamlit run app_multimodal.py
```
