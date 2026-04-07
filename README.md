# Aadhaar OCR Processor (Python + Streamlit)

Production-style Aadhaar document processing tool with a modern web UI.

It accepts Aadhaar images and PDFs, runs OCR + computer-vision cleanup, extracts `Name` and `Aadhaar Number`, and generates standardized PDF output:

`Name_AadhaarNumber.pdf`

---

## Features

- Supports `.jpg`, `.jpeg`, `.png`, `.pdf` (multi-page PDF supported)
- Computer-vision preprocessing:
  - document contour detection and perspective correction
  - denoise, contrast enhancement, thresholding
  - orientation variants for better OCR
- OCR engine strategy:
  - primary: `pytesseract`
  - fallback: `rapidocr-onnxruntime`
- PDF conversion strategy:
  - primary: `pdf2image` (Poppler)
  - fallback: `pypdfium2`
- Robust extraction:
  - Aadhaar regex validation for strict 12 digits
  - name extraction heuristics with keyword filtering
- Web UI with batch sessions:
  - multiple tabs/sessions
  - process batch or single file
  - skip already processed files in same tab
  - progress indicators and per-file status
- Downloads:
  - individual processed PDFs
  - all processed PDFs as ZIP
  - failed files ZIP
  - CSV log per tab/session

---

## Project Structure

- `api/index.py` - Vercel serverless API entrypoint (`app = FastAPI(...)`)
- `aadhaar_processor.py` - core OCR, CV preprocessing, text extraction, batch CLI
- `web_app.py` - Streamlit UI (tab-based sessions, uploads, downloads)
- `requirements.txt` - Python dependencies
- `vercel.json` - Vercel build and routing config

---

## Requirements

- Python `3.9+`
- pip

Optional but recommended:
- Tesseract OCR (for best OCR quality)
- Poppler (for `pdf2image` path-based PDF rendering)

The app includes fallback paths when these are missing.

---

## Installation

```bash
pip install -r requirements.txt --upgrade
```

---

## Run Web App (Recommended)

```bash
streamlit run web_app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

### Web Workflow

1. Create/select a batch tab
2. Upload Aadhaar files (image/PDF)
3. Click:
   - `Process this batch`, or
   - `Process selected file`
4. Download outputs and logs

---

## Deploy on Vercel (API)

This repository now includes a Vercel-compatible Python entrypoint at `api/index.py`.

### Endpoints

- `GET /` - basic status
- `GET /health` - health check
- `POST /process` - process uploads and return JSON results
- `POST /process/download` - process uploads and return ZIP of output PDFs

### Deploy Steps

1. Push this repo to GitHub.
2. In Vercel, create a new project from the repo.
3. Keep framework as `Other`.
4. Deploy (Vercel detects `vercel.json` and uses `api/index.py`).

### Local API run (optional)

```bash
uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
```

### Quick API test

```bash
curl -X POST "http://127.0.0.1:8000/process" \
  -F "files=@/absolute/path/to/aadhaar.jpg"
```

---

## Run CLI Mode

```bash
python aadhaar_processor.py --input "D:/path/to/folder"
```

Optional flags:

```bash
python aadhaar_processor.py \
  --input "D:/path/to/folder" \
  --log "processing_log.csv" \
  --failed-dir "failed" \
  --tesseract-cmd "C:/Program Files/Tesseract-OCR/tesseract.exe" \
  --poppler-path "C:/poppler/Library/bin"
```

---

## Output Format

- Final successful file format is always PDF:
  - `Name_AadhaarNumber.pdf`
- Failed files are marked failed (and can be downloaded from UI)
- CSV log columns:
  - `original_filename`
  - `extracted_name`
  - `aadhaar_number`
  - `status`
  - `renamed_filename`

---

## Troubleshooting

- If OCR quality is poor:
  - use clearer scans/photos
  - avoid glare and blur
  - ensure full card is visible
- If Tesseract not found:
  - install Tesseract or use fallback OCR dependency
- If Poppler not found:
  - install Poppler or rely on PDFium fallback (`pypdfium2`)

---

## Security and Compliance Note

This tool processes sensitive identity documents. Use responsibly and ensure compliance with local privacy and data-protection requirements. Avoid uploading real Aadhaar data to public/demo environments.
