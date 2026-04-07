# Vercel Deployment Guide

## Why deployment failed before

Vercel Python requires a valid serverless entrypoint under `api/` that exposes an application object (`app`) for ASGI/WSGI.

Your project previously had only:
- `web_app.py` (Streamlit app)
- `aadhaar_processor.py` (CLI/core logic)

So Vercel could not find an API entrypoint and raised:
`Error: No python entrypoint found`

## What is fixed now

- Added `api/index.py` with `app = FastAPI(...)`
- Added `vercel.json` mapping all routes to `api/index.py`
- Added API dependencies to `requirements.txt`

## Files used by Vercel

- `api/index.py`
- `vercel.json`
- `requirements.txt`

## Deploy steps

1. Commit and push to GitHub.
2. Import repo in Vercel.
3. Use default project settings (`Other` framework is fine).
4. Deploy.

## Endpoints after deploy

- `GET /`
- `GET /health`
- `POST /process`
- `POST /process/download`

## cURL examples

### 1) JSON extraction response

```bash
curl -X POST "https://<your-vercel-domain>/process" \
  -F "files=@/absolute/path/to/aadhaar1.jpg" \
  -F "files=@/absolute/path/to/aadhaar2.pdf"
```

### 2) Download output ZIP

```bash
curl -X POST "https://<your-vercel-domain>/process/download" \
  -F "files=@/absolute/path/to/aadhaar1.jpg" \
  -o processed_aadhaar_files.zip
```

## Notes

- OCR quality is best when Tesseract is available, but fallback OCR is included.
- If PDF conversion via Poppler is unavailable, PDFium fallback is used.
- Streamlit UI (`web_app.py`) is for local use; Vercel deploy target is the FastAPI backend.
