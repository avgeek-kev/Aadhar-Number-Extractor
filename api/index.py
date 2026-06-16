from __future__ import annotations

import io
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import pytesseract

from aadhaar_processor import (
    ALL_SUPPORTED_EXTENSIONS,
    collect_text_from_file,
    extract_from_text,
    sanitize_filename_part,
)


app = FastAPI(title="Aadhaar OCR API", version="1.0.0")


def image_bytes_to_pdf(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        out = io.BytesIO()
        rgb.save(out, format="PDF")
        return out.getvalue()


def source_to_pdf_bytes(filename: str, data: bytes) -> bytes:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        return image_bytes_to_pdf(data)
    return data


def dedupe_filename(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "aadhaar-ocr-api"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/process")
async def process_files(
    files: list[UploadFile] = File(...),
    tesseract_cmd: str = Form(default=""),
    poppler_path: str = Form(default=""),
) -> JSONResponse:
    if tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip()

    rows: list[dict[str, Any]] = []
    used_output_names: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="aadhaar_api_") as temp_dir:
        tmp_root = Path(temp_dir)

        for item in files:
            original_name = item.filename or f"upload_{uuid.uuid4().hex}.bin"
            ext = Path(original_name).suffix.lower()
            data = await item.read()

            if ext not in ALL_SUPPORTED_EXTENSIONS:
                rows.append(
                    {
                        "original_filename": original_name,
                        "extracted_name": "",
                        "aadhaar_number": "",
                        "extracted_address": "",
                        "status": "failure: unsupported file type",
                        "renamed_filename": "",
                    }
                )
                continue

            tmp_name = f"{uuid.uuid4().hex}_{Path(original_name).name}"
            tmp_path = tmp_root / tmp_name
            tmp_path.write_bytes(data)

            try:
                text = collect_text_from_file(tmp_path, poppler_path=poppler_path.strip() or None)
                if not text.strip():
                    rows.append(
                        {
                            "original_filename": original_name,
                            "extracted_name": "",
                            "aadhaar_number": "",
                            "extracted_address": "",
                            "status": "failure: OCR returned empty text",
                            "renamed_filename": "",
                        }
                    )
                    continue

                extracted = extract_from_text(text)
                if not extracted.name or not extracted.aadhaar_number:
                    rows.append(
                        {
                            "original_filename": original_name,
                            "extracted_name": extracted.name or "",
                            "aadhaar_number": extracted.aadhaar_number or "",
                            "extracted_address": extracted.address or "",
                            "status": f"failure: {extracted.reason or 'extraction failed'}",
                            "renamed_filename": "",
                        }
                    )
                    continue

                clean_name = sanitize_filename_part(extracted.name)
                renamed_pdf = dedupe_filename(f"{clean_name}_{extracted.aadhaar_number}.pdf", used_output_names)
                rows.append(
                    {
                        "original_filename": original_name,
                        "extracted_name": extracted.name,
                        "aadhaar_number": extracted.aadhaar_number,
                        "extracted_address": extracted.address or "",
                        "status": "success",
                        "renamed_filename": renamed_pdf,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "original_filename": original_name,
                        "extracted_name": "",
                        "aadhaar_number": "",
                        "extracted_address": "",
                        "status": f"failure: {type(exc).__name__}: {exc}",
                        "renamed_filename": "",
                    }
                )

    return JSONResponse({"results": rows})


@app.post("/process/download")
async def process_and_download(
    files: list[UploadFile] = File(...),
    tesseract_cmd: str = Form(default=""),
    poppler_path: str = Form(default=""),
) -> StreamingResponse:
    if tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip()

    used_output_names: set[str] = set()
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        with tempfile.TemporaryDirectory(prefix="aadhaar_api_") as temp_dir:
            tmp_root = Path(temp_dir)

            for item in files:
                original_name = item.filename or f"upload_{uuid.uuid4().hex}.bin"
                ext = Path(original_name).suffix.lower()
                data = await item.read()

                if ext not in ALL_SUPPORTED_EXTENSIONS:
                    zf.writestr(f"failed/{original_name}", data)
                    continue

                tmp_name = f"{uuid.uuid4().hex}_{Path(original_name).name}"
                tmp_path = tmp_root / tmp_name
                tmp_path.write_bytes(data)

                try:
                    text = collect_text_from_file(tmp_path, poppler_path=poppler_path.strip() or None)
                    extracted = extract_from_text(text)
                    if not extracted.name or not extracted.aadhaar_number:
                        zf.writestr(f"failed/{original_name}", data)
                        continue

                    clean_name = sanitize_filename_part(extracted.name)
                    renamed_pdf = dedupe_filename(f"{clean_name}_{extracted.aadhaar_number}.pdf", used_output_names)
                    pdf_bytes = source_to_pdf_bytes(original_name, data)
                    zf.writestr(renamed_pdf, pdf_bytes)
                except Exception:
                    zf.writestr(f"failed/{original_name}", data)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=processed_aadhaar_files.zip"},
    )
