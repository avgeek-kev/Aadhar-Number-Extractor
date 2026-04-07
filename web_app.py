#!/usr/bin/env python3
"""
Streamlit UI for Aadhaar document processing.
"""

from __future__ import annotations

import csv
import hashlib
import io
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image
import pytesseract
import streamlit as st

from aadhaar_processor import (
    ALL_SUPPORTED_EXTENSIONS,
    collect_text_from_file,
    extract_from_text,
    sanitize_filename_part,
)


def build_zip_blob(file_map: List[Tuple[str, bytes]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in file_map:
            zf.writestr(name, content)
    mem.seek(0)
    return mem.getvalue()


def make_csv_blob(rows: List[Dict[str, str]]) -> bytes:
    mem = io.StringIO()
    writer = csv.writer(mem)
    writer.writerow(
        [
            "original_filename",
            "extracted_name",
            "aadhaar_number",
            "status",
            "renamed_filename",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.get("original_filename", ""),
                row.get("extracted_name", ""),
                row.get("aadhaar_number", ""),
                row.get("status", ""),
                row.get("renamed_filename", ""),
            ]
        )
    return mem.getvalue().encode("utf-8")


def image_bytes_to_pdf(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        output = io.BytesIO()
        rgb.save(output, format="PDF")
        return output.getvalue()


def source_to_pdf_bytes(filename: str, source_bytes: bytes) -> bytes:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        return image_bytes_to_pdf(source_bytes)
    # Input is already PDF; keep as PDF bytes.
    return source_bytes


def dedupe_filename(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    idx = 1
    while True:
        candidate = f"{stem}_{idx}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def process_uploaded_files(
    uploaded_files,
    tesseract_cmd: str,
    poppler_path: str,
    used_output_names: set[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Tuple[List[Dict[str, str]], List[Tuple[str, bytes]], List[Tuple[str, bytes]]]:
    rows: List[Dict[str, str]] = []
    success_files_pdf: List[Tuple[str, bytes]] = []
    failed_files: List[Tuple[str, bytes]] = []
    output_names = used_output_names if used_output_names is not None else set()

    if tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip()

    with tempfile.TemporaryDirectory(prefix="aadhaar_ui_") as temp_dir:
        tmp_root = Path(temp_dir)
        total = len(uploaded_files)

        for idx, uploaded in enumerate(uploaded_files, start=1):
            original_name = uploaded.name
            ext = Path(original_name).suffix.lower()
            file_bytes = uploaded.getvalue()
            if ext not in ALL_SUPPORTED_EXTENSIONS:
                rows.append(
                    {
                        "original_filename": original_name,
                        "extracted_name": "",
                        "aadhaar_number": "",
                        "status": "failure: unsupported file type",
                        "renamed_filename": "",
                    }
                )
                failed_files.append((original_name, file_bytes))
                if progress_callback:
                    progress_callback(idx, total)
                continue

            safe_tmp_name = f"{uuid.uuid4().hex}_{Path(original_name).name}"
            tmp_input = tmp_root / safe_tmp_name
            tmp_input.write_bytes(file_bytes)

            try:
                text = collect_text_from_file(
                    tmp_input,
                    poppler_path=poppler_path.strip() or None,
                )
                if not text.strip():
                    rows.append(
                        {
                            "original_filename": original_name,
                            "extracted_name": "",
                            "aadhaar_number": "",
                            "status": "failure: OCR returned empty text",
                            "renamed_filename": "",
                        }
                    )
                    failed_files.append((original_name, file_bytes))
                    if progress_callback:
                        progress_callback(idx, total)
                    continue

                extracted = extract_from_text(text)
                if not extracted.name or not extracted.aadhaar_number:
                    rows.append(
                        {
                            "original_filename": original_name,
                            "extracted_name": extracted.name or "",
                            "aadhaar_number": extracted.aadhaar_number or "",
                            "status": f"failure: {extracted.reason or 'extraction failed'}",
                            "renamed_filename": "",
                        }
                    )
                    failed_files.append((original_name, file_bytes))
                    if progress_callback:
                        progress_callback(idx, total)
                    continue

                clean_name = sanitize_filename_part(extracted.name)
                renamed_pdf = f"{clean_name}_{extracted.aadhaar_number}.pdf"
                renamed_pdf = dedupe_filename(renamed_pdf, output_names)
                pdf_bytes = source_to_pdf_bytes(original_name, file_bytes)
                success_files_pdf.append((renamed_pdf, pdf_bytes))

                rows.append(
                    {
                        "original_filename": original_name,
                        "extracted_name": extracted.name,
                        "aadhaar_number": extracted.aadhaar_number,
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
                        "status": f"failure: {type(exc).__name__}: {exc}",
                        "renamed_filename": "",
                    }
                )
                failed_files.append((original_name, file_bytes))
            finally:
                if progress_callback:
                    progress_callback(idx, total)

    return rows, success_files_pdf, failed_files


def init_session_state() -> None:
    if "aadhaar_sessions" not in st.session_state:
        st.session_state.aadhaar_sessions = []
    if "aadhaar_session_counter" not in st.session_state:
        st.session_state.aadhaar_session_counter = 0


def new_session() -> None:
    st.session_state.aadhaar_session_counter += 1
    n = st.session_state.aadhaar_session_counter
    st.session_state.aadhaar_sessions.append(
        {
            "id": f"session_{uuid.uuid4().hex[:8]}",
            "name": f"Batch {n}",
            "rows": [],
            "success_files_pdf": [],
            "failed_files": [],
            "csv_log": b"",
            "processed_signatures": [],
        }
    )


def status_with_icon(status: str) -> str:
    return f"✅ {status}" if status == "success" else f"❌ {status}"


def process_session_files(session: Dict[str, object], tesseract_cmd: str, poppler_path: str, files) -> None:
    progress = st.progress(0, text="Starting batch processing...")

    def update_progress(current: int, total: int) -> None:
        pct = int((current / max(total, 1)) * 100)
        progress.progress(pct, text=f"Processing {current}/{total} files...")

    existing_rows = list(session.get("rows", []))
    existing_success = list(session.get("success_files_pdf", []))
    existing_failed = list(session.get("failed_files", []))
    processed_signatures = set(session.get("processed_signatures", []))
    used_output_names = {name for name, _ in existing_success}

    incoming_files = list(files)
    pending_files = []
    skipped = 0
    for f in incoming_files:
        signature = hashlib.sha1(f.getvalue()).hexdigest()
        if signature in processed_signatures:
            skipped += 1
            continue
        pending_files.append(f)
        processed_signatures.add(signature)

    if not pending_files:
        progress.progress(100, text="No new files to process in this tab.")
        if skipped:
            st.warning(f"Skipped {skipped} already-processed file(s) in this tab.")
        return

    rows, success_files_pdf, failed_files = process_uploaded_files(
        uploaded_files=pending_files,
        tesseract_cmd=tesseract_cmd,
        poppler_path=poppler_path,
        used_output_names=used_output_names,
        progress_callback=update_progress,
    )
    progress.progress(100, text="Processing completed.")
    if skipped:
        st.warning(f"Skipped {skipped} already-processed file(s) in this tab.")

    for row in rows:
        row["status"] = status_with_icon(row["status"])

    merged_rows = existing_rows + rows
    merged_success = existing_success + success_files_pdf
    merged_failed = existing_failed + failed_files

    session["rows"] = merged_rows
    session["success_files_pdf"] = merged_success
    session["failed_files"] = merged_failed
    session["processed_signatures"] = list(processed_signatures)
    session["csv_log"] = make_csv_blob(merged_rows)


def render_session_tab(session: Dict[str, object], tesseract_cmd: str, poppler_path: str) -> None:
    sid = session["id"]
    st.markdown("### Upload and Process")
    files = st.file_uploader(
        "Upload files for this batch",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        key=f"uploader_{sid}",
    )
    if files:
        st.info(f"{len(files)} files queued.")

    process_col1, process_col2 = st.columns(2)
    with process_col1:
        if st.button("Process this batch", type="primary", use_container_width=True, key=f"process_{sid}", disabled=not files):
            process_session_files(session, tesseract_cmd, poppler_path, files)

    with process_col2:
        selected_name = None
        selected_file = None
        if files:
            file_names = [f.name for f in files]
            selected_name = st.selectbox(
                "Process individual file",
                options=file_names,
                key=f"single_select_{sid}",
            )
            for f in files:
                if f.name == selected_name:
                    selected_file = f
                    break
        if st.button(
            "Process selected file",
            use_container_width=True,
            key=f"process_one_{sid}",
            disabled=selected_file is None,
        ):
            process_session_files(session, tesseract_cmd, poppler_path, [selected_file])

    rows = session.get("rows", [])
    success_files_pdf = session.get("success_files_pdf", [])
    failed_files = session.get("failed_files", [])
    csv_log = session.get("csv_log", b"")

    total = len(rows)
    success_count = sum(1 for r in rows if str(r.get("status", "")).startswith("✅"))
    failed_count = total - success_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Success", success_count)
    c3.metric("Failed", failed_count)

    st.markdown("### Results")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if success_files_pdf:
        st.markdown("### Individual Downloads")
        for i, (fname, data) in enumerate(success_files_pdf):
            st.download_button(
                label=f"Download {fname}",
                data=data,
                file_name=fname,
                mime="application/pdf",
                key=f"dl_{sid}_{i}",
                use_container_width=True,
            )

    all_success_zip = build_zip_blob(success_files_pdf) if success_files_pdf else b""
    all_failed_zip = build_zip_blob(failed_files) if failed_files else b""

    d1, d2, d3 = st.columns(3)
    with d1:
        if all_success_zip:
            st.download_button(
                "Download All Processed PDFs (ZIP)",
                data=all_success_zip,
                file_name=f"{session['name']}_processed_pdfs.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_success_{sid}",
            )
    with d2:
        if all_failed_zip:
            st.download_button(
                "Download Failed Files (ZIP)",
                data=all_failed_zip,
                file_name=f"{session['name']}_failed_files.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_failed_{sid}",
            )
    with d3:
        if csv_log:
            st.download_button(
                "Download CSV Log",
                data=csv_log,
                file_name=f"{session['name']}_processing_log.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"csv_{sid}",
            )


def render_app() -> None:
    st.set_page_config(
        page_title="Aadhaar OCR Renamer",
        page_icon="🪪",
        layout="wide",
    )

    st.title("Aadhaar Document Processor")
    st.caption(
        "Upload Aadhaar files, extract Name + Aadhaar, and generate final PDFs."
    )
    st.caption(
        "Output format is always PDF: Name_AadhaarNumber.pdf"
    )

    st.markdown(
        """
        <style>
        .session-card {
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 10px 14px;
            margin-bottom: 10px;
            background-color: rgba(17, 24, 39, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            tesseract_cmd = st.text_input(
                "Tesseract executable path (optional)",
                placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            )
        with col2:
            poppler_path = st.text_input(
                "Poppler bin path (optional, for PDF support)",
                placeholder=r"C:\poppler\Library\bin",
            )

    init_session_state()
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.markdown("<div class='session-card'><b>Batch Sessions</b>: create multiple independent processing tabs.</div>", unsafe_allow_html=True)
    with top_c2:
        if st.button("➕ New Batch Tab", use_container_width=True):
            new_session()

    if not st.session_state.aadhaar_sessions:
        new_session()

    tab_labels = [s["name"] for s in st.session_state.aadhaar_sessions]
    tabs = st.tabs(tab_labels)
    for idx, tab in enumerate(tabs):
        with tab:
            render_session_tab(st.session_state.aadhaar_sessions[idx], tesseract_cmd, poppler_path)


if __name__ == "__main__":
    render_app()
