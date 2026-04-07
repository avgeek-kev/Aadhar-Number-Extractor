#!/usr/bin/env python3
"""
Aadhaar document processor.

Features:
- Batch process images (JPG/PNG) and PDFs (multi-page).
- OCR using Tesseract with preprocessing.
- Extract Aadhaar number and name using heuristics + regex.
- Rename files to: Name_AadhaarNumber.ext
- Move failed files to a "failed" folder and log reasons.
- Write processing log to CSV.

Usage:
    python aadhaar_processor.py --input "path/to/folder"
    python aadhaar_processor.py --input "path/to/folder" --poppler-path "path/to/poppler/bin"
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:
    RapidOCR = None  # type: ignore

try:
    import pypdfium2 as pdfium  # type: ignore
except Exception:
    pdfium = None  # type: ignore


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


AADHAAR_REGEX_GROUPED = re.compile(r"\b(\d{4})\s+(\d{4})\s+(\d{4})\b")
AADHAAR_REGEX_COMPACT = re.compile(r"\b\d{12}\b")
DOB_REGEX = re.compile(r"\b(?:DOB|DOB:|Year of Birth|YOB|Birth)\b", re.IGNORECASE)
GENDER_REGEX = re.compile(r"\b(?:Male|Female|MALE|FEMALE|Transgender)\b", re.IGNORECASE)

_RAPID_OCR_ENGINE = None


@dataclass
class ExtractionResult:
    name: Optional[str]
    aadhaar_number: Optional[str]
    reason: Optional[str] = None


def sanitize_filename_part(value: str) -> str:
    """
    Remove characters that are invalid or problematic in file names.
    """
    cleaned = re.sub(r"[\\/:*?\"<>|]", "", value)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:80] if cleaned else "UNKNOWN"


def preprocess_image_for_ocr(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to an OCR-friendly OpenCV image.
    Steps:
    1. Grayscale
    2. Denoise
    3. Adaptive threshold
    """
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpened = cv2.addWeighted(enhanced, 1.4, cv2.GaussianBlur(enhanced, (0, 0), 2), -0.4, 0)
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return binary


def generate_oriented_variants(img: np.ndarray) -> List[np.ndarray]:
    """
    Generate orientation variants to improve OCR robustness.
    """
    variants = [img]
    variants.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    variants.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    variants.append(cv2.rotate(img, cv2.ROTATE_180))
    return variants


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 200 or max_height < 120:
        return image

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def detect_document_crop(rgb_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect likely ID-card contour and return perspective-corrected crop.
    """
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    img_area = float(h * w)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 0.2 * img_area:
            pts = approx.reshape(4, 2)
            crop = four_point_transform(rgb_img, pts)
            if crop.shape[0] > 100 and crop.shape[1] > 150:
                return crop

    return None


def build_ocr_views(rgb_img: np.ndarray) -> List[np.ndarray]:
    """
    Build multiple views: full image + scanned card crop + enhanced variants.
    """
    views: List[np.ndarray] = []
    views.append(rgb_img)

    doc_crop = detect_document_crop(rgb_img)
    if doc_crop is not None:
        views.append(doc_crop)

    all_views: List[np.ndarray] = []
    for view in views:
        gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
        )
        all_views.extend([denoised, otsu, adaptive])

    return all_views


def run_ocr_on_image(img: np.ndarray) -> str:
    """
    Run OCR and return best text.
    Priority:
    1) Tesseract (if available)
    2) RapidOCR (pure Python fallback)
    """
    texts: List[str] = []

    if tesseract_available():
        configs = ["--oem 3 --psm 6", "--oem 3 --psm 11"]
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(img, lang="eng", config=cfg)
                if text:
                    texts.append(text)
            except Exception:
                pass

    if texts:
        return max(texts, key=len)

    rapid_text = run_rapidocr(img)
    if rapid_text:
        return rapid_text

    return ""


def tesseract_available() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def get_rapidocr_engine():
    global _RAPID_OCR_ENGINE
    if RapidOCR is None:
        return None
    if _RAPID_OCR_ENGINE is None:
        _RAPID_OCR_ENGINE = RapidOCR()
    return _RAPID_OCR_ENGINE


def run_rapidocr(img: np.ndarray) -> str:
    """
    RapidOCR fallback for environments without Tesseract binary.
    """
    engine = get_rapidocr_engine()
    if engine is None:
        return ""
    try:
        result, _ = engine(img)
        if not result:
            return ""
        # result items usually: [box_points, text, score]
        lines: List[str] = []
        for item in result:
            if len(item) >= 2 and isinstance(item[1], str):
                lines.append(item[1])
        return "\n".join(lines)
    except Exception:
        return ""


def extract_aadhaar_number(text: str) -> Optional[str]:
    """
    Extract Aadhaar number while reducing false positives.
    Prefers grouped format (XXXX XXXX XXXX), then compact 12 digits.
    """
    for match in AADHAAR_REGEX_GROUPED.finditer(text):
        candidate = "".join(match.groups())
        if is_plausible_aadhaar(candidate):
            return candidate

    for match in AADHAAR_REGEX_COMPACT.finditer(text):
        candidate = match.group(0)
        if is_plausible_aadhaar(candidate):
            return candidate

    return None


def is_plausible_aadhaar(value: str) -> bool:
    """
    Minimal validation:
    - Exactly 12 digits
    - Not all digits same
    - Not obvious filler sequence
    """
    if not re.fullmatch(r"\d{12}", value):
        return False
    if len(set(value)) == 1:
        return False
    if value in {"123412341234", "111122223333", "000000000000", "999999999999"}:
        return False
    return True


def cleanup_ocr_line(line: str) -> str:
    """
    Normalize OCR line to make name detection easier.
    """
    line = line.strip()
    line = re.sub(r"[^A-Za-z .'-]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def is_likely_name(line: str) -> bool:
    if not line:
        return False
    if len(line) < 3 or len(line) > 50:
        return False
    if any(ch.isdigit() for ch in line):
        return False
    words = line.split()
    if len(words) < 2 or len(words) > 5:
        return False
    banned_tokens = {
        "government",
        "india",
        "aadhaar",
        "authority",
        "enrolment",
        "enrollment",
        "enrolmentno",
        "enrollmentno",
        "no",
        "number",
        "vid",
        "help",
        "download",
        "mobile",
        "address",
        "father",
        "husband",
        "care",
        "of",
        "uidai",
        "dob",
        "male",
        "female",
        "year",
        "birth",
        "virtual",
        "identification",
        "issue",
        "date",
        "issued",
        "valid",
        "signature",
        "card",
        "id",
        "document",
    }
    lower_words = {w.lower() for w in words}
    if lower_words & banned_tokens:
        return False
    if any(len(w) <= 1 for w in words):
        return False
    capitalized_count = sum(1 for w in words if w[0].isupper())
    return capitalized_count >= max(1, len(words) - 1)


def score_name_candidate(line: str) -> int:
    """
    Score candidate names; higher is better.
    """
    if not is_likely_name(line):
        return -10
    score = 0
    words = line.split()
    if 2 <= len(words) <= 3:
        score += 3
    if all(w[:1].isupper() for w in words):
        score += 2
    if all(len(w) >= 3 for w in words):
        score += 1
    if any(len(w) <= 2 for w in words):
        score -= 1
    if re.search(r"\b(?:No|Number|Enrolment|Enrollment|Issue|Date|Issued|Valid)\b", line, re.IGNORECASE):
        score -= 8
    return score


def extract_name(text: str) -> Optional[str]:
    """
    Heuristics:
    1) If a line contains 'Name', take same or next valid line
    2) Name near DOB/Gender lines (line above)
    3) First prominent title-cased line candidate
    """
    lines_raw = [ln for ln in text.splitlines() if ln.strip()]
    lines = [cleanup_ocr_line(ln) for ln in lines_raw]
    lines = [ln for ln in lines if ln]

    if not lines:
        return None

    for i, ln in enumerate(lines):
        if re.search(r"\bname\b", ln, flags=re.IGNORECASE):
            same_line_after = re.sub(r"(?i)\bname\b[:\-\s]*", "", ln).strip()
            if is_likely_name(same_line_after):
                return same_line_after
            if i + 1 < len(lines) and is_likely_name(lines[i + 1]):
                return lines[i + 1]

    # Name is often near the Aadhaar number line; search backwards.
    aadhaar_line_idx = None
    for i, ln in enumerate(lines):
        if extract_aadhaar_number(ln):
            aadhaar_line_idx = i
            break
    if aadhaar_line_idx is not None:
        start = max(0, aadhaar_line_idx - 8)
        candidates = []
        for j in range(start, aadhaar_line_idx):
            cand = lines[j]
            s = score_name_candidate(cand)
            if s >= 0:
                candidates.append((s, cand))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

    for i, ln in enumerate(lines):
        if DOB_REGEX.search(ln) or GENDER_REGEX.search(ln):
            if i - 1 >= 0 and is_likely_name(lines[i - 1]):
                return lines[i - 1]

    scored = [(score_name_candidate(ln), ln) for ln in lines]
    # Keep fallback strict to avoid random labels (e.g. "Issue Date").
    scored = [pair for pair in scored if pair[0] >= 4]
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    return None


def extract_from_text(text: str) -> ExtractionResult:
    aadhaar = extract_aadhaar_number(text)
    name = extract_name(text)
    if not aadhaar and not name:
        return ExtractionResult(name=None, aadhaar_number=None, reason="Name and Aadhaar not found")
    if not aadhaar:
        return ExtractionResult(name=name, aadhaar_number=None, reason="Aadhaar not found")
    if not name:
        return ExtractionResult(name=None, aadhaar_number=aadhaar, reason="Name not found")
    return ExtractionResult(name=name, aadhaar_number=aadhaar, reason=None)


def ocr_from_pil_image(pil_image: Image.Image) -> str:
    rgb = np.array(pil_image.convert("RGB"))
    views = build_ocr_views(rgb)
    if not views:
        views = [preprocess_image_for_ocr(pil_image)]

    best_text = ""
    best_score = -1

    for view in views:
        for oriented in generate_oriented_variants(view):
            txt = run_ocr_on_image(oriented)
            if not txt:
                continue
            extraction = extract_from_text(txt)
            score = 0
            if extraction.aadhaar_number:
                score += 5
            if extraction.name:
                score += 3
            score += min(len(txt) // 300, 2)
            if score > best_score:
                best_score = score
                best_text = txt
            if score >= 8:
                return txt

    return best_text


def collect_text_from_file(file_path: Path, poppler_path: Optional[str]) -> str:
    """
    Collect OCR text from supported file.
    """
    ext = file_path.suffix.lower()
    combined_texts: List[str] = []

    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        with Image.open(file_path) as img:
            combined_texts.append(ocr_from_pil_image(img))
    elif ext in SUPPORTED_PDF_EXTENSIONS:
        pages = convert_pdf_to_images(file_path, poppler_path=poppler_path)
        for page in pages:
            combined_texts.append(ocr_from_pil_image(page))
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    return "\n".join(combined_texts)


def convert_pdf_to_images(file_path: Path, poppler_path: Optional[str]) -> List[Image.Image]:
    """
    Convert PDF pages to PIL images.
    Tries pdf2image first; falls back to pypdfium2 when Poppler is unavailable.
    """
    try:
        return convert_from_path(
            str(file_path),
            dpi=300,
            poppler_path=poppler_path,
            fmt="png",
        )
    except Exception:
        pass

    if pdfium is None:
        raise RuntimeError(
            "PDF conversion failed. Install Poppler or add pypdfium2 as fallback."
        )

    pdf = pdfium.PdfDocument(str(file_path))
    pages: List[Image.Image] = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                bitmap = page.render(scale=300 / 72).to_pil()
                # Force pixel copy so no handle remains tied to the PDF file.
                pages.append(bitmap.convert("RGB").copy())
            finally:
                page.close()
    finally:
        pdf.close()
    return pages


def unique_destination_path(target_path: Path) -> Path:
    """
    If target exists, append numeric suffix.
    """
    if not target_path.exists():
        return target_path
    base = target_path.stem
    ext = target_path.suffix
    parent = target_path.parent
    idx = 1
    while True:
        candidate = parent / f"{base}_{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1


def move_to_failed(file_path: Path, failed_dir: Path) -> Path:
    failed_dir.mkdir(exist_ok=True)
    target = unique_destination_path(failed_dir / file_path.name)
    shutil.move(str(file_path), str(target))
    return target


def process_document(file_path: Path, base_dir: Path, failed_dir: Path, poppler_path: Optional[str]) -> Tuple[str, str, str, str]:
    """
    Process one file and return log tuple:
    (original_filename, extracted_name, aadhaar_number, status)
    """
    try:
        text = collect_text_from_file(file_path, poppler_path=poppler_path)
        if not text.strip():
            move_to_failed(file_path, failed_dir)
            return file_path.name, "", "", "failure: OCR returned empty text"

        extraction = extract_from_text(text)
        if not extraction.name or not extraction.aadhaar_number:
            move_to_failed(file_path, failed_dir)
            reason = extraction.reason or "Unknown extraction failure"
            return file_path.name, extraction.name or "", extraction.aadhaar_number or "", f"failure: {reason}"

        clean_name = sanitize_filename_part(extraction.name)
        new_filename = f"{clean_name}_{extraction.aadhaar_number}{file_path.suffix.lower()}"
        destination = unique_destination_path(base_dir / new_filename)
        shutil.move(str(file_path), str(destination))
        return file_path.name, extraction.name, extraction.aadhaar_number, "success"

    except Exception as exc:
        # Attempt to quarantine problematic file.
        try:
            move_to_failed(file_path, failed_dir)
        except Exception:
            pass
        err = f"{type(exc).__name__}: {exc}"
        if "tesseract" in str(exc).lower():
            err += " (install Tesseract or use RapidOCR fallback dependency)"
        if "poppler" in str(exc).lower():
            err += " (install Poppler or use pypdfium2 fallback dependency)"
        return file_path.name, "", "", f"failure: {err}"


def write_log(log_path: Path, rows: Iterable[Tuple[str, str, str, str]]) -> None:
    with log_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_filename", "extracted_name", "aadhaar_number", "status"])
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch process Aadhaar image/PDF files.")
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing Aadhaar files",
    )
    parser.add_argument(
        "--log",
        default="processing_log.csv",
        help="Output CSV log filename (inside input folder by default)",
    )
    parser.add_argument(
        "--failed-dir",
        default="failed",
        help="Folder name for failed files (inside input folder)",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional path to tesseract executable (for custom installs)",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help="Optional path to Poppler bin folder (required for pdf2image on some setups)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    failed_dir = input_dir / args.failed_dir
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = input_dir / log_path

    candidates = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALL_SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        print("No supported files found.")
        write_log(log_path, [])
        return

    rows: List[Tuple[str, str, str, str]] = []
    for path in candidates:
        result = process_document(
            file_path=path,
            base_dir=input_dir,
            failed_dir=failed_dir,
            poppler_path=args.poppler_path,
        )
        rows.append(result)
        print(f"{result[0]} -> {result[3]}")

    write_log(log_path, rows)
    print(f"Completed. Log written to: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Fatal error while processing documents.")
        traceback.print_exc()
        raise
