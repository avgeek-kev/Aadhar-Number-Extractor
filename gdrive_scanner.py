#!/usr/bin/env python3
"""
gdrive_scanner.py

Google Drive folder scanner for Aadhaar Processor.

Supports two authentication modes:
  1. API Key  – for public folders shared with "Anyone with the link"
  2. Service Account JSON – for private folders (upload the JSON key file in the UI)

Usage (standalone):
    python gdrive_scanner.py --folder-url "https://drive.google.com/..." --api-key "YOUR_KEY"
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Supported MIME types and their file extensions
# ---------------------------------------------------------------------------
DRIVE_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "application/pdf": ".pdf",
}

SUPPORTED_MIME_TYPES = set(DRIVE_MIME_TO_EXT.keys())

DRIVE_FILES_API = "https://www.googleapis.com/drive/v3/files"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    size: int = 0

    @property
    def extension(self) -> str:
        return DRIVE_MIME_TO_EXT.get(self.mime_type, "")


@dataclass
class DriveScanResult:
    files: List[DriveFile] = field(default_factory=list)
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    folder_name: str = ""


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def extract_folder_id(url: str) -> Optional[str]:
    """
    Extract a Google Drive folder ID from various URL formats:
      - https://drive.google.com/drive/folders/<ID>
      - https://drive.google.com/drive/u/0/folders/<ID>
      - https://drive.google.com/open?id=<ID>
      - Raw 33-char ID string passed directly
    """
    url = url.strip()

    # Pattern: /folders/<ID>
    m = re.search(r"/folders/([a-zA-Z0-9_-]{10,})", url)
    if m:
        return m.group(1)

    # Pattern: ?id=<ID>  or  &id=<ID>
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
    if m:
        return m.group(1)

    # Raw ID (no slashes, looks like a Drive ID)
    if re.match(r"^[a-zA-Z0-9_-]{10,}$", url):
        return url

    return None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_auth_params(api_key: Optional[str], access_token: Optional[str]) -> dict:
    """Return query-param dict or raise if neither credential supplied."""
    if api_key:
        return {"key": api_key}
    if access_token:
        return {}
    raise ValueError(
        "Provide either a Google API Key (for public folders) "
        "or a Service Account / OAuth access token."
    )


def _get_auth_headers(access_token: Optional[str]) -> dict:
    if access_token:
        return {"Authorization": f"Bearer {access_token}"}
    return {}


def _api_get(
    url: str,
    params: dict,
    headers: dict,
    timeout: int = 30,
) -> dict:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    if resp.status_code == 403:
        raise PermissionError(
            "Access denied (HTTP 403). "
            "Make sure the folder is shared as 'Anyone with the link' "
            "or use a Service Account with folder access."
        )
    if resp.status_code == 404:
        raise FileNotFoundError("Folder not found (HTTP 404). Check the folder ID.")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

def get_folder_name(
    folder_id: str,
    api_key: Optional[str] = None,
    access_token: Optional[str] = None,
) -> str:
    """Return the display name of a Drive folder, or empty string on failure."""
    try:
        params = _get_auth_params(api_key, access_token)
        params["fields"] = "name"
        headers = _get_auth_headers(access_token)
        data = _api_get(f"{DRIVE_FILES_API}/{folder_id}", params, headers)
        return data.get("name", "")
    except Exception:
        return ""


def list_drive_folder(
    folder_id: str,
    api_key: Optional[str] = None,
    access_token: Optional[str] = None,
    recursive: bool = False,
) -> DriveScanResult:
    """
    List all supported image/PDF files in a Google Drive folder.

    Parameters
    ----------
    folder_id : str
        The Drive folder ID.
    api_key : str, optional
        Google Cloud API key (works for public folders).
    access_token : str, optional
        OAuth2 / Service Account bearer token (works for private folders).
    recursive : bool
        If True, recurse into sub-folders.

    Returns
    -------
    DriveScanResult
    """
    result = DriveScanResult()
    result.folder_name = get_folder_name(folder_id, api_key, access_token)

    _collect_files(folder_id, api_key, access_token, recursive, result)
    return result


def _collect_files(
    folder_id: str,
    api_key: Optional[str],
    access_token: Optional[str],
    recursive: bool,
    result: DriveScanResult,
) -> None:
    """Recursive helper – populates `result` in-place."""
    params = _get_auth_params(api_key, access_token)
    headers = _get_auth_headers(access_token)

    page_token: Optional[str] = None
    while True:
        query_params = {
            **params,
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType, size)",
            "pageSize": 1000,
        }
        if page_token:
            query_params["pageToken"] = page_token

        data = _api_get(DRIVE_FILES_API, query_params, headers)

        for item in data.get("files", []):
            mime = item.get("mimeType", "")
            if mime == "application/vnd.google-apps.folder":
                if recursive:
                    _collect_files(item["id"], api_key, access_token, recursive, result)
            elif mime in SUPPORTED_MIME_TYPES:
                result.files.append(
                    DriveFile(
                        file_id=item["id"],
                        name=item.get("name", item["id"]),
                        mime_type=mime,
                        size=int(item.get("size", 0)),
                    )
                )
            else:
                result.skipped += 1

        page_token = data.get("nextPageToken")
        if not page_token:
            break


# ---------------------------------------------------------------------------
# File downloading
# ---------------------------------------------------------------------------

def download_drive_file(
    drive_file: DriveFile,
    api_key: Optional[str] = None,
    access_token: Optional[str] = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> bytes:
    """
    Download a single Drive file and return its raw bytes.
    """
    params = _get_auth_params(api_key, access_token)
    params["alt"] = "media"
    headers = _get_auth_headers(access_token)

    resp = requests.get(
        f"{DRIVE_FILES_API}/{drive_file.file_id}",
        params=params,
        headers=headers,
        stream=True,
        timeout=120,
    )
    if resp.status_code == 403:
        raise PermissionError(
            f"Cannot download '{drive_file.name}' (HTTP 403). "
            "Check sharing settings or credentials."
        )
    resp.raise_for_status()

    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=chunk_size):
        if chunk:
            buf.write(chunk)
    return buf.getvalue()


def scan_and_download(
    folder_url: str,
    api_key: Optional[str] = None,
    access_token: Optional[str] = None,
    recursive: bool = False,
    progress_callback=None,
) -> Tuple[List[Tuple[str, bytes]], List[str]]:
    """
    High-level helper: scan a Drive folder URL and download all supported files.

    Parameters
    ----------
    folder_url : str
        Full Drive folder URL or raw folder ID.
    api_key : str, optional
    access_token : str, optional
    recursive : bool
        Recurse into sub-folders.
    progress_callback : callable(current, total, filename), optional
        Called after each file download.

    Returns
    -------
    (downloaded_files, errors)
        downloaded_files: list of (filename, bytes) tuples ready for processing
        errors: list of error message strings
    """
    folder_id = extract_folder_id(folder_url)
    if not folder_id:
        raise ValueError(
            f"Could not extract a folder ID from: {folder_url!r}\n"
            "Expected a URL like: https://drive.google.com/drive/folders/XXXX"
        )

    scan_result = list_drive_folder(folder_id, api_key, access_token, recursive)

    if not scan_result.files:
        return [], scan_result.errors

    downloaded: List[Tuple[str, bytes]] = []
    errors: List[str] = list(scan_result.errors)
    total = len(scan_result.files)

    for idx, drive_file in enumerate(scan_result.files, start=1):
        try:
            data = download_drive_file(drive_file, api_key, access_token)
            # Ensure the filename has the correct extension
            filename = drive_file.name
            if not filename.lower().endswith(drive_file.extension):
                filename = f"{filename}{drive_file.extension}"
            downloaded.append((filename, data))
        except Exception as exc:
            errors.append(f"{drive_file.name}: {exc}")
        finally:
            if progress_callback:
                progress_callback(idx, total, drive_file.name)

    return downloaded, errors


# ---------------------------------------------------------------------------
# Service Account helper (generates an access token from a JSON key)
# ---------------------------------------------------------------------------

def service_account_token(sa_json: dict) -> str:
    """
    Exchange a Service Account JSON key for a short-lived Bearer token.
    Requires PyJWT and cryptography packages.
    """
    import time
    import json as _json

    try:
        import jwt  # PyJWT
    except ImportError:
        raise ImportError(
            "PyJWT is required for Service Account auth. "
            "Run: pip install PyJWT cryptography"
        )

    now = int(time.time())
    payload = {
        "iss": sa_json["client_email"],
        "scope": "https://www.googleapis.com/auth/drive.readonly",
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now,
        "exp": now + 3600,
    }
    signed_jwt = jwt.encode(
        payload,
        sa_json["private_key"],
        algorithm="RS256",
    )
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": signed_jwt,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]
