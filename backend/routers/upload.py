"""Video Upload, Job Polling and Match Management Router."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status

from ..config import settings
from ..models.schemas import JobStatus
from ..services.job_runner import job_runner
from ..services.match_service import match_service

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv"}
MAX_FILE_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB
CORE_MATCH_IDS = {"match_01_counter", "match_02_buildup", "match_03_pressing", "match_03_press"}


@router.post("/matches/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_match(
    file: UploadFile = File(...),
    title: str = Form("User Uploaded Match"),
    mode: str = Form("cpu_quick"),
    max_seconds: float = Form(3.0),
    remote_gpu_url: Optional[str] = Form(None),
    enable_jersey_ocr: bool = Form(True),
):
    """Upload and schedule a video for tactical AI analysis."""
    # 1. Validate file extension
    filename = file.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Remote GPU mode validation
    if mode == "remote_gpu":
        target_url = (remote_gpu_url or settings.remote_gpu_worker_url).strip()
        if not target_url or not (target_url.startswith("http://") or target_url.startswith("https://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Remote GPU worker URL is not configured or invalid. Please provide a valid http(s) URL.",
            )

        # Forward upload to remote GPU worker
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                content = await file.read()
                files = {"file": (filename, content, file.content_type)}
                data = {
                    "title": title,
                    "mode": "remote_gpu",
                    "max_seconds": str(max_seconds),
                    "enable_jersey_ocr": str(enable_jersey_ocr).lower(),
                }
                res = await client.post(f"{target_url.rstrip('/')}/api/matches/analyze", files=files, data=data)
                if res.status_code not in (200, 202):
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Remote worker rejected job: {res.text}",
                    )
                return res.json()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to remote GPU worker: {e}",
            )

    # 3. Local CPU Quick-Scan Mode
    job_uuid = uuid.uuid4().hex[:8]
    safe_name = f"{job_uuid}_{Path(filename).stem}{ext}"
    dest_path = settings.tmp_uploads_dir / safe_name

    # Save uploaded file in chunks with size guard
    total_bytes = 0
    try:
        with open(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded file exceeds maximum allowed limit ({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB).",
                    )
                f.write(chunk)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {e}")

    # Launch worker via job_runner
    job_id = job_runner.start_job(
        video_path=dest_path,
        title=title,
        mode=mode,
        max_seconds=max_seconds,
        enable_jersey_ocr=enable_jersey_ocr,
    )

    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Retrieve live processing status and progress percentage."""
    st = job_runner.get_job_status(job_id)
    if st is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return st


@router.post("/jobs/{job_id}/cancel", response_model=JobStatus)
async def cancel_job(job_id: str):
    """Cancel an active analysis job, terminate its worker subprocess, and clean up temp files."""
    st = job_runner.cancel_job(job_id)
    if st is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return st


@router.delete("/matches/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_match(match_id: str):
    """Permanently delete a user-uploaded match (core demo matches are protected)."""
    # 1. Protect core matches
    if match_id in CORE_MATCH_IDS or match_id.startswith("match_0"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Match '{match_id}' is a core pre-analyzed demo and cannot be deleted.",
        )

    # 2. Path traversal security check
    target_dir = (settings.outputs_dir / match_id).resolve()
    if target_dir.parent != settings.outputs_dir.resolve():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid match path.",
        )

    if not target_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match '{match_id}' not found.",
        )

    # 3. Confirm is_user_uploaded from _meta.json
    meta_path = target_dir / "_meta.json"
    is_user_uploaded = False
    if meta_path.exists():
        try:
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                is_user_uploaded = bool(meta.get("is_user_uploaded", False))
        except Exception:
            pass

    if not is_user_uploaded and not match_id.startswith("user_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only user-uploaded matches can be deleted.",
        )

    # 4. Delete match directory safely
    try:
        shutil.rmtree(target_dir)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete match directory: {e}",
        )

    # Invalidate cache
    match_service.invalidate_cache(match_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
