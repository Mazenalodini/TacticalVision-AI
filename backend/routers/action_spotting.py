from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..models.schemas import ActionSpottingAnalyzeResponse
from ..services import action_spotting_service

router = APIRouter(prefix="/api/action-spotting", tags=["action-spotting"])


@router.get("/matches")
async def list_action_spotting_matches():
    """List pre-extracted SoccerNet demo matches and saved Model 3 action spotting runs."""
    return action_spotting_service.list_spotting_matches()


@router.get("/results/{run_id}", response_model=ActionSpottingAnalyzeResponse)
async def get_action_spotting_result(run_id: str):
    """Retrieve a previously computed Model 3 action spotting run."""
    result = action_spotting_service.get_spotting_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Action spotting run '{run_id}' not found.")
    return result


@router.post("/analyze", response_model=ActionSpottingAnalyzeResponse)
async def analyze_action_spotting(
    request: Request,
    match_id: Optional[str] = Form(default=None),
    half: str = Form(default="half1"),
    min_conf: float = Form(default=0.35),
    nms_window_sec: float = Form(default=10.0),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Run live Model 3 CPU Temporal Action Spotting on either:
      1. A pre-extracted demo match (`match_id="test_000"`) via Form or JSON body, OR
      2. An uploaded `(T, 512)` `.npy`/`.npz` feature matrix (`file` multipart upload).
    Executes CPU inference inside `asyncio.to_thread` so the event loop stays non-blocking.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON request body: {exc}") from exc
        if isinstance(body, dict):
            match_id = body.get("match_id", match_id)
            half = str(body.get("half", half))
            min_conf = float(body.get("min_conf", min_conf))
            nms_window_sec = float(body.get("nms_window_sec", nms_window_sec))

    if not (0.01 <= float(min_conf) <= 0.99):
        raise HTTPException(status_code=400, detail="min_conf must be in [0.01, 0.99].")
    if not (1.0 <= float(nms_window_sec) <= 60.0):
        raise HTTPException(status_code=400, detail="nms_window_sec must be in [1.0, 60.0].")

    if file is not None and file.filename:
        raw_bytes = await file.read()
        try:
            return await asyncio.to_thread(
                action_spotting_service.analyze_uploaded_features,
                file.filename,
                raw_bytes,
                half,
                float(min_conf),
                float(nms_window_sec),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    target_id = (match_id or "").strip()
    if not target_id:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'match_id' (e.g. 'test_000') or a '.npy'/'.npz' feature 'file'.",
        )

    try:
        return await asyncio.to_thread(
            action_spotting_service.analyze_demo_match,
            target_id,
            half,
            float(min_conf),
            float(nms_window_sec),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
