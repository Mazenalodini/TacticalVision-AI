"""Remote GPU Worker Proxy Client.

Handles:
  1. Strict `http(s)://` URL validation
  2. Live health ping (`GET {url}/health`) via local backend proxy (zero browser CORS issues)
  3. Async Kaggle / Colab job orchestration:
       POST /analyze -> poll GET /jobs/{id}/status every 2s -> download GET /jobs/{id}/bundle
       -> extract telemetry.json + broadcast_analyzed.mp4 + tactical_radar.mp4 into local outputs/{match_id}/
       -> invalidate MatchService cache & clean up temporary files.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import httpx

from .match_service import match_service

ALLOWED_BUNDLE_FILES = {
    "telemetry.json",
    "broadcast_analyzed.mp4",
    "tactical_radar.mp4",
    "preview_snapshot.png",
    "_meta.json",
}

REMOTE_STAGE_MAP = {
    "queued": "queued",
    "detecting": "detecting",
    "classifying": "classifying",
    "computing": "computing",
    "encoding": "encoding",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}


def validate_remote_gpu_url(raw_url: str) -> str:
    """Validate and normalize an http(s):// Remote GPU Worker URL."""
    cleaned = (raw_url or "").strip().rstrip("/")
    if not cleaned:
        raise ValueError("Remote GPU worker URL is required.")
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Remote GPU worker URL must be a valid http:// or https:// URL.")
    return cleaned


async def ping_remote_gpu(raw_url: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    """Ping `{url}/health` and return live GPU hardware & model readiness."""
    target_url = validate_remote_gpu_url(raw_url)
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            res = await client.get(f"{target_url}/health")
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if res.status_code != 200:
            return {
                "reachable": False,
                "url": target_url,
                "status": "error",
                "gpu_available": False,
                "gpu_name": None,
                "vram_gb": None,
                "models_ready": False,
                "missing_models": [],
                "latency_ms": latency_ms,
                "error": f"Worker returned HTTP {res.status_code}: {res.text[:200]}",
            }
        data = res.json()
        return {
            "reachable": True,
            "url": target_url,
            "status": str(data.get("status", "ok")),
            "gpu_available": bool(data.get("gpu_available", False)),
            "gpu_name": data.get("gpu_name"),
            "vram_gb": data.get("vram_gb"),
            "models_ready": bool(data.get("models_ready", False)),
            "missing_models": list(data.get("missing_models", [])),
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "reachable": False,
            "url": target_url,
            "status": "unreachable",
            "gpu_available": False,
            "gpu_name": None,
            "vram_gb": None,
            "models_ready": False,
            "missing_models": [],
            "latency_ms": latency_ms,
            "error": f"Unable to reach GPU worker: {exc}",
        }


def _safe_extract_bundle(zip_path: Path, output_dir: Path) -> None:
    """Safely extract verified match files from bundle.zip with zip-slip protection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_resolved = output_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            fname = Path(member.filename).name
            if fname not in ALLOWED_BUNDLE_FILES:
                continue
            dest = (output_dir / fname).resolve()
            if dest.parent != out_resolved:
                continue
            with zf.open(member, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    required = ["telemetry.json", "broadcast_analyzed.mp4", "tactical_radar.mp4"]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Downloaded GPU bundle is missing required files: {missing}")


def cancel_remote_gpu_job_best_effort(target_url: str, remote_job_id: str) -> None:
    """Best-effort remote cancellation call so Kaggle GPU immediately stops processing."""
    if not target_url or not remote_job_id:
        return
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            client.post(f"{target_url.rstrip('/')}/jobs/{remote_job_id}/cancel")
    except Exception:
        pass


def run_remote_gpu_bridge(
    *,
    local_job_id: str,
    match_id: str,
    video_path: Path,
    output_dir: Path,
    title: str,
    target_url: str,
    max_seconds: float,
    start_seconds: float,
    enable_jersey_ocr: bool,
    cancel_event: threading.Event,
    on_status: Callable[[str, float, str, float, Optional[str]], None],
    on_remote_job_id: Callable[[str], None],
    poll_interval_s: float = 2.0,
) -> None:
    """Full lifecycle orchestrator for a Remote GPU Worker job."""
    start_t = time.time()
    remote_job_id: Optional[str] = None
    tmp_bundle = video_path.parent / f"{local_job_id}_bundle.zip"

    def _elapsed() -> float:
        return round(time.time() - start_t, 1)

    try:
        clean_url = validate_remote_gpu_url(target_url)
        on_status("detecting", 4.0, "Uploading video to Remote GPU Worker...", _elapsed(), None)

        if cancel_event.is_set():
            return

        # 1. Upload video to POST {clean_url}/analyze
        with httpx.Client(timeout=httpx.Timeout(180.0, connect=15.0), follow_redirects=True) as client:
            with open(video_path, "rb") as vf:
                files = {"file": (video_path.name, vf, "video/mp4")}
                data = {
                    "match_name": title,
                    "max_seconds": str(max_seconds),
                    "start_seconds": str(start_seconds),
                    "enable_jersey_ocr": str(bool(enable_jersey_ocr)).lower(),
                    "fast_scan": "false",
                }
                res = client.post(f"{clean_url}/analyze", files=files, data=data)

            if res.status_code not in (200, 202):
                raise RuntimeError(
                    f"Remote GPU worker rejected upload (HTTP {res.status_code}): {res.text[:300]}"
                )
            payload = res.json()
            remote_job_id = str(payload.get("job_id") or "")
            if not remote_job_id:
                raise RuntimeError("Remote GPU worker did not return a job_id.")
            on_remote_job_id(remote_job_id)

            on_status(
                "detecting",
                8.0,
                "Connected to Remote GPU — Starting 4-Model Full-Precision Pipeline...",
                _elapsed(),
                None,
            )

            # 2. Poll GET {clean_url}/jobs/{remote_job_id}/status every 2s
            consecutive_poll_errors = 0
            while True:
                if cancel_event.is_set():
                    cancel_remote_gpu_job_best_effort(clean_url, remote_job_id)
                    return

                try:
                    st_res = client.get(f"{clean_url}/jobs/{remote_job_id}/status", timeout=20.0)
                    if st_res.status_code != 200:
                        raise RuntimeError(f"HTTP {st_res.status_code}: {st_res.text[:200]}")
                    st_data = st_res.json()
                    consecutive_poll_errors = 0
                except Exception as poll_exc:
                    consecutive_poll_errors += 1
                    if consecutive_poll_errors >= 5:
                        raise RuntimeError(
                            f"Lost connection to Remote GPU worker after 5 retries: {poll_exc}"
                        )
                    cancel_event.wait(poll_interval_s)
                    continue

                r_status = str(st_data.get("status", "running"))
                r_stage = REMOTE_STAGE_MAP.get(str(st_data.get("stage", "detecting")), "detecting")
                r_pct = float(st_data.get("percent", 10.0))
                r_msg = str(st_data.get("message") or "Processing on Remote GPU...")

                if r_status == "completed":
                    break
                if r_status == "cancelled":
                    on_status("cancelled", r_pct, "Cancelled on Remote GPU", _elapsed(), None)
                    return
                if r_status == "failed":
                    err_txt = str(st_data.get("error") or r_msg or "Remote GPU analysis failed.")
                    raise RuntimeError(err_txt)

                # Reserve 95%..100% for local bundle download & extraction
                mapped_pct = min(94.0, max(8.0, r_pct * 0.94))
                on_status(r_stage, mapped_pct, r_msg, _elapsed(), None)

                if cancel_event.wait(poll_interval_s):
                    cancel_remote_gpu_job_best_effort(clean_url, remote_job_id)
                    return

            # 3. Download GET {clean_url}/jobs/{remote_job_id}/bundle
            on_status(
                "encoding",
                96.0,
                "Downloading analyzed telemetry & H.264 videos from GPU Worker...",
                _elapsed(),
                None,
            )
            with client.stream(
                "GET",
                f"{clean_url}/jobs/{remote_job_id}/bundle",
                timeout=httpx.Timeout(300.0, connect=20.0),
            ) as b_res:
                if b_res.status_code != 200:
                    body = b_res.read().decode("utf-8", errors="replace")[:300]
                    raise RuntimeError(f"Failed to download GPU bundle (HTTP {b_res.status_code}): {body}")
                with open(tmp_bundle, "wb") as bf:
                    for chunk in b_res.iter_bytes(chunk_size=1024 * 1024):
                        if cancel_event.is_set():
                            return
                        bf.write(chunk)

        # 4. Extract bundle cleanly into local outputs/{match_id}/
        on_status("encoding", 98.5, "Extracting match bundle into local studio...", _elapsed(), None)
        _safe_extract_bundle(tmp_bundle, output_dir)

        meta_payload = {
            "title": title,
            "is_user_uploaded": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "remote_gpu",
            "duration_analyzed_s": max_seconds,
            "remote_gpu_url": clean_url,
        }
        with open(output_dir / "_meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta_payload, mf, indent=2, ensure_ascii=False)

        match_service.invalidate_cache(match_id)
        on_status("completed", 100.0, "Remote GPU Analysis Complete!", _elapsed(), None)

    except Exception as exc:
        if cancel_event.is_set():
            return
        on_status("failed", 0.0, "Remote GPU Analysis Failed", _elapsed(), str(exc))
        # Remove incomplete output directory on failure
        if output_dir.exists() and not (output_dir / "telemetry.json").exists():
            shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        for p in (video_path, tmp_bundle):
            try:
                if p.exists():
                    p.unlink(missing_ok=True)
            except Exception:
                pass
