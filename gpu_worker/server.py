"""TacticalVision AI — Standalone Kaggle / Colab GPU Worker Server.

Designed to run inside a Kaggle GPU Notebook (T4 / P100, 30GB RAM) or Google Colab
behind a Cloudflare Quick Tunnel (`cloudflared tunnel --url http://127.0.0.1:8001`).

Endpoints:
  GET  /health               -> GPU telemetry + verified 8-model readiness checklist
  POST /analyze              -> Async upload returning { job_id } immediately (202)
  GET  /jobs/{job_id}/status -> Live stage progress, percent, message, logs_tail
  GET  /jobs/{job_id}/bundle -> Streams bundle.zip (telemetry + MP4s) and cleans up
  POST /jobs/{job_id}/cancel -> Terminates active worker process and cleans temp files

Environment variables:
  MODELS_DIR        Path to read-only model directory (e.g. /kaggle/input/tacticalvision-models)
  GPU_WORK_DIR      Scratch directory for job uploads & zip bundles (default: /tmp/tv_gpu_jobs or temp)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB on cloud GPU worker

# 8 core production artifacts (with FP32 fallback acceptance for INT8 ONNX models)
REQUIRED_MODEL_SPECS: List[Tuple[str, ...]] = [
    ("players_best.pt",),
    ("pitch_best.pt",),
    ("action_spotting_v1_int8.onnx", "action_spotting_v1.onnx"),
    ("action_spotting_v1.pt",),
    ("jersey_legibility_v1_int8.onnx", "jersey_legibility_v1.onnx"),
    ("jersey_legibility_v1.pt",),
    ("resnet152_tf2_backbone.onnx",),
    ("soccernet_pca512_tf2_weights.npz",),
]


def resolve_models_dir() -> Path:
    """Resolve MODELS_DIR supporting read-only Kaggle /kaggle/input/... mounts without copying."""
    env_dir = os.environ.get("MODELS_DIR", "").strip()
    if env_dir:
        cand = Path(env_dir)
        if (cand / "players_best.pt").exists():
            return cand
        # Handle nested dataset mount (e.g., /kaggle/input/tacticalvision-models/models)
        if cand.is_dir():
            for hit in cand.rglob("players_best.pt"):
                return hit.parent
        return cand
    return PROJECT_ROOT / "models"


def check_models_readiness(models_dir: Optional[Path] = None) -> Tuple[bool, List[str], Path]:
    """Return (models_ready, missing_models, resolved_dir)."""
    mdir = models_dir or resolve_models_dir()
    missing: List[str] = []
    for candidates in REQUIRED_MODEL_SPECS:
        if not any((mdir / name).exists() for name in candidates):
            missing.append(candidates[0])
    return (len(missing) == 0), missing, mdir


def detect_gpu_info() -> Dict[str, Any]:
    """Query CUDA GPU availability, device name, and VRAM GB safely."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(float(props.total_memory) / (1024.0 ** 3), 1)
            return {
                "gpu_available": True,
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "vram_gb": vram_gb,
            }
    except Exception:
        pass
    return {
        "gpu_available": False,
        "gpu_name": "CPU (No CUDA)",
        "vram_gb": 0.0,
    }


def _get_work_root() -> Path:
    env_work = os.environ.get("GPU_WORK_DIR", "").strip()
    root = Path(env_work) if env_work else (Path(tempfile.gettempdir()) / "tacticalvision_gpu_jobs")
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class GpuJobState:
    job_id: str
    match_name: str
    job_dir: Path
    input_path: Path
    output_dir: Path
    bundle_path: Path
    max_seconds: float = 0.0
    start_seconds: float = 0.0
    enable_jersey_ocr: bool = True
    fast_scan: bool = False
    status: str = "queued"  # queued | running | completed | failed | cancelled
    stage: str = "queued"   # queued | detecting | classifying | computing | encoding | completed | failed | cancelled
    percent: float = 0.0
    message: str = "Queued on Remote GPU Worker..."
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=40))
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    bundle_ready: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)
    process: Optional[subprocess.Popen] = None

    def elapsed_s(self) -> float:
        end_t = self.finished_at if self.finished_at is not None else time.time()
        return round(max(0.0, end_t - self.created_at), 1)

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "percent": round(float(self.percent), 1),
            "message": self.message,
            "logs_tail": list(self.logs),
            "elapsed_s": self.elapsed_s(),
            "error": self.error,
            "bundle_ready": self.bundle_ready,
        }


_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, GpuJobState] = {}


def _parse_pipeline_log_line(job: GpuJobState, line: str, current_pass: List[int]) -> None:
    text = line.strip()
    if not text:
        return
    if "%|" not in text:
        job.logs.append(text[:240])
        print(f"[Job {job.job_id}] {text[:240]}", flush=True)

    if "[Init]" in text:
        job.stage = "detecting"
        job.percent = max(job.percent, 10.0)
        job.message = "Loading Full-Precision 4-Model GPU Weights..."
        return
    if "[Stage 1b]" in text:
        job.stage = "classifying"
        job.percent = max(job.percent, 56.0)
        job.message = "Running Model 3 ResNet-152 + Action Spotting..."
        return
    if "[Stage 2]" in text:
        job.stage = "classifying"
        job.percent = max(job.percent, 62.0)
        job.message = "Running SigLIP Team Clustering & Model 4 Jersey OCR..."
        return
    if "[Stage 3]" in text:
        job.stage = "computing"
        job.percent = max(job.percent, 72.0)
        job.message = "Computing Kinematics, Voronoi Control, Passes & xT..."
        return
    if "Pass 2/2" in text:
        current_pass[0] = 2
        m = re.search(r"(\d+)%\|", text)
        if m:
            pct = float(m.group(1))
            job.stage = "encoding"
            job.percent = max(job.percent, 76.0 + (pct / 100.0) * 14.0)
            job.message = f"Rendering Broadcast Overlays & 2D Radar ({int(pct)}%)..."
        else:
            job.stage = "encoding"
            job.percent = max(job.percent, 76.0)
            job.message = "Rendering Broadcast Overlays & 2D Radar..."
        return
    if "[Encoding]" in text:
        job.stage = "encoding"
        job.percent = max(job.percent, 92.0)
        job.message = "Transcoding Browser-Ready H.264 Videos..."
        return
    if "Pass 1/2" in text or current_pass[0] == 1:
        m = re.search(r"(\d+)%\|", text)
        if m:
            pct = float(m.group(1))
            job.stage = "detecting"
            job.percent = max(job.percent, 14.0 + (pct / 100.0) * 40.0)
            job.message = f"GPU Dual-YOLOv11 Detection & Tracking ({int(pct)}%)..."


def default_pipeline_executor(job: GpuJobState) -> None:
    """Run full-precision 4-Model GPU pipeline in an isolated subprocess."""
    models_dir = resolve_models_dir()
    cmd = [
        sys.executable,
        "-m",
        "backend.services.worker",
        "--job-id",
        f"gpu_{job.job_id}",
        "--video",
        str(job.input_path),
        "--output-dir",
        str(job.output_dir),
        "--title",
        job.match_name,
        "--max-seconds",
        str(job.max_seconds),
        "--start-seconds",
        str(job.start_seconds),
        "--mode",
        "remote_gpu",
        "--models-dir",
        str(models_dir),
        "--enable-jersey-ocr" if job.enable_jersey_ocr else "--no-jersey-ocr",
    ]
    if job.fast_scan:
        cmd.append("--fast-scan")

    env = os.environ.copy()
    env["MODELS_DIR"] = str(models_dir)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    job.process = proc
    current_pass = [1]

    assert proc.stdout is not None
    buf = ""
    while True:
        if job.cancel_event.is_set():
            _terminate_proc(proc)
            raise RuntimeError("Job cancelled by client.")
        ch = proc.stdout.read(1)
        if ch == "" and proc.poll() is not None:
            if buf.strip():
                _parse_pipeline_log_line(job, buf, current_pass)
            break
        if ch in ("\n", "\r"):
            if buf.strip():
                _parse_pipeline_log_line(job, buf, current_pass)
            buf = ""
        else:
            buf += ch

    rc = proc.wait()
    job.process = None
    if job.cancel_event.is_set():
        raise RuntimeError("Job cancelled by client.")
    if rc != 0:
        tail = "\n".join(list(job.logs)[-8:]) or f"Pipeline subprocess exited with code {rc}"
        raise RuntimeError(tail)


# Injectable hook for unit testing (overridable via monkeypatch)
PIPELINE_EXECUTOR: Callable[[GpuJobState], None] = default_pipeline_executor


def _terminate_proc(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=4.0)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass


def _package_bundle(job: GpuJobState) -> None:
    """Create bundle.zip containing telemetry.json, broadcast_analyzed.mp4, tactical_radar.mp4, and preview."""
    required = ["telemetry.json", "broadcast_analyzed.mp4", "tactical_radar.mp4"]
    missing = [name for name in required if not (job.output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Pipeline finished but expected outputs are missing: {missing}")

    tmp_zip = job.job_dir / "bundle.zip.tmp"
    with zipfile.ZipFile(tmp_zip, mode="w") as zf:
        for fname in required + ["preview_snapshot.png", "_meta.json"]:
            fpath = job.output_dir / fname
            if not fpath.exists():
                continue
            # MP4 & PNG are already compressed — ZIP_STORED avoids wasted CPU time
            comp = zipfile.ZIP_DEFLATED if fname.endswith(".json") else zipfile.ZIP_STORED
            zf.write(fpath, arcname=fname, compress_type=comp)
    tmp_zip.replace(job.bundle_path)


def _cleanup_job_intermediates(job: GpuJobState, remove_bundle: bool = False) -> None:
    try:
        if job.input_path.exists():
            job.input_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        if job.output_dir.exists():
            shutil.rmtree(job.output_dir, ignore_errors=True)
    except Exception:
        pass
    if remove_bundle:
        try:
            if job.job_dir.exists():
                shutil.rmtree(job.job_dir, ignore_errors=True)
        except Exception:
            pass


def _run_job_thread(job: GpuJobState) -> None:
    try:
        if job.cancel_event.is_set():
            return
        job.status = "running"
        job.stage = "detecting"
        job.percent = 5.0
        job.message = "Initializing Full-Precision 4-Model GPU Pipeline..."

        PIPELINE_EXECUTOR(job)

        if job.cancel_event.is_set():
            return

        job.stage = "encoding"
        job.percent = 96.0
        job.message = "Packaging telemetry & H.264 videos into bundle.zip..."
        _package_bundle(job)
        _cleanup_job_intermediates(job, remove_bundle=False)

        job.bundle_ready = True
        job.percent = 100.0
        job.stage = "completed"
        job.status = "completed"
        job.message = "GPU Analysis Complete — Bundle Ready!"
        job.finished_at = time.time()
    except Exception as exc:
        if job.cancel_event.is_set() or job.status == "cancelled":
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "Job Cancelled"
            job.finished_at = time.time()
            _cleanup_job_intermediates(job, remove_bundle=True)
            return
        job.status = "failed"
        job.stage = "failed"
        job.error = str(exc)
        job.message = "GPU Analysis Failed"
        job.finished_at = time.time()
        job.logs.append(f"[Error] {exc}")
        _cleanup_job_intermediates(job, remove_bundle=True)


app = FastAPI(
    title="TacticalVision AI — Remote GPU Worker",
    description="Async CUDA GPU Worker Bridge for Kaggle / Colab 4-Model Football Analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def gpu_health() -> Dict[str, Any]:
    """Return live GPU hardware info, verified model readiness, and 3D PnLCalib status."""
    from engine.pnlcalib_calibrator import check_pnlcalib_readiness

    gpu = detect_gpu_info()
    models_ready, missing_models, mdir = check_models_readiness()
    pnl = check_pnlcalib_readiness(models_dir=mdir)
    return {
        "status": "ok",
        "gpu_available": gpu["gpu_available"],
        "gpu_name": gpu["gpu_name"],
        "vram_gb": gpu["vram_gb"],
        "models_dir": str(mdir),
        "models_ready": models_ready,
        "missing_models": missing_models,
        "calibrator_backend": pnl["calibrator_backend"],
        "pnlcalib_ready": pnl["pnlcalib_ready"],
    }


@app.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def submit_gpu_analysis(
    file: UploadFile = File(...),
    match_name: str = Form("User Uploaded Match"),
    max_seconds: float = Form(0.0),
    start_seconds: float = Form(0.0),
    enable_jersey_ocr: bool = Form(True),
    fast_scan: bool = Form(False),
) -> Dict[str, Any]:
    """Accept video upload asynchronously and return { job_id } immediately."""
    filename = file.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = _get_work_root() / job_id
    output_dir = job_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{ext}"
    bundle_path = job_dir / "bundle.zip"

    total_bytes = 0
    try:
        with open(input_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to store upload: {exc}")

    job = GpuJobState(
        job_id=job_id,
        match_name=match_name.strip() or "User Uploaded Match",
        job_dir=job_dir,
        input_path=input_path,
        output_dir=output_dir,
        bundle_path=bundle_path,
        max_seconds=max(0.0, float(max_seconds or 0.0)),
        start_seconds=max(0.0, float(start_seconds or 0.0)),
        enable_jersey_ocr=bool(enable_jersey_ocr),
        fast_scan=bool(fast_scan),
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    worker_thread = threading.Thread(target=_run_job_thread, args=(job,), daemon=True)
    worker_thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}/status")
async def get_gpu_job_status(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"GPU job '{job_id}' not found.")
    return job.to_status_dict()


@app.get("/jobs/{job_id}/bundle")
async def download_gpu_bundle(job_id: str, background_tasks: BackgroundTasks):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"GPU job '{job_id}' not found.")
    if job.status != "completed" or not job.bundle_ready or not job.bundle_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Bundle not ready (current status: {job.status}).",
        )

    # Clean up temporary job directory automatically after streaming completes
    background_tasks.add_task(_cleanup_job_intermediates, job, True)
    return FileResponse(
        path=str(job.bundle_path),
        media_type="application/zip",
        filename=f"tacticalvision_{job_id}.zip",
    )


@app.post("/jobs/{job_id}/cancel")
async def cancel_gpu_job(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"GPU job '{job_id}' not found.")

    if job.status not in ("completed", "failed", "cancelled"):
        job.cancel_event.set()
        job.status = "cancelled"
        job.stage = "cancelled"
        job.message = "Job Cancelled"
        job.finished_at = time.time()
        _terminate_proc(job.process)
        job.process = None
        _cleanup_job_intermediates(job, remove_bundle=True)

    return job.to_status_dict()
