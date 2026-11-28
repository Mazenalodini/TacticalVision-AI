"""Unit tests for Kaggle Remote GPU Worker Bridge, Backend Proxy, and Local CPU RAM fixes."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app as backend_app
from gpu_worker import server as gpu_server


# ---------------------------------------------------------------------------
# 1. gpu_worker/server.py — /health + /analyze -> /status -> /bundle + /cancel
# ---------------------------------------------------------------------------


def test_gpu_worker_health_reports_models_dir_and_readiness(tmp_path: Path, monkeypatch):
    """Verify /health honors MODELS_DIR and reports missing vs ready model checklists."""
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    client = TestClient(gpu_server.app)

    r1 = client.get("/health")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "ok"
    assert "gpu_available" in d1
    assert "gpu_name" in d1
    assert "vram_gb" in d1
    assert d1["models_ready"] is False
    assert len(d1["missing_models"]) == 8

    # Create all 8 expected model files in tmp_path
    for fname in [
        "players_best.pt",
        "pitch_best.pt",
        "action_spotting_v1_int8.onnx",
        "action_spotting_v1.pt",
        "jersey_legibility_v1_int8.onnx",
        "jersey_legibility_v1.pt",
        "resnet152_tf2_backbone.onnx",
        "soccernet_pca512_tf2_weights.npz",
    ]:
        (tmp_path / fname).write_bytes(b"mock-weights")

    r2 = client.get("/health")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["models_ready"] is True
    assert d2["missing_models"] == []


def test_gpu_worker_analyze_status_bundle_lifecycle(tmp_path: Path, monkeypatch):
    """Verify async /analyze -> /jobs/{id}/status -> /jobs/{id}/bundle zip lifecycle."""
    monkeypatch.setenv("GPU_WORK_DIR", str(tmp_path / "work"))

    def fake_executor(job: gpu_server.GpuJobState) -> None:
        job.stage = "detecting"
        job.percent = 45.0
        job.message = "Mock GPU Inference..."
        job.logs.append("[Init] Mock GPU weights loaded")
        job.output_dir.mkdir(parents=True, exist_ok=True)
        (job.output_dir / "telemetry.json").write_text(
            json.dumps({"summary": {"fps": 25.0, "duration_s": 5.0}}),
            encoding="utf-8",
        )
        (job.output_dir / "broadcast_analyzed.mp4").write_bytes(b"FAKE_H264_BROADCAST")
        (job.output_dir / "tactical_radar.mp4").write_bytes(b"FAKE_H264_RADAR")
        (job.output_dir / "preview_snapshot.png").write_bytes(b"FAKE_PNG")

    monkeypatch.setattr(gpu_server, "PIPELINE_EXECUTOR", fake_executor)
    client = TestClient(gpu_server.app)

    # 1. Submit video asynchronously
    resp = client.post(
        "/analyze",
        files={"file": ("clip.mp4", b"FAKE_MP4_BYTES", "video/mp4")},
        data={
            "match_name": "Serie A Test",
            "max_seconds": "30",
            "start_seconds": "0",
            "enable_jersey_ocr": "true",
            "fast_scan": "false",
        },
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert job_id

    # 2. Poll until completed (fast <1s)
    final_status = None
    for _ in range(50):
        st = client.get(f"/jobs/{job_id}/status")
        assert st.status_code == 200
        data = st.json()
        if data["status"] in ("completed", "failed"):
            final_status = data
            break
        time.sleep(0.05)

    assert final_status is not None
    assert final_status["status"] == "completed", final_status
    assert final_status["percent"] == 100.0
    assert final_status["bundle_ready"] is True
    assert any("Mock GPU weights loaded" in ln for ln in final_status["logs_tail"])

    # 3. Download bundle.zip and verify all artifacts are packed cleanly
    b_res = client.get(f"/jobs/{job_id}/bundle")
    assert b_res.status_code == 200
    assert b_res.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(b_res.content), "r") as zf:
        names = set(zf.namelist())
        assert {"telemetry.json", "broadcast_analyzed.mp4", "tactical_radar.mp4"}.issubset(names)
        tel = json.loads(zf.read("telemetry.json").decode("utf-8"))
        assert tel["summary"]["fps"] == 25.0


def test_gpu_worker_cancel_endpoint(tmp_path: Path, monkeypatch):
    """Verify POST /jobs/{id}/cancel cleanly cancels an in-flight job."""
    monkeypatch.setenv("GPU_WORK_DIR", str(tmp_path / "work_cancel"))

    def slow_executor(job: gpu_server.GpuJobState) -> None:
        for _ in range(40):
            if job.cancel_event.is_set():
                raise RuntimeError("Cancelled")
            time.sleep(0.05)

    monkeypatch.setattr(gpu_server, "PIPELINE_EXECUTOR", slow_executor)
    client = TestClient(gpu_server.app)

    r = client.post(
        "/analyze",
        files={"file": ("clip.mp4", b"FAKE_MP4", "video/mp4")},
        data={"match_name": "Cancel Test"},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    c_res = client.post(f"/jobs/{job_id}/cancel")
    assert c_res.status_code == 200
    assert c_res.json()["status"] == "cancelled"


# ---------------------------------------------------------------------------
# 2. Local backend /api/gpu/ping & remote_gpu proxy error handling
# ---------------------------------------------------------------------------


def test_backend_gpu_ping_validation_and_unreachable_handling():
    """Verify /api/gpu/ping rejects invalid URLs (400) and reports unreachable workers cleanly."""
    client = TestClient(backend_app)

    # Invalid URL scheme -> HTTP 400
    bad = client.post("/api/gpu/ping", json={"url": "ftp://invalid-scheme"})
    assert bad.status_code == 400

    empty = client.post("/api/gpu/ping", json={"url": ""})
    assert empty.status_code == 400

    # Unreachable local port -> reachable=False with clear error (no crash / 500)
    unreach = client.post("/api/gpu/ping", json={"url": "http://127.0.0.1:1"})
    assert unreach.status_code == 200
    payload = unreach.json()
    assert payload["reachable"] is False
    assert payload["error"]


def test_backend_remote_gpu_analyze_rejects_bad_url_and_handles_unreachable(tmp_path: Path):
    """Verify POST /api/matches/analyze validates URLs and records clean failure on unreachable GPU."""
    client = TestClient(backend_app)

    # Bad scheme rejected immediately with 400
    bad = client.post(
        "/api/matches/analyze",
        files={"file": ("sample.mp4", b"FAKE_VIDEO", "video/mp4")},
        data={
            "title": "Bad URL Test",
            "mode": "remote_gpu",
            "remote_gpu_url": "not-a-valid-url",
        },
    )
    assert bad.status_code == 400

    # Valid http URL pointing to unreachable port -> accepts 202 and transitions cleanly to failed
    res = client.post(
        "/api/matches/analyze",
        files={"file": ("sample.mp4", b"FAKE_VIDEO", "video/mp4")},
        data={
            "title": "Unreachable GPU Test",
            "mode": "remote_gpu",
            "remote_gpu_url": "http://127.0.0.1:1",
        },
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    final = None
    for _ in range(60):
        st = client.get(f"/api/jobs/{job_id}")
        assert st.status_code == 200
        d = st.json()
        if d["status"] in ("failed", "completed", "cancelled"):
            final = d
            break
        time.sleep(0.1)

    assert final is not None
    assert final["status"] == "failed"
    assert final["error"]


# ---------------------------------------------------------------------------
# 3. Local CPU RAM OOM Fixes (Adaptive Batch Size + Detached Jersey Crops)
# ---------------------------------------------------------------------------


def test_video_feature_extractor_cpu_low_memory_batching(monkeypatch):
    """Verify SoccerNetVideoFeatureExtractor uses batch_size <= 4 on CPU and 16 on CUDA."""
    from engine import video_feature_extractor as vfe

    monkeypatch.setattr(vfe, "_cuda_available", lambda: False)
    ext_cpu = vfe.SoccerNetVideoFeatureExtractor()
    assert ext_cpu.use_cuda is False
    assert ext_cpu.batch_size <= 4

    monkeypatch.setattr(vfe, "_cuda_available", lambda: True)
    ext_gpu = vfe.SoccerNetVideoFeatureExtractor()
    assert ext_gpu.use_cuda is True
    assert ext_gpu.batch_size == 16


def test_crop_jersey_returns_detached_copy_no_frame_leak():
    """Verify crop_jersey returns an owning copy so 1080p frames are never pinned in RAM."""
    from engine.team_classifier import SigLIPTeamClassifier

    big_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop = SigLIPTeamClassifier.crop_jersey(big_frame, np.array([100, 100, 180, 300], dtype=np.float32))
    assert crop is not None
    assert crop.base is None  # Owning copy, NOT a view into big_frame!
    assert crop.nbytes < 50_000  # ~4-15 KB instead of 6.22 MB


# ---------------------------------------------------------------------------
# 4. Windows Concurrent Atomic Job JSON Stress Test (Zero WinError 5)
# ---------------------------------------------------------------------------


def test_job_runner_concurrent_windows_atomic_json_stress(tmp_path: Path):
    """Hammer the same job JSON file from writer + reader threads on Windows with 0 PermissionErrors."""
    import threading
    from backend.services.job_runner import JobRunner

    runner = JobRunner()
    runner.jobs_dir = tmp_path
    job_id = "job_stress_win5"

    # Seed initial job status
    runner._write_status(
        job_id=job_id,
        status="detecting",
        progress_pct=4.0,
        stage_label="Initial",
        elapsed_s=0.0,
        match_id="user_stress",
    )

    errors: list[Exception] = []
    reads_ok = 0
    reads_lock = threading.Lock()
    stop_readers = threading.Event()

    def writer_worker(worker_idx: int) -> None:
        try:
            for step in range(45):
                runner._write_status(
                    job_id=job_id,
                    status="detecting",
                    progress_pct=round(4.0 + (step / 45.0) * 90.0, 1),
                    stage_label=f"Writer {worker_idx} step {step}",
                    elapsed_s=float(step),
                    match_id="user_stress",
                )
        except Exception as exc:
            errors.append(exc)

    def reader_worker() -> None:
        nonlocal reads_ok
        try:
            while not stop_readers.is_set():
                st = runner.get_job_status(job_id)
                assert st is not None, "get_job_status returned None under concurrent writes"
                assert st["job_id"] == job_id
                with reads_lock:
                    reads_ok += 1
        except Exception as exc:
            errors.append(exc)

    writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(3)]
    readers = [threading.Thread(target=reader_worker) for _ in range(4)]

    for r in readers:
        r.start()
    for w in writers:
        w.start()
    for w in writers:
        w.join()
    stop_readers.set()
    for r in readers:
        r.join()

    assert not errors, f"Concurrent Windows JSON I/O raised errors: {errors}"
    assert reads_ok >= 50
    leftover_tmps = list(tmp_path.glob("*.tmp"))
    assert leftover_tmps == [], f"Leftover temp files found: {leftover_tmps}"
    final = runner.get_job_status(job_id)
    assert final is not None and final["job_id"] == job_id

