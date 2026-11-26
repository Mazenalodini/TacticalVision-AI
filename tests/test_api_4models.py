from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "artifacts" / "action_spotting_results"
client = TestClient(app)


def test_get_models_suite():
    resp = client.get("/api/models/suite")
    assert resp.status_code == 200
    data = resp.json()

    assert set(data["models"].keys()) == {
        "players_detector",
        "pitch_keypoints",
        "action_spotting",
        "jersey_legibility",
    }
    assert isinstance(data["metrics"], dict) and len(data["metrics"]) > 0
    assert "model_3_action_spotting_onnx" in data["cpu_onnx_benchmarks"]
    assert data["runtime"]["demo_features_available"] is True
    assert data["disk_inventory"]["action_spotting"]["onnx_cpu_int8"]["exists"] is True
    assert data["disk_inventory"]["jersey_legibility"]["onnx_cpu_int8"]["exists"] is True


def test_get_action_spotting_matches():
    resp = client.get("/api/action-spotting/matches")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["demo_matches"]) >= 1
    first = data["demo_matches"][0]
    assert first["match_id"] == "test_000"
    assert first["features_ready"] is True


def test_post_analyze_demo_match_and_cleanup():
    created_run_id = None
    try:
        resp = client.post(
            "/api/action-spotting/analyze",
            data={"match_id": "test_000", "half": "half1", "min_conf": "0.35", "nms_window_sec": "10.0"},
        )
        assert resp.status_code == 200
        body = resp.json()
        created_run_id = body["run_id"]

        assert body["match_id"] == "test_000"
        assert body["half_mode"] == "half1"
        assert body["total_inference_ms"] > 0.0
        assert body["backend_used"] in ("onnx_int8", "onnx_fp32", "pytorch_cpu")
        assert len(body["events"]) > 0
        assert body["gt_comparison"] is not None
        assert body["gt_comparison"]["gt_total"] > 0

        # Also verify GET /api/action-spotting/results/{run_id} retrieves persisted run
        get_resp = client.get(f"/api/action-spotting/results/{created_run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["run_id"] == created_run_id
    finally:
        if created_run_id:
            saved = RESULTS_DIR / f"{created_run_id}.json"
            if saved.exists():
                saved.unlink()


def test_post_analyze_rejects_invalid_npy_shape():
    bad_arr = np.zeros((20, 128), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, bad_arr, allow_pickle=False)
    buf.seek(0)

    resp = client.post(
        "/api/action-spotting/analyze",
        files={"file": ("bad_features.npy", buf.read(), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "512" in resp.json()["detail"]


def test_legacy_matches_and_telemetry_backward_compatible():
    m_resp = client.get("/api/matches")
    assert m_resp.status_code == 200
    matches = m_resp.json()
    assert len(matches) >= 1

    match_id = matches[0]["id"]
    t_resp = client.get(f"/api/matches/{match_id}/telemetry")
    assert t_resp.status_code == 200
    telemetry = t_resp.json()
    for key in ("summary", "players", "passing_network", "events", "heatmaps", "frames", "match_info", "engine_diagnostics"):
        assert key in telemetry
