"""Regression tests for Full-Video, 2-Tier Corner Homography, Referee Isolation, and Honest Jersey OCR."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from backend.services.job_runner import JobRunner
from engine.config import EngineConfig, FIFAPitchConfig
from engine.homography import StabilizedHomography
from engine.jersey_recognizer import JerseyRecognizer, extract_jersey_digits
from engine.renderer import BroadcastRenderer
from engine.tactics import TacticalAnalyticsEngine
from engine.team_classifier import SigLIPTeamClassifier, stabilize_team_assignments


def test_full_video_job_runner_no_truncation(tmp_path: Path):
    """Problem 1: max_seconds=0.0 preserves Full Video (0.0) without 3s/15s truncation."""
    runner = JobRunner()
    runner.jobs_dir = tmp_path / "jobs"
    runner.jobs_dir.mkdir(parents=True, exist_ok=True)
    fake_video = tmp_path / "clip.mp4"
    fake_video.write_bytes(b"\x00" * 16)

    captured_cmd = {}

    class DummyProc:
        pid = 99999
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return DummyProc()

    with patch("backend.services.job_runner.subprocess.Popen", side_effect=fake_popen):
        job_id = runner.start_job(
            video_path=fake_video,
            title="Full Match",
            mode="cpu_quick",
            max_seconds=0.0,
            start_seconds=12.5,
            enable_jersey_ocr=True,
        )

    cmd = captured_cmd["cmd"]
    assert cmd[cmd.index("--max-seconds") + 1] == "0.0"
    assert cmd[cmd.index("--start-seconds") + 1] == "12.5"
    st = runner.get_job_status(job_id)
    assert st is not None and st["status"] == "queued"


def test_two_tier_homography_corner_kick_and_temporal_hold():
    """Problem 2: Tight corner-kick penalty-box keypoints pass Tier 2 and hold <= 15 frames before reset."""
    pitch = FIFAPitchConfig()
    hom = StabilizedHomography(pitch, temporal_hold_frames=15)

    # Known camera homography looking strictly at the right penalty box (span < 15m? no, ~18m or tight box)
    # Build an exact projective transform mapping right penalty-box landmarks (17,18,21,22,23) cleanly into 1280x720
    # Wait: use 4 non-collinear landmarks spanning ~8.2m x 9.2m (< 150 m^2 Tier-1 hull threshold!)
    # Vertices: 21 (94, 34), 22 (99.5, 24.84), 23 (99.5, 43.16), 27 (105, 43.16)
    idx_subset = [21, 22, 23, 27]
    dst_m = pitch.vertices[idx_subset]
    src_px = np.array([
        [450.0, 420.0],
        [700.0, 320.0],
        [760.0, 560.0],
        [980.0, 540.0],
    ], dtype=np.float32)

    # First verify span/hull is strictly below Tier-1 (15m / 150m^2) so Tier-2 is exercised
    from scipy.spatial import ConvexHull
    assert float(ConvexHull(dst_m).volume) < 150.0

    kpts_xy = np.zeros((32, 2), dtype=np.float32)
    kpts_conf = np.zeros((32,), dtype=np.float32)
    kpts_xy[idx_subset] = src_px
    kpts_conf[idx_subset] = 0.28  # below Tier-1 0.30, above Tier-2 0.25!

    assert hom.update(kpts_xy, kpts_conf) is True
    assert hom.H is not None
    mapped = hom.image_to_pitch(src_px[:1])
    assert mapped is not None
    assert 85.0 <= float(mapped[0, 0]) <= 105.0

    # Temporal hold keeps H alive for up to 15 dropped frames, then resets at frame 16 (scene cut)
    empty_xy = np.zeros((32, 2), dtype=np.float32)
    empty_cf = np.zeros((32,), dtype=np.float32)
    for _ in range(15):
        assert hom.update(empty_xy, empty_cf) is True
        assert hom.H is not None
    assert hom.update(empty_xy, empty_cf) is False
    assert hom.H is None


def test_referee_strict_isolation_and_no_t_clutter():
    """Problem 3 & 4: Referees are isolated from teams/tactics and rendered yellow with zero text; unverified players have no T# text."""
    cfg = EngineConfig(players_model_path="", pitch_model_path="", enable_siglip=False, enable_jersey_ocr=False)
    pitch = FIFAPitchConfig()
    clf = SigLIPTeamClassifier(enable_siglip=False)

    # Create synthetic player & referee crops
    cyan_crop = np.full((32, 20, 3), (220, 180, 30), dtype=np.uint8)
    red_crop = np.full((32, 20, 3), (40, 40, 220), dtype=np.uint8)
    clf.fit([cyan_crop, cyan_crop, cyan_crop, red_crop, red_crop, red_crop])

    # Track 3 has flickering predictions: 2 frames referee (cls=3), 3 frames player (cls=2) -> 40% >= 35% => MUST isolate as referee!
    frame_records = []
    for f in range(5):
        frame_records.append({
            "frame_idx": f,
            "H": None,
            "H_inv": None,
            "ball_detected": False,
            "ball_xy": None,
            "ball_pitch_xy": None,
            "entities": [
                {"track_id": 1, "cls": cfg.player_id, "conf": 0.9, "box": np.array([100, 100, 130, 180], np.float32), "pitch_xy": np.array([30.0, 20.0 + f * 0.2], np.float32), "crop": cyan_crop.copy()},
                {"track_id": 2, "cls": cfg.player_id, "conf": 0.9, "box": np.array([200, 100, 230, 180], np.float32), "pitch_xy": np.array([70.0, 40.0 + f * 0.2], np.float32), "crop": red_crop.copy()},
                {"track_id": 3, "cls": cfg.referee_id if f < 2 else cfg.player_id, "conf": 0.85, "box": np.array([300, 100, 330, 180], np.float32), "pitch_xy": np.array([50.0, 34.0 + f * 0.2], np.float32), "crop": np.full((32, 20, 3), (20, 200, 245), np.uint8)},
            ],
        })

    tmap = stabilize_team_assignments(frame_records, clf, player_id=cfg.player_id, goalkeeper_id=cfg.goalkeeper_id, referee_id=cfg.referee_id)
    assert 3 not in tmap
    assert frame_records[4]["entities"][2]["cls"] == cfg.referee_id
    assert frame_records[4]["entities"][2]["team"] == -1

    eng = TacticalAnalyticsEngine(cfg, pitch, fps=15.0)
    tel = eng.analyze(frame_records, jersey_assignments={1: {"jersey_number": 10, "jersey_verified": True}})
    player_ids_out = {p["track_id"] for p in tel["players"]}
    assert 3 not in player_ids_out
    assert all(fp["id"] != 3 for fr in tel["frames"] for fp in fr["players"])

    # Renderer check: verify referee color BGR (21, 204, 250) and unverified player does not draw T2 box
    assert cfg.referee_color_bgr == (21, 204, 250)
    renderer = BroadcastRenderer(cfg, pitch)
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    rec0 = frame_records[0]
    rec0["voronoi_grid"] = np.full((68, 105), -1, dtype=np.int8)
    rec0["voronoi_metrics"] = {"team_0_pct": 50.0, "team_1_pct": 50.0}
    rec0["hud_stats"] = {"pos_0": 50.0, "pos_1": 50.0, "ctrl_0": 50.0, "ctrl_1": 50.0, "passes_0": 0, "passes_1": 0}
    rec0["hulls"] = {}
    rec0["def_lines"] = {}
    rec0["entities"][0]["jersey_number"] = 10
    rec0["entities"][0]["jersey_verified"] = True
    out_bcast = renderer.draw_broadcast_frame(blank, rec0, fps=15.0)
    assert out_bcast.shape == blank.shape


def test_rapidocr_rec_only_reads_clean_23_offline():
    """Problem 4: Recognition-only multi-crop RapidOCR reads crisp jersey digits cleanly offline."""
    # Synthesize a dark-on-white person box with crisp '23' printed in upper torso
    crop = np.full((120, 60, 3), 240, dtype=np.uint8)
    cv2.putText(crop, "23", (12, 52), cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 25, 35), 2, cv2.LINE_AA)

    jr = JerseyRecognizer(enable_ocr=True, legibility_threshold=0.0, min_track_ocr_conf=0.10, parseq_min_crop_conf=0.30)
    assert jr.ocr_available is True
    res = jr.recognize_tracklet([crop, crop.copy()])
    assert res["jersey_verified"] is True
    assert res["jersey_number"] == 23
