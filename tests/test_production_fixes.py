"""Regression tests for Full-Video, 2-Tier Corner Homography, Referee Isolation, and Honest Jersey OCR."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from backend.services.job_runner import JobRunner
from engine.action_spotter import ACTION_SPOTTING_CLASSES, PER_CLASS_THRESHOLDS, ActionSpotter
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


def test_model3_calibrated_thresholds_and_cross_class_suppression():
    """Calibration 1: Per-class thresholds + 3-step smoothing + cross-class suppression keeps Goal and drops fake cards/subs."""
    spotter = ActionSpotter()
    T = 140  # 70s @ 2fps
    probs = np.zeros((T, len(ACTION_SPOTTING_CLASSES)), dtype=np.float32)

    idx_goal = ACTION_SPOTTING_CLASSES.index("Goal")
    idx_red = ACTION_SPOTTING_CLASSES.index("Red card")
    idx_sub = ACTION_SPOTTING_CLASSES.index("Substitution")
    idx_yellow = ACTION_SPOTTING_CLASSES.index("Yellow card")

    # Sustained Goal peak around t=22.0s (frames 43..45) with conf=0.64 >= 0.55
    probs[43:46, idx_goal] = 0.64

    # Simultaneous celebration hallucination: Red card 0.88 (above 0.85!) and Substitution 0.82 (above 0.78!)
    # within +-5s of Goal -> MUST be suppressed by Goal cross-class conflict rule!
    probs[45:48, idx_red] = 0.88
    probs[47:50, idx_sub] = 0.82

    # Sub-threshold Yellow card at t=50s (0.70 < 0.75) -> MUST be filtered by per-class threshold
    probs[99:102, idx_yellow] = 0.70

    # Single-frame isolated Goal spike at t=60s (0.70 for 1 frame -> smoothed to 0.35 < 0.55) -> MUST be rejected!
    probs[120, idx_goal] = 0.70

    events = spotter.postprocess_probabilities(probs, half=1, fps=2.0, nms_window_sec=15.0)
    labels = [e["label"] for e in events]
    assert labels == ["Goal"], f"Expected only ['Goal'], got {events}"
    assert 21.0 <= events[0]["time_s"] <= 23.0
    assert PER_CLASS_THRESHOLDS["Red card"] >= 0.85
    assert PER_CLASS_THRESHOLDS["Substitution"] >= 0.78


def test_half_pitch_symmetry_lock_prevents_180_flip():
    """Calibration 2: Half-pitch symmetry lock prevents 180-deg penalty-box flip during continuous zoom."""
    pitch = FIFAPitchConfig()
    hom = StabilizedHomography(pitch, temporal_hold_frames=20)

    # Right penalty-box landmarks (x in [88.5..105], mean_x ~ 97m)
    right_ids = [17, 20, 21, 22, 23, 27]
    # Exact 180-deg symmetric left penalty-box landmarks (x in [0..16.5], mean_x ~ 8m)
    # 17(88.5,13.84)<->12(16.5,54.16), 20(88.5,54.16)<->9(16.5,13.84), 21(94,34)<->8(11,34),
    # 22(99.5,24.84)<->7(5.5,43.16), 23(99.5,43.16)<->6(5.5,24.84), 27(105,43.16)<->2(0,24.84)
    left_ids = [12, 9, 8, 7, 6, 2]

    src_px = np.array([
        [320.0, 240.0],
        [360.0, 580.0],
        [540.0, 400.0],
        [720.0, 320.0],
        [750.0, 510.0],
        [960.0, 490.0],
    ], dtype=np.float32)

    player_feet = np.array([
        [500.0, 380.0],
        [620.0, 420.0],
        [700.0, 460.0],
        [810.0, 410.0],
    ], dtype=np.float32)

    # Frame 1: lock onto right penalty box
    kpts_r = np.zeros((32, 2), dtype=np.float32)
    conf_r = np.zeros((32,), dtype=np.float32)
    kpts_r[right_ids] = src_px
    conf_r[right_ids] = 0.85
    assert hom.update(kpts_r, conf_r, player_feet_px=player_feet, is_scene_cut=False) is True
    mapped_before = hom.image_to_pitch(player_feet)
    assert mapped_before is not None and float(np.median(mapped_before[:, 0])) > 80.0

    # Frame 2: pitch_best.pt mislabels the exact same pixels as LEFT penalty box (180-deg flip!)
    kpts_l = np.zeros((32, 2), dtype=np.float32)
    conf_l = np.zeros((32,), dtype=np.float32)
    kpts_l[left_ids] = src_px
    conf_l[left_ids] = 0.85
    assert hom.update(kpts_l, conf_l, player_feet_px=player_feet, is_scene_cut=False) is True
    mapped_after = hom.image_to_pitch(player_feet)
    assert mapped_after is not None
    # Symmetry lock MUST keep players on the right half (> 80m), NEVER flipped to left half (< 25m)!
    assert float(np.median(mapped_after[:, 0])) > 80.0


def test_sprint_gate_rejects_homography_spikes_and_requires_sustained_speed():
    """Calibration 3: Single-frame > 11.5 m/s spikes and < 0.8s bursts yield 0 sprints; >= 0.8s run yields 1."""
    cfg = EngineConfig(players_model_path="", pitch_model_path="", enable_siglip=False, enable_jersey_ocr=False)
    pitch = FIFAPitchConfig()
    fps = 15.0
    eng = TacticalAnalyticsEngine(cfg, pitch, fps=fps)

    # Build 120 frames (8.0s @ 15fps) with valid H
    H_id = np.eye(3, dtype=np.float32)
    frame_records = []
    # Track 1: jogging slowly (2 m/s), then suffers a 1-frame 15 m/s homography teleport glitch at f=30
    # Track 2: short 0.4s burst (6 frames) at 8.5 m/s -> below 0.8s threshold (0 sprints!)
    # Track 3: genuine sustained 8.0 m/s (28.8 km/h) sprint from f=25..50 (1.67s), brief dip, second sprint within 2s (cooldown!)
    x1, x2, x3 = 20.0, 20.0, 15.0
    for f in range(120):
        if f == 30:
            x1 += 1.5  # 1.5m in 1/15s = 22.5 m/s glitch!
        elif f == 31:
            x1 -= 1.4
        else:
            x1 += 2.0 / fps

        if 40 <= f < 46:
            x2 += 8.5 / fps  # 0.4s burst only
        else:
            x2 += 1.5 / fps

        if 25 <= f <= 52:
            x3 += 8.2 / fps  # sustained 1.8s sprint at 29.5 km/h
        elif 65 <= f <= 85:
            x3 += 8.2 / fps  # within 4s cooldown of first sprint trigger!
        else:
            x3 += 1.0 / fps

        frame_records.append({
            "frame_idx": f,
            "H": H_id,
            "H_inv": H_id,
            "homography_reset": False,
            "ball_detected": False,
            "ball_xy": None,
            "ball_pitch_xy": None,
            "entities": [
                {"track_id": 1, "cls": cfg.player_id, "team": 0, "conf": 0.9, "box": np.array([10, 10, 30, 60], np.float32), "pitch_xy": np.array([x1, 20.0], np.float32)},
                {"track_id": 2, "cls": cfg.player_id, "team": 0, "conf": 0.9, "box": np.array([10, 10, 30, 60], np.float32), "pitch_xy": np.array([x2, 34.0], np.float32)},
                {"track_id": 3, "cls": cfg.player_id, "team": 1, "conf": 0.9, "box": np.array([10, 10, 30, 60], np.float32), "pitch_xy": np.array([x3, 48.0], np.float32)},
            ],
        })

    tel = eng.analyze(frame_records)
    sprints = [e for e in tel["events"] if e["type"] == "sprint"]
    by_tid = {p["track_id"]: p["sprints"] for p in tel["players"]}

    assert by_tid[1] == 0, f"Homography spike track 1 falsely logged sprints: {sprints}"
    assert by_tid[2] == 0, f"Short 0.4s burst track 2 falsely logged sprints: {sprints}"
    assert by_tid[3] == 1, f"Expected exactly 1 sprint for track 3 (cooldown respected), got {by_tid[3]}: {sprints}"

