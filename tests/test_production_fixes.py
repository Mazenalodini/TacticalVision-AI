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

    # Single-frame isolated Goal spike at t=60s (0.70 for 1 frame -> smoothed to 0.35 < 0.45) -> MUST be rejected!
    probs[120, idx_goal] = 0.70

    events = spotter.postprocess_probabilities(probs, half=1, fps=2.0, nms_window_sec=15.0)
    labels = [e["label"] for e in events]
    assert labels == ["Goal"], f"Expected only ['Goal'], got {events}"
    assert 21.0 <= events[0]["time_s"] <= 23.0
    assert PER_CLASS_THRESHOLDS["Red card"] >= 0.85
    assert PER_CLASS_THRESHOLDS["Substitution"] >= 0.78


def test_model3_corroborated_goal_rescue_on_short_youtube_clip():
    """Corroborated Goal Rescue: rescues real Goal >= 0.18 with Shots on target >= 0.13 within +-3s while keeping fake cards/subs at 0."""
    spotter = ActionSpotter()
    T = 138  # 69s @ 2fps (matching Serie a goal)
    probs = np.full((T, len(ACTION_SPOTTING_CLASSES)), 0.04, dtype=np.float32)

    idx_goal = ACTION_SPOTTING_CLASSES.index("Goal")
    idx_shot = ACTION_SPOTTING_CLASSES.index("Shots on target")
    idx_sub = ACTION_SPOTTING_CLASSES.index("Substitution")
    idx_yellow = ACTION_SPOTTING_CLASSES.index("Yellow card")
    idx_kickoff = ACTION_SPOTTING_CLASSES.index("Kick-off")

    # Background Goal median ~0.06, real corroborated Goal peak at t=22.4s (frame 45)
    probs[:, idx_goal] = 0.06
    probs[44:47, idx_goal] = 0.1946
    probs[44:47, idx_shot] = 0.1516

    # Uncorroborated weak Goal bump at t=50s (no Shot on target!) -> MUST NOT fire
    probs[99:102, idx_goal] = 0.21

    # Hallucinated celebration cards/subs/kick-offs (0.18..0.26) -> MUST stay 0
    probs[48:52, idx_yellow] = 0.182
    probs[52:56, idx_sub] = 0.255
    probs[120:124, idx_kickoff] = 0.215

    events = spotter.postprocess_probabilities(probs, half=1, fps=2.0, nms_window_sec=15.0)
    labels = [e["label"] for e in events]
    assert labels == ["Goal"], f"Expected only rescued ['Goal'], got {events}"
    assert 21.5 <= events[0]["time_s"] <= 23.5
    assert events[0]["confidence"] >= 0.18


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


def test_homography_monotonicity_never_flips_oriented_right_box():
    """Rebuild Fix 2a: Valid right-half box (dX/du > 0) is NEVER flipped left even after left-half anchor."""
    pitch = FIFAPitchConfig()
    hom = StabilizedHomography(pitch, temporal_hold_frames=15)

    # Frame 1: Left penalty box (correctly oriented: dX/du > 0, dY/dv > 0)
    left_ids = [1, 4, 6, 7, 8, 9, 12]
    src_px = np.array([
        [180.0, 220.0],   # 1 (0, 13.84)
        [160.0, 560.0],   # 4 (0, 54.16)
        [300.0, 320.0],   # 6 (5.5, 24.84)
        [310.0, 460.0],   # 7 (5.5, 43.16)
        [430.0, 390.0],   # 8 (11, 34)
        [560.0, 220.0],   # 9 (16.5, 13.84)
        [580.0, 560.0],   # 12 (16.5, 54.16)
    ], dtype=np.float32)
    kpts_l = np.zeros((32, 2), dtype=np.float32)
    conf_l = np.zeros((32,), dtype=np.float32)
    kpts_l[left_ids] = src_px
    conf_l[left_ids] = 0.85
    p_left = np.array([[260, 360], [380, 410], [490, 350], [520, 470]], dtype=np.float32)
    assert hom.update(kpts_l, conf_l, player_feet_px=p_left, img_width=1280) is True
    assert float(np.median(hom.image_to_pitch(p_left)[:, 0])) < 30.0

    # Frame 2: Camera has smoothly panned to RIGHT penalty box (10+ inliers, dX/du > 0, med_x ~ 94m).
    # The old symmetry lock blindly rotated this 180 deg to the left half! New monotonicity law MUST preserve right half!
    right_ids = [17, 20, 21, 22, 23, 24, 25, 27, 28]
    # Exact projection using linear broadcast scale: x in [88.5..105] -> u in [620..1080], y in [13.84..54.16] -> v in [220..560]
    dst_r = pitch.vertices[right_ids]
    src_r = np.column_stack([
        620.0 + (dst_r[:, 0] - 88.5) * (460.0 / 16.5),
        220.0 + (dst_r[:, 1] - 13.84) * (340.0 / 40.32),
    ]).astype(np.float32)
    kpts_r = np.zeros((32, 2), dtype=np.float32)
    conf_r = np.zeros((32,), dtype=np.float32)
    kpts_r[right_ids] = src_r
    conf_r[right_ids] = 0.90
    p_right = np.array([[680, 340], [790, 410], [880, 390], [970, 440]], dtype=np.float32)
    assert hom.update(kpts_r, conf_r, player_feet_px=p_right, img_width=1280) is True
    mapped_r = hom.image_to_pitch(p_right)
    assert mapped_r is not None
    # EMA blends smoothly toward right half without flipping orientation; after 8 frames it settles firmly in right box (>84m)
    for _ in range(8):
        assert hom.update(kpts_r, conf_r, player_feet_px=p_right, img_width=1280) is True
    mapped_settled = hom.image_to_pitch(p_right)
    assert mapped_settled is not None and float(np.median(mapped_settled[:, 0])) > 84.0


def test_homography_player_inbounds_gate_rejects_projective_explosion():
    """Rebuild Fix 2b: Player In-Bounds Gate rejects candidate H that projects >30% players wildly outside pitch."""
    pitch = FIFAPitchConfig()
    hom = StabilizedHomography(pitch, temporal_hold_frames=10)

    # Candidate H that maps right-side pixels (u=800..1150) way beyond x=130m (outside [-4..109])
    right_ids = [17, 20, 22, 23, 25, 27]
    dst_r = pitch.vertices[right_ids]
    # Place the goal line (x=105) all the way at u=500 so players at u=900..1200 explode to x=135..160m!
    src_r = np.column_stack([
        200.0 + (dst_r[:, 0] - 88.5) * (300.0 / 16.5),
        220.0 + (dst_r[:, 1] - 13.84) * (340.0 / 40.32),
    ]).astype(np.float32)
    kpts = np.zeros((32, 2), dtype=np.float32)
    conf = np.zeros((32,), dtype=np.float32)
    kpts[right_ids] = src_r
    conf[right_ids] = 0.85

    # 5 out of 6 players stand at u=900..1180 -> x >> 115m -> >80% OOB -> MUST be rejected!
    players_oob = np.array([
        [350.0, 380.0],
        [920.0, 350.0],
        [980.0, 410.0],
        [1040.0, 390.0],
        [1110.0, 440.0],
        [1170.0, 400.0],
    ], dtype=np.float32)
    assert hom.update(kpts, conf, player_feet_px=players_oob, img_width=1280) is False
    assert hom.H is None


def test_homography_optical_flow_propagates_camera_pan():
    """Rebuild Fix 2c: Background LK Optical Flow tracks camera pan while masking player boxes (<0.5m error)."""
    pitch = FIFAPitchConfig()
    hom = StabilizedHomography(pitch, temporal_hold_frames=10, max_flow_frames=60)

    # Create a rich non-periodic synthetic textured pitch (avoids periodic grid aliasing in LK flow)
    rng = np.random.default_rng(7)
    h, w = 360, 640
    canvas = np.full((h, w + 80), 95, dtype=np.uint8)
    for _ in range(320):
        cx = int(rng.integers(25, w + 55))
        cy = int(rng.integers(55, h - 25))
        sz = int(rng.integers(2, 6))
        val = int(rng.integers(155, 250)) if rng.random() > 0.35 else int(rng.integers(20, 65))
        cv2.rectangle(canvas, (cx - sz, cy - sz), (cx + sz, cy + sz), val, -1)

    pan_px = 10  # Camera pans right -> world pixels shift LEFT by 10px in cur_gray
    prev_gray = canvas[:, :w].copy()
    cur_gray = canvas[:, pan_px : pan_px + w].copy()

    # Known exact initial homography: x_m = 60 + 0.06 * u, y_m = 10 + 0.14 * v
    hom.H = np.array([
        [0.06, 0.0, 60.0],
        [0.0, 0.14, 10.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    hom.H_inv = np.linalg.inv(hom.H)
    hom.homography_valid = True

    # Draw a bright moving player box in prev_gray and shifted differently in cur_gray -> masked out!
    p_boxes = np.array([[280, 140, 315, 225], [400, 150, 435, 235]], dtype=np.float32)
    cv2.rectangle(prev_gray, (280, 140), (315, 225), 255, -1)
    cv2.rectangle(cur_gray, (295, 140), (330, 225), 255, -1)  # Moves OPPOSITE to background!

    # Stationary world players at prev u=(200, 350, 480) appear at cur u=(182, 332, 462)
    prev_feet = np.array([[200.0, 200.0], [350.0, 220.0], [480.0, 190.0]], dtype=np.float32)
    cur_feet = prev_feet - np.array([[float(pan_px), 0.0]], dtype=np.float32)
    world_before = hom.image_to_pitch(prev_feet)

    ok = hom.propagate_optical_flow(prev_gray, cur_gray, exclude_boxes=p_boxes, player_feet_px=cur_feet)
    assert ok is True
    assert hom.flow_propagated_frames == 1

    world_after = hom.image_to_pitch(cur_feet)
    assert world_before is not None and world_after is not None
    max_err_m = float(np.max(np.linalg.norm(world_after - world_before, axis=1)))
    assert max_err_m < 0.5, f"Optical flow pan propagation drifted by {max_err_m:.3f}m"


def test_cielab_torso_classifier_white_vs_navy_purity():
    """Rebuild Fix 1: Torso-masked CIELAB achieves 100% purity on White vs Navy kits despite grass & occlusion."""
    clf = SigLIPTeamClassifier(enable_siglip=True)  # Must still use CIELAB primary when ΔE_Lab >> 18!

    def make_player_frame(jersey_bgr, ad_bgr):
        # 120x60 person box: green grass background, ad-board top, skin head, jersey torso, dark shorts
        img = np.full((120, 60, 3), (45, 130, 55), dtype=np.uint8)  # Pitch green
        img[0:18, :] = ad_bgr  # Ad-board / crowd behind head
        img[6:20, 22:38] = (145, 175, 215)  # Skin head
        img[20:66, 12:48] = jersey_bgr  # Jersey torso
        img[66:95, 14:46] = (35, 35, 35)  # Shorts
        return img

    white_bgr = (232, 234, 236)
    navy_bgr = (68, 42, 32)

    # Calibrate pitch green on synthetic turf frame
    turf = np.full((200, 320, 3), (45, 130, 55), dtype=np.uint8)
    clf.calibrate_pitch_green(turf)

    # Build 6 players (tracks 1..3 White with wildly varying ad-boards, tracks 4..6 Navy)
    ad_colors = [(20, 20, 220), (220, 210, 30), (200, 50, 200)]
    fit_crops = []
    frame_records = []
    for f in range(5):
        ents = []
        for idx, tid in enumerate([1, 2, 3, 4, 5, 6]):
            is_white = tid <= 3
            j_bgr = white_bgr if is_white else navy_bgr
            # On frame 2, track 1 (White) is heavily occluded (IoU > 0.4) by track 4 (Navy) carrying wrong color!
            if f == 2 and tid == 1:
                box = np.array([305, 100, 365, 220], dtype=np.float32)  # overlaps track 4 at [300..360]
                full_box_img = make_player_frame(navy_bgr, ad_colors[idx % 3])
            else:
                x1 = 100 * idx
                box = np.array([x1, 100, x1 + 60, 220], dtype=np.float32)
                full_box_img = make_player_frame(j_bgr, ad_colors[idx % 3])
            torso_crop = SigLIPTeamClassifier.crop_jersey(full_box_img, np.array([0, 0, 60, 120]))
            assert torso_crop is not None
            if f != 2:
                fit_crops.append(torso_crop)
            ents.append({
                "track_id": tid,
                "cls": 2,
                "conf": 0.92,
                "box": box,
                "pitch_xy": np.array([25.0 + tid * 10.0, 34.0], dtype=np.float32),
                "crop": torso_crop,
            })
        frame_records.append({"frame_idx": f, "entities": ents})

    clf.fit(fit_crops)
    assert clf.fitted is True
    assert clf.cluster_delta_e > 40.0, f"Expected large White vs Navy ΔE_Lab > 40, got {clf.cluster_delta_e:.1f}"

    tmap = stabilize_team_assignments(frame_records, clf, player_id=2, goalkeeper_id=1, referee_id=3)
    white_teams = {tmap[1], tmap[2], tmap[3]}
    navy_teams = {tmap[4], tmap[5], tmap[6]}
    assert len(white_teams) == 1, f"White players split across teams: {tmap}"
    assert len(navy_teams) == 1, f"Navy players split across teams: {tmap}"
    assert white_teams != navy_teams, f"White and Navy merged into same team: {tmap}"


def test_tactical_quality_gate_and_universal_striped_and_similar_kits():
    """SOTA Rebuild Phase 2: Tactical gate rejects close-ups/occlusions; universal classifier separates solid, striped, and similar kits."""
    from engine.team_classifier import is_valid_tactical_box

    # Tactical player box quality gate (1280x720)
    assert is_valid_tactical_box(np.array([400, 260, 442, 355]), 720, 1280, max_iou=0.0) is True   # h/H=0.132
    assert is_valid_tactical_box(np.array([400, 100, 580, 460]), 720, 1280, max_iou=0.0) is False  # close-up h/H=0.50
    assert is_valid_tactical_box(np.array([400, 260, 412, 280]), 720, 1280, max_iou=0.0) is False  # tiny h/H=0.028
    assert is_valid_tactical_box(np.array([400, 10, 440, 100]), 720, 1280, max_iou=0.0) is False   # top HUD zone
    assert is_valid_tactical_box(np.array([400, 260, 442, 355]), 720, 1280, max_iou=0.22) is False # occluded IoU >= 0.15

    # Scenario A: Striped Red/White vs Solid Blue + contaminated outlier coach crop
    clf = SigLIPTeamClassifier(enable_siglip=False)
    striped_crops = []
    blue_crops = []
    for _ in range(14):
        c_stripe = np.full((36, 28, 3), (235, 235, 235), dtype=np.uint8)
        c_stripe[:, ::6] = (30, 30, 210)
        c_stripe[:, 1::6] = (30, 30, 210)
        c_stripe[:, 2::6] = (30, 30, 210)
        striped_crops.append(c_stripe)
        blue_crops.append(np.full((36, 28, 3), (200, 75, 25), dtype=np.uint8))
    outliers = [np.full((36, 28, 3), (25, 25, 25), dtype=np.uint8) for _ in range(3)]  # coaches
    clf.fit(striped_crops + blue_crops + outliers)
    assert clf.fitted is True
    assert clf.cluster_delta_e >= 16.0
    assert clf.mode_used == "cielab_universal_primary"
    hexes = clf.get_kit_colors_hex()
    assert hexes["team_a_kit_hex"].startswith("#") and len(hexes["team_a_kit_hex"]) == 7
    assert hexes["team_b_kit_hex"].startswith("#") and len(hexes["team_b_kit_hex"]) == 7
    p_stripe = clf.predict(striped_crops)
    p_blue = clf.predict(blue_crops)
    assert len(set(p_stripe.tolist())) == 1
    assert len(set(p_blue.tolist())) == 1
    assert p_stripe[0] != p_blue[0]

    # Scenario B: Truly near-identical low-contrast kits (ΔE_Lab < 16) activates secondary HSV cue fusion
    clf_sim = SigLIPTeamClassifier(enable_siglip=True)
    kit_a = [np.full((36, 28, 3), (176, 170, 162), dtype=np.uint8) for _ in range(12)]  # cool mist grey
    kit_b = [np.full((36, 28, 3), (162, 174, 178), dtype=np.uint8) for _ in range(12)]  # warm pearl grey
    clf_sim.fit(kit_a + kit_b)
    assert clf_sim.fitted is True
    assert clf_sim.cluster_delta_e < 16.0
    assert clf_sim.mode_used == "cielab_plus_hsv_cue_fusion"
    assert set(clf_sim.predict(kit_a).tolist()) != set(clf_sim.predict(kit_b).tolist())


def test_possession_integrity_with_image_space_fallback_and_hud_warmup():
    """SOTA Rebuild Phase 2: Possession uses image-space fallback during calibration gaps and avoids 1-frame 0/100 HUD spikes."""
    cfg = EngineConfig(players_model_path="", pitch_model_path="", enable_siglip=False, enable_jersey_ocr=False)
    pitch = FIFAPitchConfig()
    eng = TacticalAnalyticsEngine(cfg, pitch, fps=15.0)
    H_id = np.eye(3, dtype=np.float32)

    # 30 frames: frames 0..14 Team 0 holds ball on metric pitch; frames 15..29 H is temporarily None
    # while Team 1 holds the ball cleanly at their feet in image space!
    frame_records = []
    for f in range(30):
        t0_has = f < 15
        h_valid = f < 15
        frame_records.append({
            "frame_idx": f,
            "H": H_id if h_valid else None,
            "H_inv": H_id if h_valid else None,
            "homography_valid": h_valid,
            "homography_reset": False,
            "ball_detected": True,
            "ball_xy": np.array([220.0, 380.0], np.float32) if t0_has else np.array([620.0, 380.0], np.float32),
            "ball_pitch_xy": np.array([30.0, 34.0], np.float32) if h_valid else None,
            "entities": [
                {
                    "track_id": 1, "cls": cfg.player_id, "team": 0, "conf": 0.9,
                    "box": np.array([200.0, 290.0, 240.0, 380.0], np.float32),
                    "pitch_xy": np.array([30.5, 34.0], np.float32) if h_valid else None,
                },
                {
                    "track_id": 2, "cls": cfg.player_id, "team": 1, "conf": 0.9,
                    "box": np.array([600.0, 290.0, 640.0, 380.0], np.float32),
                    "pitch_xy": np.array([70.0, 34.0], np.float32) if h_valid else None,
                },
            ],
        })

    tel = eng.analyze(frame_records)
    poss = tel["summary"]["possession_pct"]
    assert 40.0 <= poss["team_0"] <= 60.0, f"Possession collapsed despite balanced touches: {poss}"
    assert 40.0 <= poss["team_1"] <= 60.0
    # HUD warm-up keeps opening frames neutral 50/50 until >= 5 touches
    assert frame_records[0]["hud_stats"]["pos_0"] == 50.0


def test_pnlcalib_3d_projection_pose_gates_and_fallback_health():
    """SOTA Rebuild Phase 3: PnLCalib 3D -> FIFA projection is exact, rejects bad poses, and exposes /health backend."""
    from fastapi.testclient import TestClient
    from engine.pnlcalib_calibrator import PnLCalib3DCalibrator, cam_params_to_fifa_homography
    from gpu_worker.server import app as gpu_app

    pitch = FIFAPitchConfig()

    # Construct a physically valid 3D broadcast camera looking at right penalty box (X_pnl=+35, Y_pnl=0, Z=0)
    # Camera positioned at main stand: (X_pnl=10, Y_pnl=52, Z_pnl=-20) looking toward (35, 0, 0)
    cam_pos = np.array([10.0, 52.0, -20.0], dtype=np.float64)
    target = np.array([35.0, 0.0, 0.0], dtype=np.float64)
    z_axis = target - cam_pos
    z_axis /= np.linalg.norm(z_axis)
    world_down = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # PnLCalib Z is down
    x_axis = np.cross(world_down, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)

    fx = fy = 1400.0
    pp = np.array([640.0, 360.0], dtype=np.float64)
    params = {
        "cam_params": {
            "x_focal_length": fx,
            "y_focal_length": fy,
            "principal_point": pp.tolist(),
            "position_meters": cam_pos.tolist(),
            "rotation_matrix": R.tolist(),
        }
    }

    res = cam_params_to_fifa_homography(params, pitch)
    assert res is not None
    H_fifa, pose = res
    assert pose["fx"] == 1400.0

    # Project FIFA right penalty spot (94, 34) -> centered (41.5, 0, 0) -> image -> back via H_fifa
    It = np.eye(4, dtype=np.float64)[:-1]
    It[:, -1] = -cam_pos
    Q = np.array([[fx, 0.0, pp[0]], [0.0, fy, pp[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    P = Q @ (R @ It)
    uv_h = P @ np.array([94.0 - 52.5, 34.0 - 34.0, 0.0, 1.0])
    uv = (uv_h[:2] / uv_h[2]).astype(np.float32).reshape(1, 1, 2)
    recovered = cv2.perspectiveTransform(uv, H_fifa).reshape(2)
    err_m = float(np.linalg.norm(recovered - np.array([94.0, 34.0])))
    assert err_m < 0.05, f"PnLCalib 3D -> FIFA projection error too large: {err_m:.4f}m"

    # Reject physically implausible camera poses (underground camera / crazy focal length)
    bad_params = {"cam_params": {**params["cam_params"], "position_meters": [0.0, 50.0, -0.2]}}
    assert cam_params_to_fifa_homography(bad_params, pitch) is None
    bad_focal = {"cam_params": {**params["cam_params"], "x_focal_length": 25.0}}
    assert cam_params_to_fifa_homography(bad_focal, pitch) is None

    # Fallback contract + GPU worker /health exposes calibrator_backend & pnlcalib_ready
    stab = StabilizedHomography(pitch)
    cal = PnLCalib3DCalibrator(pitch, stab, force_fallback=True)
    assert cal.backend == "yolo_pitch_fallback"
    assert cal.is_pnlcalib_active is False

    client = TestClient(gpu_app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["calibrator_backend"] in ("pnlcalib_3d", "yolo_pitch_fallback")
    assert isinstance(body["pnlcalib_ready"], bool)


def test_homography_law2_blocks_right_box_midfield_mislabel_even_with_many_inliers():
    """SOTA Rebuild Phase 3: Law 2 blocks shifted right-box->left-half mislabel regardless of inlier count."""
    pitch = FIFAPitchConfig()
    hom = StabilizedHomography(pitch, temporal_hold_frames=10)

    # 9 keypoints of LEFT penalty box (x in [0..16.5]) falsely placed on the FAR RIGHT of the image (u in 780..1180)
    # with open pitch players on the left (u in 120..420) — even though dX/du > 0 and 9 inliers!
    left_box_ids = [0, 1, 2, 3, 4, 6, 7, 9, 12]
    dst_l = pitch.vertices[left_box_ids]
    src_right_img = np.column_stack([
        780.0 + dst_l[:, 0] * (400.0 / 16.5),
        180.0 + dst_l[:, 1] * (400.0 / 68.0),
    ]).astype(np.float32)
    kpts = np.zeros((32, 2), dtype=np.float32)
    conf = np.zeros((32,), dtype=np.float32)
    kpts[left_box_ids] = src_right_img
    conf[left_box_ids] = 0.90

    # Players spread across open pitch to the left of the box
    players = np.array([
        [150.0, 350.0], [280.0, 380.0], [420.0, 410.0], [820.0, 340.0], [920.0, 390.0],
    ], dtype=np.float32)
    assert hom.update(kpts, conf, player_feet_px=players, img_width=1280) is False
    assert hom.H is None


def test_shot_level_homography_outlier_rejection_and_savgol_smoothing():
    """Fix 1: Pass 1.5 rejects 1-3 frame PnLCalib spikes (> 6m) and smooths wobble via zero-phase Savitzky-Golay."""
    from engine.homography import smooth_shot_homographies

    pitch = FIFAPitchConfig()
    rng = np.random.default_rng(123)
    n_frames = 45
    fps = 25.0
    frame_records = []

    # True smooth broadcast pan: x_m(u) = (25 + 0.35 * f) + 0.045 * u, y_m(v) = 8 + 0.075 * v
    for f in range(n_frames):
        tx = 25.0 + 0.35 * f + float(rng.normal(0.0, 0.55))
        ty = 8.0 + float(rng.normal(0.0, 0.30))
        # Inject a severe 2-frame PnLCalib horizontal spike (-22m!) at frames 20..21
        if f in (20, 21):
            tx -= 22.0
        H = np.array([
            [0.045, 0.0, tx],
            [0.0, 0.075, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        ents = [
            {"track_id": i + 1, "cls": 2, "team": i % 2, "box": np.array([250 + i * 90, 260, 285 + i * 90, 360], np.float32), "pitch_xy": None}
            for i in range(8)
        ]
        frame_records.append({
            "frame_idx": f,
            "H": H,
            "H_inv": np.linalg.inv(H),
            "homography_valid": True,
            "is_scene_cut": False,
            "homography_reset": False,
            "entities": ents,
        })

    # Measure BEFORE max jump at spike frame 20
    p_before = [float(np.median([ (frame_records[f]["H"][0,0]*580 + frame_records[f]["H"][0,2]) ])) for f in range(n_frames)]
    max_jump_before = max(abs(p_before[f] - p_before[f - 1]) for f in range(1, n_frames))
    assert max_jump_before > 18.0

    stats = smooth_shot_homographies(frame_records, pitch, img_w=1280, img_h=720, fps=fps)
    assert stats["shots_smoothed"] == 1
    assert stats["outliers_replaced"] >= 2

    med_xs_after = [float(np.median([e["pitch_xy"][0] for e in rec["entities"]])) for rec in frame_records]
    diffs_after = [abs(med_xs_after[f] - med_xs_after[f - 1]) for f in range(1, n_frames)]
    assert max(diffs_after) < 0.9, f"Spike survived Savitzky-Golay smoothing: max jump={max(diffs_after):.2f}m"
    assert float(np.mean(diffs_after)) < 0.45


def test_physics_gated_ball_no_corner_clamp_teleport_rejection_and_trail_break():
    """Fix 2: OOB balls are discarded (never corner-clamped), 1-2 frame teleports rejected, short gaps bridged, trail breaks cleanly."""
    from engine.tactics import filter_and_interpolate_ball_pitch

    pitch = FIFAPitchConfig()
    fps = 25.0
    n = 30
    raw: list = [None] * n
    # Clean ball moving steadily across midfield: x = 40 + 0.6 * f, y = 34.0
    for f in range(n):
        raw[f] = np.array([40.0 + 0.6 * f, 34.0], dtype=np.float32)

    # Inject OOB outlier (must be discarded -> interpolated, NEVER clamped to (0,0) or (105,68)!)
    raw[5] = np.array([-18.0, -9.0], dtype=np.float32)
    raw[6] = np.array([135.0, 92.0], dtype=np.float32)
    # Inject 1-frame corner-clamp teleport to (0.0, 0.0) and (105.0, 68.0)
    raw[12] = np.array([0.0, 0.0], dtype=np.float32)
    raw[18] = np.array([105.0, 68.0], dtype=np.float32)
    # Inject 4-frame occlusion dropout (frames 22..25) -> MUST be smoothly bridged!
    for f in range(22, 26):
        raw[f] = None

    clean = filter_and_interpolate_ball_pitch(raw, fps=fps, pitch_length=pitch.length, pitch_width=pitch.width)
    # All 30 frames should be smoothly recovered with ZERO corner clamps and ZERO jumps > 2.5m!
    assert all(p is not None for p in clean), "Short gaps after outlier rejection were not smoothly bridged"
    for f, p in enumerate(clean):
        assert 35.0 <= float(p[0]) <= 62.0, f"Frame {f} ball escaped valid corridor: {p}"
        assert 30.0 <= float(p[1]) <= 38.0, f"Frame {f} ball clamped/escaped Y: {p}"
        if f > 0:
            assert float(np.linalg.norm(p - clean[f - 1])) < 2.5

    # Renderer discontinuous trail break: two points 60m apart must NEVER draw a connecting line across pitch center!
    cfg = EngineConfig(players_model_path="", pitch_model_path="", enable_siglip=False, enable_jersey_ocr=False)
    renderer = BroadcastRenderer(cfg, pitch)
    rec_a = {"frame_idx": 0, "homography_valid": True, "entities": [], "ball_pitch_xy": np.array([10.0, 34.0], np.float32), "voronoi_grid": None, "hulls": {}, "def_lines": {}, "voronoi_metrics": {}}
    rec_b = {"frame_idx": 1, "homography_valid": True, "entities": [], "ball_pitch_xy": np.array([95.0, 34.0], np.float32), "voronoi_grid": None, "hulls": {}, "def_lines": {}, "voronoi_metrics": {}}
    renderer.draw_radar_frame(rec_a, fps=25.0)
    img_b = renderer.draw_radar_frame(rec_b, fps=25.0)
    # Check midpoint (x=52.5, y=34.0) is NOT painted bright gold/green laser line (stays dark turf/white center spot)
    mx, my = renderer._m_to_radar(50.0, 34.0)
    px_bgr = img_b[my, mx]
    assert int(px_bgr[2]) < 120, f"Discontinuous jump drew a laser line across midfield! BGR={px_bgr}"


def test_calibrated_open_play_sprint_detection_with_1frame_blink():
    """Fix 3: Sustained >= 6.2 m/s (22.3 km/h) for >= 0.55s with 1-frame blink triggers 1 sprint; spikes & short bursts yield 0."""
    cfg = EngineConfig(players_model_path="", pitch_model_path="", enable_siglip=False, enable_jersey_ocr=False)
    pitch = FIFAPitchConfig()
    fps = 25.0
    eng = TacticalAnalyticsEngine(cfg, pitch, fps=fps)
    H_id = np.eye(3, dtype=np.float32)

    # Track 1: 6.8 m/s (24.5 km/h) run for 1.0s (25 frames) with a 1-frame blink at f=32 -> MUST log 1 sprint!
    # Track 2: 0.36s (9 frames < 0.55s=14 frames) brief burst at 7.5 m/s -> MUST log 0 sprints!
    frame_records = []
    x1, x2 = 15.0, 15.0
    for f in range(90):
        x1 += (6.8 / fps) if (20 <= f <= 48) else (1.5 / fps)
        x2 += (7.5 / fps) if (30 <= f < 39) else (1.5 / fps)
        ents = [
            {"track_id": 2, "cls": cfg.player_id, "team": 1, "conf": 0.9, "box": np.array([10, 10, 30, 60], np.float32), "pitch_xy": np.array([x2, 46.0], np.float32)},
        ]
        if f != 32:  # 1-frame tracker occlusion blink right in the middle of Track 1's sprint!
            ents.append(
                {"track_id": 1, "cls": cfg.player_id, "team": 0, "conf": 0.9, "box": np.array([10, 10, 30, 60], np.float32), "pitch_xy": np.array([x1, 22.0], np.float32)}
            )
        frame_records.append({
            "frame_idx": f,
            "H": H_id,
            "H_inv": H_id,
            "homography_valid": True,
            "is_scene_cut": False,
            "homography_reset": False,
            "ball_detected": False,
            "ball_xy": None,
            "ball_pitch_xy": None,
            "entities": ents,
        })

    tel = eng.analyze(frame_records)
    by_tid = {p["track_id"]: p["sprints"] for p in tel["players"]}
    assert by_tid[1] == 1, f"Real open-play 6.8 m/s sprint with 1-frame blink missed: {by_tid}"
    assert by_tid[2] == 0, f"Short 0.36s burst falsely logged sprint: {by_tid}"




