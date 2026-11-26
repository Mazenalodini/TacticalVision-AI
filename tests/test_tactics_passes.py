from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.config import EngineConfig, FIFAPitchConfig
from engine.tactics import (
    bridge_ball_pitch_trajectory,
    extract_possession_and_passes,
    is_progressive_pass,
)

ROOT = Path(__file__).resolve().parent.parent


def test_ball_trajectory_bridging_fills_short_gaps():
    # 5.0m over 4 frames @ 25fps = 31.25 m/s (plausible fast pass <= 38.0 m/s)
    raw = [
        np.array([20.0, 30.0], dtype=np.float32),
        None,
        None,
        None,
        np.array([24.0, 33.0], dtype=np.float32),
    ]
    bridged = bridge_ball_pitch_trajectory(raw, fps=25.0, max_gap_frames=8, max_speed_mps=38.0)
    assert all(p is not None for p in bridged)
    assert np.allclose(bridged[2], [22.0, 31.5], atol=0.05)


def test_progressive_pass_classification():
    # Team 0 attacks Left -> Right
    assert is_progressive_pass([30.0, 34.0], [55.0, 34.0], team=0) is True
    assert is_progressive_pass([60.0, 20.0], [72.0, 20.0], team=0) is True
    assert is_progressive_pass([60.0, 20.0], [64.0, 20.0], team=0) is False
    # Team 1 attacks Right -> Left
    assert is_progressive_pass([60.0, 34.0], [42.0, 34.0], team=1) is True
    assert is_progressive_pass([40.0, 20.0], [45.0, 20.0], team=1) is False


def test_symmetric_pass_detection_with_defender_brush_hysteresis():
    cfg = EngineConfig(players_model_path="", pitch_model_path="")
    pitch = FIFAPitchConfig()
    fps = 25.0

    # Simulate Team 0 player 1 passing 12m forward to Team 0 player 2,
    # with a 3-frame ball dropout mid-flight and a brief 2-frame brush near defender 99.
    frames_players = []
    raw_ball = []
    for f in range(30):
        frames_players.append([
            (1, 0, 30.0, 34.0),
            (2, 0, 42.0, 34.0),
            (99, 1, 36.0, 32.2),  # 1.8m off pass line -> nearest for ~2 frames without trapping
        ])
        if f < 8:
            raw_ball.append(np.array([30.2, 34.0], dtype=np.float32))
        elif 8 <= f <= 9:
            raw_ball.append(None)  # fast kick gap
        elif 10 <= f <= 13:
            bx = 30.2 + (f - 7) * 1.9
            raw_ball.append(np.array([bx, 34.0], dtype=np.float32))
        else:
            raw_ball.append(np.array([41.8, 34.0], dtype=np.float32))

    bridged = bridge_ball_pitch_trajectory(raw_ball, fps=fps)
    res = extract_possession_and_passes(frames_players, bridged, fps, cfg, pitch)
    assert len(res["passes"]) == 1
    assert len(res["interceptions"]) == 0
    p = res["passes"][0]
    assert p["team"] == 0
    assert p["from_id"] == 1 and p["to_id"] == 2
    assert p["distance_m"] >= 10.0


def test_match_03_press_produces_both_teams_passes():
    tel_path = ROOT / "outputs" / "match_03_press" / "telemetry.json"
    assert tel_path.exists()
    data = json.loads(tel_path.read_text(encoding="utf-8"))
    assert data["summary"]["passes"]["team_0"] > 0
    assert data["summary"]["passes"]["team_1"] > 0
    edges_t0 = [e for e in data["passing_network"] if e["team"] == 0]
    edges_t1 = [e for e in data["passing_network"] if e["team"] == 1]
    assert len(edges_t0) > 0 and len(edges_t1) > 0
