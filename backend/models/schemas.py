"""Pydantic v2 response / request schemas matching the exact telemetry.json contract.

All 8 top-level keys of telemetry.json are covered:
  summary, players, passing_network, events,
  heatmaps, frames, match_info, engine_diagnostics
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── summary sub-models ──────────────────────────────────────────────

class TeamPairFloat(BaseModel):
    team_0: float
    team_1: float


class TeamPairInt(BaseModel):
    team_0: int
    team_1: int


class ZonalControl(BaseModel):
    defensive_third: float
    middle_third: float
    attacking_third: float


class PitchControlPct(BaseModel):
    team_0: float
    team_1: float
    team_0_zonal: ZonalControl


class Summary(BaseModel):
    total_frames: int
    fps: float
    duration_s: float
    possession_pct: TeamPairFloat
    pitch_control_pct: PitchControlPct
    passes: TeamPairInt
    turnovers: int
    expected_threat_xt: TeamPairFloat
    avg_convex_hull_area_m2: TeamPairFloat
    top_speed_kmh: float


# ── players ─────────────────────────────────────────────────────────

class PlayerStat(BaseModel):
    track_id: int
    team: int
    role: str                        # "player" | "goalkeeper"
    distance_m: float
    max_speed_kmh: float
    avg_speed_kmh: float
    sprints: int
    passes_completed: int
    xt_generated: float
    avg_pitch_xy: list[float]        # [x, y] meters
    frames_visible: int


# ── passing_network ─────────────────────────────────────────────────

class PassingEdge(BaseModel):
    team: int
    from_id: int
    to_id: int
    count: int


# ── events (discriminated union on `type`) ──────────────────────────

class PassEvent(BaseModel):
    type: Literal["pass"]
    frame: int
    time_s: float
    team: int
    from_id: int
    to_id: int
    start_xy: list[float]
    end_xy: list[float]
    distance_m: float
    xt_delta: float


class SprintEvent(BaseModel):
    type: Literal["sprint"]
    frame: int
    time_s: float
    track_id: int
    team: int
    speed_kmh: float


class TurnoverEvent(BaseModel):
    type: Literal["turnover"]
    frame: int
    time_s: float
    from_team: int
    to_team: int
    by_id: int
    xy: list[float]


Event = Annotated[
    Union[PassEvent, SprintEvent, TurnoverEvent],
    Field(discriminator="type"),
]


# ── heatmaps ────────────────────────────────────────────────────────

class Heatmaps(BaseModel):
    team_0: list[list[int]]          # int[8][12]
    team_1: list[list[int]]


# ── frames ──────────────────────────────────────────────────────────

class FramePlayer(BaseModel):
    id: int
    team: int
    role: str                        # "p" | "gk"
    x: float
    y: float
    spd: float


class TimelineFrame(BaseModel):
    f: int
    t: float
    players: list[FramePlayer]
    ball: Optional[list[float]] = None   # [x, y] or null
    vor: list[float]                     # [team_0_pct, team_1_pct]
    def_x: dict[str, float]             # keys are strings "0" / "1"


# ── match_info ──────────────────────────────────────────────────────

class MatchInfo(BaseModel):
    source_video: str
    resolution: list[int]
    fps: float
    pitch_dimensions_m: list[float]
    team_colors: dict[str, str]      # {"0": "#1EC8FF", "1": "#FF5555"}


# ── engine_diagnostics ──────────────────────────────────────────────

class HomographySummary(BaseModel):
    valid_frames: int
    rejected_frames: int
    stability_rate_pct: float
    mean_inlier_reproj_error_m: Optional[float] = None


class EngineDiagnostics(BaseModel):
    homography: HomographySummary
    ball_coverage_pct: float
    processing_time_s: float


# ══════════════════════════════════════════════════════════════════════
# Complete Telemetry — all 8 top-level keys
# ══════════════════════════════════════════════════════════════════════

class Telemetry(BaseModel):
    summary: Summary
    players: list[PlayerStat]
    passing_network: list[PassingEdge]
    events: list[Event]
    heatmaps: Heatmaps
    frames: list[TimelineFrame]
    match_info: MatchInfo
    engine_diagnostics: EngineDiagnostics


# ══════════════════════════════════════════════════════════════════════
# API-specific models
# ══════════════════════════════════════════════════════════════════════

class MatchCard(BaseModel):
    id: str
    title: str
    duration_s: float
    fps: float
    resolution: list[int]
    possession_pct: dict[str, float]
    pitch_control_pct: dict[str, float]
    passes: dict[str, int]
    turnovers: int
    top_speed_kmh: float
    expected_threat_xt: dict[str, float]
    events_count: int
    players_count: int
    ball_coverage_pct: float
    homography_stability_pct: float
    has_thumbnail: bool
    is_user_uploaded: bool
    source_video: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal[
        "queued", "detecting", "classifying",
        "computing", "encoding", "completed", "failed", "cancelled",
    ]
    progress_pct: float
    stage_label: str
    elapsed_s: float
    error: Optional[str] = None
    match_id: Optional[str] = None
    pid: Optional[int] = None
