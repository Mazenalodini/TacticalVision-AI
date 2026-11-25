/**
 * TacticalVision AI — TypeScript interfaces matching the exact
 * telemetry.json schema produced by engine/pipeline.py.
 *
 * All 8 top-level keys are covered:
 *   summary, players, passing_network, events,
 *   heatmaps, frames, match_info, engine_diagnostics
 */

// ── summary ──────────────────────────────────────────────────────

export interface TeamPairNum {
  team_0: number;
  team_1: number;
}

export interface ZonalControl {
  defensive_third: number;
  middle_third: number;
  attacking_third: number;
}

export interface PitchControlPct {
  team_0: number;
  team_1: number;
  team_0_zonal: ZonalControl;
}

export interface Summary {
  total_frames: number;
  fps: number;
  duration_s: number;
  possession_pct: TeamPairNum;
  pitch_control_pct: PitchControlPct;
  passes: TeamPairNum;
  turnovers: number;
  expected_threat_xt: TeamPairNum;
  avg_convex_hull_area_m2: TeamPairNum;
  top_speed_kmh: number;
}

// ── players ──────────────────────────────────────────────────────

export interface PlayerStat {
  track_id: number;
  team: 0 | 1;
  role: "player" | "goalkeeper";
  distance_m: number;
  max_speed_kmh: number;
  avg_speed_kmh: number;
  sprints: number;
  passes_completed: number;
  xt_generated: number;
  avg_pitch_xy: [number, number]; // [x_m, y_m]
  frames_visible: number;
}

// ── passing_network ──────────────────────────────────────────────

export interface PassingEdge {
  team: 0 | 1;
  from_id: number;
  to_id: number;
  count: number;
}

// ── events (discriminated union on `type`) ───────────────────────

export interface PassEvent {
  type: "pass";
  frame: number;
  time_s: number;
  team: 0 | 1;
  from_id: number;
  to_id: number;
  start_xy: [number, number];
  end_xy: [number, number];
  distance_m: number;
  xt_delta: number;
}

export interface SprintEvent {
  type: "sprint";
  frame: number;
  time_s: number;
  track_id: number;
  team: 0 | 1;
  speed_kmh: number;
}

export interface TurnoverEvent {
  type: "turnover";
  frame: number;
  time_s: number;
  from_team: 0 | 1;
  to_team: 0 | 1;
  by_id: number;
  xy: [number, number];
}

export type MatchEvent = PassEvent | SprintEvent | TurnoverEvent;

// ── heatmaps ─────────────────────────────────────────────────────

export interface Heatmaps {
  team_0: number[][]; // int[8][12] — frame-counts per zone
  team_1: number[][]; // int[8][12]
}

// ── frames (per-frame timeline — drives 2D Canvas) ───────────────

export interface FramePlayer {
  id: number; // track_id
  team: 0 | 1;
  role: "p" | "gk"; // ⚠️ abbreviated vs PlayerStat.role
  x: number; // pitch X (0–105 m)
  y: number; // pitch Y (0–68 m)
  spd: number; // km/h
}

export interface TimelineFrame {
  f: number; // frame index (0-based)
  t: number; // time in seconds
  players: FramePlayer[];
  ball: [number, number] | null; // [x_m, y_m] or null when untracked
  vor: [number, number]; // [team_0_pct, team_1_pct]
  def_x: Record<string, number>; // ⚠️ keys are STRINGS "0"/"1" (JSON serialization)
}

// ── match_info ───────────────────────────────────────────────────

export interface MatchInfo {
  source_video: string;
  resolution: [number, number];
  fps: number;
  pitch_dimensions_m: [number, number]; // [105, 68]
  team_colors: Record<string, string>; // {"0":"#1EC8FF", "1":"#FF5555"}
}

// ── engine_diagnostics ───────────────────────────────────────────

export interface HomographySummary {
  valid_frames: number;
  rejected_frames: number;
  stability_rate_pct: number;
  mean_inlier_reproj_error_m: number | null;
}

export interface EngineDiagnostics {
  homography: HomographySummary;
  ball_coverage_pct: number;
  processing_time_s: number;
}

// ═══════════════════════════════════════════════════════════════════
// Complete Telemetry — all 8 top-level keys
// ═══════════════════════════════════════════════════════════════════

export interface Telemetry {
  summary: Summary;
  players: PlayerStat[];
  passing_network: PassingEdge[];
  events: MatchEvent[];
  heatmaps: Heatmaps;
  frames: TimelineFrame[];
  match_info: MatchInfo;
  engine_diagnostics: EngineDiagnostics;
}

// ═══════════════════════════════════════════════════════════════════
// API-specific models
// ═══════════════════════════════════════════════════════════════════

export interface MatchCard {
  id: string;
  title: string;
  duration_s: number;
  fps: number;
  resolution: [number, number];
  possession_pct: TeamPairNum;
  pitch_control_pct: TeamPairNum;
  passes: TeamPairNum;
  turnovers: number;
  top_speed_kmh: number;
  expected_threat_xt: TeamPairNum;
  events_count: number;
  players_count: number;
  ball_coverage_pct: number;
  homography_stability_pct: number;
  has_thumbnail: boolean;
  is_user_uploaded: boolean;
  source_video: string;
}

export type JobStage =
  | "queued"
  | "detecting"
  | "classifying"
  | "computing"
  | "encoding"
  | "completed"
  | "failed"
  | "cancelled";

export interface JobStatus {
  job_id: string;
  status: JobStage;
  progress_pct: number;
  stage_label: string;
  elapsed_s: number;
  error?: string | null;
  match_id?: string | null;
  pid?: number | null;
}

// ═══════════════════════════════════════════════════════════════════
// Constants (matching engine/config.py)
// ═══════════════════════════════════════════════════════════════════

export const PITCH_LENGTH = 105; // meters
export const PITCH_WIDTH = 68;   // meters
export const SPRINT_THRESHOLD_KMH = 24.0;

export const TEAM_COLORS = {
  0: "#38BDF8", // Broadcast Sky        (Team A)
  1: "#FB7185", // Luminous Rose-Coral  (Team B)
} as const;

export const BALL_COLOR = "#34D399";    // Broadcast Mint
export const REFEREE_COLOR = "#F59E0B"; // Tactical Amber
