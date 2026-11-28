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
  jersey_number?: number | null;
  jersey_confidence?: number | null;
  jersey_verified?: boolean;
  legibility_score?: number | null;
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
  is_progressive?: boolean;
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
  role: "p" | "gk" | "referee"; // ⚠️ abbreviated vs PlayerStat.role
  x: number; // pitch X (0–105 m)
  y: number; // pitch Y (0–68 m)
  spd: number; // km/h
  jn?: number | null; // Model 4 verified jersey number
  jv?: boolean | null; // Model 4 verified lock flag
}

export interface TimelineFrame {
  f: number; // frame index (0-based)
  t: number; // time in seconds
  hv?: boolean; // true when pitch homography is geometrically valid (false during close-ups/replays)
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
  team_a_kit_hex?: string;
  team_b_kit_hex?: string;
}

// ── engine_diagnostics ───────────────────────────────────────────

export interface HomographySummary {
  valid_frames: number;
  rejected_frames: number;
  stability_rate_pct: number;
  mean_inlier_reproj_error_m: number | null;
  flow_propagated_frames?: number;
  calibrator_backend?: string;
}

export interface JerseyRecognitionDiagnostics {
  gate_enabled?: boolean;
  gate_backend?: string;
  ocr_available?: boolean;
  ocr_status?: string;
  tracks_evaluated?: number;
  legible_crops?: number;
  verified_players?: number;
}

export interface EngineDiagnostics {
  homography: HomographySummary;
  calibrator_backend?: string;
  team_classifier?: Record<string, unknown> | null;
  ball_coverage_pct: number;
  processing_time_s: number;
  jersey_recognition?: JerseyRecognitionDiagnostics | null;
  broadcast_spotting_ms?: number | null;
  broadcast_spotting_backend?: string | null;
}

// ═══════════════════════════════════════════════════════════════════
// Complete Telemetry — all 8 core keys + Model 3 broadcast_events
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
  broadcast_events?: ActionSpottingEvent[];
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

// ═══════════════════════════════════════════════════════════════════
// 4-Model AI Suite & Model 3 Action Spotting Schemas
// ═══════════════════════════════════════════════════════════════════

export interface DiskArtifactInfo {
  filename: string;
  exists: boolean;
  size_mb: number | null;
}

export interface ModelsSuiteResponse {
  platform: string;
  schema_version: string;
  generated_at?: string | null;
  models: Record<string, Record<string, unknown>>;
  production_tuning: Record<string, Record<string, unknown>>;
  metrics: {
    project?: string;
    generated_at?: string;
    model_1_players_detector?: {
      model: string;
      imgsz: number;
      overall_mAP50: number;
      overall_mAP50_95: number;
      per_class_map: Record<string, { mAP50: number; mAP50_95: number }>;
      per_class_calibrated_iou50: Record<
        string,
        { class_id: number; tp: number; fp: number; fn: number; precision: number; recall: number; f1: number }
      >;
      ball_missed_val_frames?: number;
    };
    model_1_coco_baseline?: {
      model: string;
      imgsz: number;
      per_class_iou50: Record<
        string,
        { class_id: number; tp: number; fp: number; fn: number; precision: number; recall: number; f1: number }
      >;
    };
    model_2_pitch_keypoints?: {
      pose_mAP50: number;
      pose_mAP50_95: number;
      mean_pixel_error_px: number;
      median_pixel_error_px: number;
      pck_5px_pct: number;
      pck_10px_pct: number;
      pck_20px_pct: number;
      homography_ready_frames_pct: number;
      mean_ransac_reproj_error_px: number;
    };
    model_3_action_spotting?: {
      architecture: string;
      params_millions: number;
      best_epoch: number;
      test_metrics: {
        tight_amap_pct: number;
        loose_amap_pct: number;
        map_at_1s_pct: number;
        map_at_5s_pct: number;
        per_class_tight_ap_pct: Record<string, number>;
      };
    };
    model_4_jersey_recognition?: {
      backbone: string;
      optimal_threshold: number;
      legibility_test_metrics: {
        threshold: number;
        crop_level: {
          roc_auc_pct: number;
          pr_auc_pct: number;
          f1_pct: number;
          precision_pct: number;
          recall_pct: number;
        };
        tracklet_level: {
          roc_auc_pct: number;
          f1_pct: number;
          precision_pct: number;
          recall_pct: number;
          confusion_matrix: number[][];
        };
      };
      parseq_ablation: {
        raw_parseq_no_gate: {
          overall_acc_pct: number;
          legible_acc_pct: number;
          illegible_rejection_pct: number;
        };
        full_gated_pipeline: {
          overall_acc_pct: number;
          legible_acc_pct: number;
          illegible_rejection_pct: number;
        };
      };
    };
  };
  cpu_onnx_benchmarks: {
    model_1_players_detector_onnx?: { size_mb: number; cpu_latency_ms: number; cpu_fps: number };
    model_2_pitch_keypoints_onnx?: { size_mb: number; cpu_latency_ms: number; cpu_fps: number };
    model_3_action_spotting_onnx?: {
      size_mb: number;
      cpu_latency_ms: number;
      cpu_fps: number;
      pytorch_onnx_max_abs_diff: number;
      int8_cpu_benchmark: { size_mb: number; cpu_latency_ms: number; cpu_fps: number };
    };
    model_4_jersey_legibility_onnx?: {
      size_mb: number;
      cpu_latency_ms: number;
      cpu_fps: number;
      pytorch_onnx_max_abs_diff: number;
      optimal_threshold: number;
      int8_cpu_benchmark: { size_mb: number; cpu_latency_ms: number; cpu_fps: number };
    };
  };
  optimal_thresholds?: Record<string, unknown>;
  disk_inventory: Record<string, Record<string, DiskArtifactInfo>>;
  runtime: {
    onnxruntime_available: boolean;
    torch_available: boolean;
    timm_available: boolean;
    strhub_ocr_available: boolean;
    model_3_preferred_backend: string;
    model_4_gate_preferred_backend: string;
    demo_features_available: boolean;
  };
}

export type ActionCategory =
  | "scoring"
  | "shot"
  | "set_piece"
  | "foul"
  | "card"
  | "tactical"
  | "flow";

export interface ActionSpottingEvent {
  class_id: number;
  label: string;
  category: ActionCategory | string;
  half: number;
  frame_idx: number;
  time_s: number;
  minute_str: string;
  confidence: number;
}

export interface ActionSpottingDensityBin {
  bin_index: number;
  label: string;
  start_s: number;
  end_s: number;
  count: number;
  high_impact_count: number;
}

export interface ActionSpottingHalfSummary {
  half: number;
  num_frames: number;
  fps: number;
  duration_s: number;
  inference_ms: number;
  backend_used: string;
  events_count: number;
}

export interface ActionSpottingGTComparison {
  tolerance_sec: number;
  gt_total: number;
  matched_gt: number;
  predicted_total: number;
  precision_pct: number;
  recall_pct: number;
}

export interface ActionSpottingGroundTruthEvent {
  frame_idx: number;
  time_s: number;
  class_id: number;
  label: string;
  visibility?: string;
  half?: number;
}

export interface ActionSpottingAnalyzeResponse {
  run_id: string;
  match_id: string;
  match_title: string;
  half_mode: string;
  fps: number;
  min_conf: number;
  nms_window_sec: number;
  total_inference_ms: number;
  backend_used: string;
  halves: ActionSpottingHalfSummary[];
  events: ActionSpottingEvent[];
  class_counts: Record<string, number>;
  category_counts: Record<string, number>;
  density_bins: ActionSpottingDensityBin[];
  ground_truth_events?: ActionSpottingGroundTruthEvent[] | null;
  gt_comparison?: ActionSpottingGTComparison | null;
  created_at: string;
}

export interface ActionSpottingDemoMatch {
  match_id: string;
  title: string;
  league_season_path?: string | null;
  fps: number;
  feature_dim: number;
  halves: { half1_frames: number; half2_frames: number };
  gt_event_counts: { half1: number; half2: number; total: number };
  features_ready: boolean;
  features_size_mb: number | null;
}

export interface ActionSpottingMatchesResponse {
  demo_matches: ActionSpottingDemoMatch[];
  sample_timeline: {
    match?: string;
    half?: number | string;
    cpu_inference_sec?: number;
    detected_events_count?: number;
  } | null;
  saved_runs: Array<{
    run_id: string;
    match_id: string;
    match_title: string;
    half_mode: string;
    backend_used: string;
    total_inference_ms: number;
    events_count: number;
    created_at: string;
  }>;
}
