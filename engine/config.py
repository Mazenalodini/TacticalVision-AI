import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FIFAPitchConfig:
    """
    Standard FIFA Football Pitch (105m x 68m) mapped to the exact 32-keypoint
    topology of the trained YOLOv11-m-pose pitch model (0-indexed: 0..31).
    """
    length: float = 105.0             # X-axis in meters (left goal -> right goal)
    width: float = 68.0               # Y-axis in meters (top touchline -> bottom touchline)
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @property
    def vertices(self) -> np.ndarray:
        """Returns (32, 2) float32 array of real-world pitch coordinates in meters."""
        L = self.length
        W = self.width
        PBL = self.penalty_box_length
        PBW = self.penalty_box_width
        GBL = self.goal_box_length
        GBW = self.goal_box_width
        R = self.centre_circle_radius
        PSD = self.penalty_spot_distance

        pts = [
            (0.0, 0.0),                         # 0  (1)
            (0.0, (W - PBW) / 2.0),             # 1  (2)
            (0.0, (W - GBW) / 2.0),             # 2  (3)
            (0.0, (W + GBW) / 2.0),             # 3  (4)
            (0.0, (W + PBW) / 2.0),             # 4  (5)
            (0.0, W),                           # 5  (6)
            (GBL, (W - GBW) / 2.0),             # 6  (7)
            (GBL, (W + GBW) / 2.0),             # 7  (8)
            (PSD, W / 2.0),                     # 8  (9)
            (PBL, (W - PBW) / 2.0),             # 9  (10)
            (PBL, (W - GBW) / 2.0),             # 10 (11)
            (PBL, (W + GBW) / 2.0),             # 11 (12)
            (PBL, (W + PBW) / 2.0),             # 12 (13)
            (L / 2.0, 0.0),                     # 13 (14)
            (L / 2.0, W / 2.0 - R),             # 14 (15)
            (L / 2.0, W / 2.0 + R),             # 15 (16)
            (L / 2.0, W),                       # 16 (17)
            (L - PBL, (W - PBW) / 2.0),         # 17 (18)
            (L - PBL, (W - GBW) / 2.0),         # 18 (19)
            (L - PBL, (W + GBW) / 2.0),         # 19 (20)
            (L - PBL, (W + PBW) / 2.0),         # 20 (21)
            (L - PSD, W / 2.0),                 # 21 (22)
            (L - GBL, (W - GBW) / 2.0),         # 22 (23)
            (L - GBL, (W + GBW) / 2.0),         # 23 (24)
            (L, 0.0),                           # 24 (25)
            (L, (W - PBW) / 2.0),               # 25 (26)
            (L, (W - GBW) / 2.0),               # 26 (27)
            (L, (W + GBW) / 2.0),               # 27 (28)
            (L, (W + PBW) / 2.0),               # 28 (29)
            (L, W),                             # 29 (30)
            (L / 2.0 - R, W / 2.0),             # 30 (31)
            (L / 2.0 + R, W / 2.0),             # 31 (32)
        ]
        return np.array(pts, dtype=np.float32)

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """0-indexed line segments connecting pitch vertices for 2D radar rendering."""
        one_based = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (7, 8), (10, 11), (11, 12), (12, 13),
            (14, 15), (15, 16), (16, 17),
            (18, 19), (19, 20), (20, 21), (23, 24),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
            (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
        ]
        return [(a - 1, b - 1) for a, b in one_based]


# Standard Karun Singh 12x8 Expected Threat (xT) Grid (Left -> Right attacking direction)
# Rows: Y zones (0..7), Cols: X zones (0..11)
XT_GRID = np.array([
    [0.00638, 0.00780, 0.00845, 0.00978, 0.01126, 0.01248, 0.01474, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792],
    [0.00750, 0.00879, 0.00942, 0.01059, 0.01215, 0.01385, 0.01612, 0.01870, 0.02402, 0.02953, 0.04067, 0.04648],
    [0.00888, 0.00978, 0.01001, 0.01110, 0.01269, 0.01429, 0.01690, 0.01997, 0.02485, 0.03249, 0.04806, 0.06801],
    [0.00941, 0.01083, 0.01017, 0.01132, 0.01263, 0.01485, 0.01690, 0.01997, 0.02685, 0.03890, 0.06500, 0.10800],
    [0.00941, 0.01083, 0.01017, 0.01132, 0.01263, 0.01485, 0.01690, 0.01997, 0.02685, 0.03890, 0.06500, 0.10800],
    [0.00888, 0.00978, 0.01001, 0.01110, 0.01269, 0.01429, 0.01690, 0.01997, 0.02485, 0.03249, 0.04806, 0.06801],
    [0.00750, 0.00879, 0.00942, 0.01059, 0.01215, 0.01385, 0.01612, 0.01870, 0.02402, 0.02953, 0.04067, 0.04648],
    [0.00638, 0.00780, 0.00845, 0.00978, 0.01126, 0.01248, 0.01474, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792],
], dtype=np.float32)


def load_production_tuning(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load verified production calibration settings from artifacts/models_meta/models_registry.json
    and optimal_thresholds_v1.json, with safe deterministic defaults if metadata files are absent.
    """
    root = project_root or Path(__file__).resolve().parent.parent
    registry_path = root / "artifacts" / "models_meta" / "models_registry.json"
    tuning: Dict[str, Any] = {}
    if registry_path.exists():
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            tuning = data.get("production_tuning", {})
        except Exception:
            tuning = {}
    return tuning


@dataclass
class EngineConfig:
    players_model_path: str
    pitch_model_path: str
    thresholds_path: str = ""
    det_imgsz: int = 1280
    pitch_imgsz: int = 640
    # Class IDs from Dataset 1: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
    ball_id: int = 0
    goalkeeper_id: int = 1
    player_id: int = 2
    referee_id: int = 3
    # Video-optimized thresholds
    ball_conf: float = 0.22
    goalkeeper_conf: float = 0.48
    player_conf: float = 0.50
    referee_conf: float = 0.45
    # Calibrated Model 2 homography parameters (aligned with models_registry.json production_tuning)
    pitch_kpt_conf: float = 0.30
    pitch_stride: int = 2
    min_span_m: float = 15.0
    min_convex_hull_area_m2: float = 150.0
    nms_iou: float = 0.50
    homography_ema_alpha: float = 0.20
    max_homography_reproj_m: float = 2.5
    # Analytics & Kinematics calibration
    position_moving_avg_window: int = 7
    speed_window_frames: int = 10
    speed_median_filter: int = 5
    max_valid_speed_kmh: float = 35.5
    max_player_speed_kmh: float = 35.5
    sprint_threshold_kmh: float = 24.0
    possession_radius_m: float = 2.8
    min_pass_distance_m: float = 4.0
    min_owner_frames: int = 3
    # Model 4: Jersey Legibility Gate + PARSeq OCR
    enable_jersey_ocr: bool = True
    jersey_legibility_onnx_path: str = ""
    jersey_legibility_pt_path: str = ""
    jersey_legibility_threshold: float = 0.45
    jersey_min_track_ocr_conf: float = 0.18
    jersey_parseq_min_crop_conf: float = 0.35
    jersey_max_crops_per_track: int = 12
    jersey_torso_crop_box: Tuple[float, float, float, float] = (0.20, 0.18, 0.80, 0.56)
    # Visual theme (BGR for OpenCV)
    team_colors_bgr: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        0: (255, 200, 30),   # Team A: Electric Cyan/Sky Blue
        1: (85, 85, 255),    # Team B: Vibrant Coral Red
    })
    team_colors_hex: Dict[int, str] = field(default_factory=lambda: {
        0: "#1EC8FF",
        1: "#FF5555",
    })
    referee_color_bgr: Tuple[int, int, int] = (40, 215, 255)  # Amber/Gold
    ball_color_bgr: Tuple[int, int, int] = (60, 255, 130)     # Neon Mint

    def __post_init__(self) -> None:
        root = Path(__file__).resolve().parent.parent
        tuning = load_production_tuning(root)
        m2 = tuning.get("model_2_pitch_homography", {})
        kin = tuning.get("kinematics_smoothing", {})
        m4 = tuning.get("model_4_jersey_recognition", {})

        # Apply verified registry calibration while preserving explicit overrides
        if self.pitch_kpt_conf == 0.30 and "kpt_conf" in m2:
            self.pitch_kpt_conf = float(m2["kpt_conf"])
        if self.pitch_stride == 2 and "pitch_stride" in m2:
            self.pitch_stride = int(m2["pitch_stride"])
        if self.min_span_m == 15.0 and "min_span_m" in m2:
            self.min_span_m = float(m2["min_span_m"])
        if self.min_convex_hull_area_m2 == 150.0 and "min_convex_hull_area_m2" in m2:
            self.min_convex_hull_area_m2 = float(m2["min_convex_hull_area_m2"])
        if self.max_homography_reproj_m == 2.5 and "max_reproj_error_m" in m2:
            self.max_homography_reproj_m = float(m2["max_reproj_error_m"])
        if self.homography_ema_alpha == 0.20 and "ema_alpha" in m2:
            self.homography_ema_alpha = float(m2["ema_alpha"])

        if self.position_moving_avg_window == 7 and "position_moving_avg_window" in kin:
            self.position_moving_avg_window = int(kin["position_moving_avg_window"])
        if self.speed_window_frames == 10 and "speed_window_frames" in kin:
            self.speed_window_frames = int(kin["speed_window_frames"])
        if self.speed_median_filter == 5 and "speed_median_filter" in kin:
            self.speed_median_filter = int(kin["speed_median_filter"])
        if self.max_valid_speed_kmh == 35.5 and "max_valid_speed_kmh" in kin:
            self.max_valid_speed_kmh = float(kin["max_valid_speed_kmh"])
        self.max_player_speed_kmh = float(self.max_valid_speed_kmh)

        if self.jersey_legibility_threshold == 0.45 and "legibility_threshold" in m4:
            self.jersey_legibility_threshold = float(m4["legibility_threshold"])
        if self.jersey_min_track_ocr_conf == 0.18 and "min_track_ocr_conf" in m4:
            self.jersey_min_track_ocr_conf = float(m4["min_track_ocr_conf"])
        if self.jersey_parseq_min_crop_conf == 0.35 and "parseq_min_crop_conf" in m4:
            self.jersey_parseq_min_crop_conf = float(m4["parseq_min_crop_conf"])
        if self.jersey_max_crops_per_track == 12 and "max_crops_per_track" in m4:
            self.jersey_max_crops_per_track = int(m4["max_crops_per_track"])
        if "torso_crop_box" in m4 and len(m4["torso_crop_box"]) == 4:
            self.jersey_torso_crop_box = tuple(float(v) for v in m4["torso_crop_box"])  # type: ignore

        models_dir = Path(self.players_model_path).resolve().parent if self.players_model_path else (root / "models")
        if not models_dir.exists():
            models_dir = root / "models"
        if not self.jersey_legibility_onnx_path:
            pref = str(m4.get("preferred_legibility_file", "jersey_legibility_v1_int8.onnx"))
            cand = models_dir / pref
            if not cand.exists():
                cand = models_dir / "jersey_legibility_v1.onnx"
            self.jersey_legibility_onnx_path = str(cand)
        if not self.jersey_legibility_pt_path:
            self.jersey_legibility_pt_path = str(models_dir / "jersey_legibility_v1.pt")

        if not self.thresholds_path:
            v1_thresh = root / "artifacts" / "models_meta" / "optimal_thresholds_v1.json"
            legacy_thresh = root / "reports" / "optimal_thresholds.json"
            if v1_thresh.exists():
                self.thresholds_path = str(v1_thresh)
            elif legacy_thresh.exists():
                self.thresholds_path = str(legacy_thresh)
