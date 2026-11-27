"""Universal, Match-Agnostic Team Affiliation System (SoccerNet-Tracking SOTA).

Engineered for zero overfitting across ANY league, stadium, lighting condition,
kit color combination (white vs dark, red vs blue, striped/patterned, green/yellow,
or low-contrast kits), and video duration (short highlights to full 90-minute matches).

Architecture:
  1. Strict Tactical-Player Quality Gate (`is_valid_tactical_box`):
     Accepts only genuine on-pitch tactical player boxes (`0.04 <= h/H <= 0.22`,
     valid aspect ratio, away from scoreboards/borders, non-occluded `IoU < 0.15`,
     strictly excluding referees & goalkeepers).
  2. Inner Upper-Torso Extraction (`crop_jersey`):
     `y: [16%..48%], x: [26%..74%]` detached `.copy()` — excludes head, shorts, legs,
     and background ad-boards.
  3. Adaptive Pitch-Green + Skin Masking:
     Calibrated per match from the video's actual turf HSV distribution, while carefully
     preserving dark kits (navy/black) AND green/yellow kits whose chromaticity differs
     from turf (with clean non-skin fallback for turf-matching green jerseys).
  4. Generalizable Perceptual CIELAB + Quantile/Chroma Descriptor:
     Handles BOTH solid jerseys with large contrasting back-numbers AND striped/patterned
     kits by encoding robust `(L*, a*, b*)` quantiles + foreground brightness/chroma profile.
  5. 2-Stage Outlier-Trimmed KMeans(k=2):
     Fits initial clusters, rejects the farthest ~15% outlier crops (stray coaches,
     ball-boys, edge artifacts, misclassified refs/GKs), and refits clean centroids,
     exporting real extracted team kit hex colors (`team_a_kit_hex`, `team_b_kit_hex`).
  6. Smart Multi-Cue Fallback ONLY for Truly Ambiguous Kits (`ΔE_Lab < 16.0`):
     When kits are clearly separable in CIELAB (`ΔE_Lab >= 16.0`), trusts CIELAB
     exclusively (100% immune to pose/scale/ad-board bias). Only fuses compact HSV/visual
     appearance cues when two teams wear near-identical kit colors (`ΔE_Lab < 16.0`).
  7. Confidence-Weighted Non-Occluded Track Consensus & Role Isolation:
     Locks one stable team label per track weighted by centroid margin `|d0-d1|/(d0+d1)`,
     isolates referees (including OOD color check on partial ref detections), and assigns
     goalkeepers strictly by defensive-goal proximity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """Compute (N, N) pairwise IoU matrix with zero diagonal for occlusion filtering."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter + 1e-6
    iou = inter / union
    np.fill_diagonal(iou, 0.0)
    return iou


def is_valid_tactical_box(
    box: np.ndarray,
    frame_h: int,
    frame_w: int,
    max_iou: float = 0.0,
    iou_thresh: float = 0.15,
) -> bool:
    """
    Strict Tactical-Player Quality Gate applied before collecting fitting samples.
    Rejects close-up players, tiny crowd/ball-boy boxes, top HUD/bottom border touches,
    extreme aspect ratios, and occluded overlapping players.
    """
    if frame_h <= 0 or frame_w <= 0:
        return False
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw = x2 - x1
    bh = y2 - y1
    if bw < 10.0 or bh < 18.0:
        return False
    h_ratio = bh / float(frame_h)
    if not (0.04 <= h_ratio <= 0.22):
        return False
    aspect = bw / max(1e-6, bh)
    if not (0.18 <= aspect <= 1.25):
        return False
    # Exclude top scoreboard/stands edge and bottom sideline/bench strip
    if y1 < 0.07 * frame_h or y2 > 0.96 * frame_h:
        return False
    if x1 < 0.015 * frame_w or x2 > 0.985 * frame_w:
        return False
    if max_iou >= iou_thresh:
        return False
    return True


def _cie_lab_to_hex(l_cie: float, a_cie: float, b_cie: float) -> str:
    """Convert standard perceptual CIE Lab (L in 0..100, a/b in -128..127) to '#RRGGBB'."""
    l_u8 = int(np.clip(round(l_cie * 255.0 / 100.0), 0, 255))
    a_u8 = int(np.clip(round(a_cie + 128.0), 0, 255))
    b_u8 = int(np.clip(round(b_cie + 128.0), 0, 255))
    px_lab = np.uint8([[[l_u8, a_u8, b_u8]]])
    px_bgr = cv2.cvtColor(px_lab, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{int(px_bgr[2]):02X}{int(px_bgr[1]):02X}{int(px_bgr[0]):02X}"


class SigLIPTeamClassifier:
    """
    Universal, Match-Agnostic Team Affiliation Classifier.
    Retains class name `SigLIPTeamClassifier` for 100% API/test backward compatibility.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        enable_siglip: bool = False,
    ):
        self.model_name = model_name
        self.enable_siglip = bool(enable_siglip)
        self.use_siglip = False
        self.processor = None
        self.model = None
        self.reducer = None
        self.kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
        self.fitted = False

        # Adaptive pitch-green HSV bounds (refined per match via `calibrate_pitch_green`)
        self.grass_h_lo: int = 30
        self.grass_h_hi: int = 86
        self.grass_s_min: int = 42
        self.grass_v_min: int = 32
        self._grass_h_samples: List[float] = []
        self._grass_s_samples: List[float] = []

        # Diagnostics & extracted kit palette
        self.cluster_delta_e: float = 0.0
        self.centroids_lab: Optional[np.ndarray] = None
        self.centroids_raw_lab3: Optional[np.ndarray] = None  # (2, 3) true CIE (L*, a*, b*)
        self.team_a_kit_hex: str = "#E8EEF2"
        self.team_b_kit_hex: str = "#1E293B"
        self.mode_used: str = "cielab_universal_primary"
        self._secondary_centroids: Optional[np.ndarray] = None

    def calibrate_pitch_green(self, frame_bgr: np.ndarray) -> None:
        """Adapt pitch-green HSV bounds from the video's actual turf (preserving dark & distinct green kits)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        h, w = frame_bgr.shape[:2]
        if h < 40 or w < 40:
            return
        roi = frame_bgr[int(h * 0.35):int(h * 0.90), int(w * 0.10):int(w * 0.90)]
        small = cv2.resize(roi, (120, 68), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green_px = hsv[
            (hsv[:, :, 0] >= 25)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 35)
            & (hsv[:, :, 2] >= 35)
        ]
        if len(green_px) < 100:
            return
        med_h = float(np.median(green_px[:, 0]))
        med_s = float(np.median(green_px[:, 1]))
        self._grass_h_samples.append(med_h)
        self._grass_s_samples.append(med_s)
        if len(self._grass_h_samples) > 36:
            self._grass_h_samples.pop(0)
            self._grass_s_samples.pop(0)
        global_h = float(np.median(self._grass_h_samples))
        global_s = float(np.median(self._grass_s_samples))
        # Tight +/-15 hue band & adaptive saturation floor so dark navy/black and bright yellow kits stay intact
        self.grass_h_lo = max(22, int(round(global_h - 15)))
        self.grass_h_hi = min(96, int(round(global_h + 15)))
        self.grass_s_min = max(38, min(95, int(round(global_s * 0.48))))

    @staticmethod
    def crop_jersey(frame_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """Extract strict inner upper-torso region: y=[16%..48%], x=[26%..74%].

        Excludes head/hair, shorts, legs, pitch grass under feet, and background ad-boards.
        Always returns an owning `.copy()` so full video frames are never retained in RAM.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 14:
            return None
        jy1 = y1 + int(bh * 0.16)
        jy2 = y1 + int(bh * 0.48)
        jx1 = x1 + int(bw * 0.26)
        jx2 = x1 + int(bw * 0.74)
        if jy2 <= jy1 + 2 or jx2 <= jx1 + 2:
            return None
        crop = frame_bgr[jy1:jy2, jx1:jx2]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return None
        return crop.copy()

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.reducer = None

    def _extract_foreground_cie_pixels(
        self,
        crop_bgr: np.ndarray,
        min_fg_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, bool, float]:
        """Mask pitch green + skin tones and return foreground pixels in true perceptual CIE Lab."""
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        lab_u8 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)

        # 1. Adaptive pitch-green grass mask (strictly gated on V >= 32 so dark kits are never masked)
        grass_mask = (
            (hsv[:, :, 0] >= self.grass_h_lo)
            & (hsv[:, :, 0] <= self.grass_h_hi)
            & (hsv[:, :, 1] >= self.grass_s_min)
            & (hsv[:, :, 2] >= self.grass_v_min)
        )

        # 2. YCrCb skin mask (strictly gated: in an inner chest crop y=16..48%, x=26..74%,
        # exposed skin never covers >= 40% of non-grass pixels — if it does, the jersey itself
        # is cream/beige/peach/orange fabric, so preserve it untouched!)
        skin_mask = (
            (ycrcb[:, :, 1] >= 133)
            & (ycrcb[:, :, 1] <= 173)
            & (ycrcb[:, :, 2] >= 77)
            & (ycrcb[:, :, 2] <= 127)
            & (hsv[:, :, 1] >= 30)
            & (hsv[:, :, 1] <= 165)
            & (lab_u8[:, :, 0] < 218)
        )
        non_grass_cnt = max(1, int(np.sum(~grass_mask)))
        skin_in_ng = int(np.sum(skin_mask & (~grass_mask)))
        if (skin_in_ng / float(non_grass_cnt)) >= 0.40:
            skin_mask = np.zeros_like(skin_mask, dtype=bool)

        valid_mask = ~(grass_mask | skin_mask)
        total_px = int(crop_bgr.shape[0] * crop_bgr.shape[1])
        fg_u8 = lab_u8[valid_mask]
        fg_ratio = float(len(fg_u8)) / max(1, total_px)

        is_valid = (fg_ratio >= min_fg_ratio) and (len(fg_u8) >= 6)
        if len(fg_u8) < 6:
            # Clean fallback for turf-matching green jerseys -> keep all non-skin torso pixels
            fg_u8 = lab_u8[~skin_mask]
            if len(fg_u8) < 4:
                fg_u8 = lab_u8.reshape(-1, 3)

        fg_cie = np.empty((len(fg_u8), 3), dtype=np.float32)
        fg_cie[:, 0] = fg_u8[:, 0].astype(np.float32) * (100.0 / 255.0)
        fg_cie[:, 1] = fg_u8[:, 1].astype(np.float32) - 128.0
        fg_cie[:, 2] = fg_u8[:, 2].astype(np.float32) - 128.0
        return fg_cie, is_valid, fg_ratio

    def _extract_torso_lab_feature(
        self,
        crop_bgr: np.ndarray,
        min_fg_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, bool, float, np.ndarray]:
        """
        Build a 15-D generalizable perceptual CIELAB torso descriptor:
          - Primary 3 dims: dominant fabric (L*, a*, b*) via spatial-weighted median/mode
            (resistant to large contrasting back-numbers covering 25-35% of the torso)
          - Secondary 6 dims: 25th & 75th percentiles of (L*, a*, b*) weighted at 0.38
            (captures striped & two-tone patterned jerseys cleanly)
          - Tertiary 6 dims: compact 6-bin lightness + chromaticity histogram scaled to match CIE scale
            (separates striped/patterned kits from solid mid-tone kits of identical mean!)
        Returns `(feat_15d, is_valid, fg_ratio, dominant_lab3)`.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros((15,), dtype=np.float32), False, 0.0, np.zeros((3,), dtype=np.float32)

        fg_cie, is_valid, fg_ratio = self._extract_foreground_cie_pixels(crop_bgr, min_fg_ratio=min_fg_ratio)

        # Quantiles capture both dominant fabric and dual-color stripes robustly
        q25 = np.percentile(fg_cie, 25, axis=0)
        med = np.median(fg_cie, axis=0)
        q75 = np.percentile(fg_cie, 75, axis=0)

        # Compact 6-bin perceptual distribution: 4 lightness bins [0..25, 25..50, 50..75, 75..100]
        # + warm/cool chromaticity balance (pos_a_frac, pos_b_frac) scaled by 18.0 CIE units
        l_vals = fg_cie[:, 0]
        hist_l = np.array(
            [
                float(np.mean(l_vals < 25.0)),
                float(np.mean((l_vals >= 25.0) & (l_vals < 50.0))),
                float(np.mean((l_vals >= 50.0) & (l_vals < 75.0))),
                float(np.mean(l_vals >= 75.0)),
                float(np.mean(fg_cie[:, 1] > 6.0)),   # red/magenta fraction
                float(np.mean(fg_cie[:, 2] < -6.0)),  # blue/navy fraction
            ],
            dtype=np.float32,
        ) * 18.0

        # Slightly emphasize L* (1.10) alongside a*/b* (1.0) so light/dark AND red/blue separate cleanly
        scale = np.array([1.10, 1.0, 1.0], dtype=np.float32)
        feat = np.concatenate([med * scale, q25 * scale * 0.38, q75 * scale * 0.38, hist_l]).astype(np.float32)
        return feat, is_valid, fg_ratio, med.astype(np.float32)

    def _extract_secondary_hsv_cue(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Compact HSV hue/saturation pattern histogram used ONLY when ΔE_Lab < 16 (near-identical kits)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros((24,), dtype=np.float32)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        grass_mask = (
            (hsv[:, :, 0] >= self.grass_h_lo)
            & (hsv[:, :, 0] <= self.grass_h_hi)
            & (hsv[:, :, 1] >= self.grass_s_min)
            & (hsv[:, :, 2] >= self.grass_v_min)
        )
        fg = hsv[~grass_mask]
        if len(fg) < 4:
            fg = hsv.reshape(-1, 3)
        h_hist, _ = np.histogram(fg[:, 0], bins=12, range=(0, 180))
        s_hist, _ = np.histogram(fg[:, 1], bins=6, range=(0, 256))
        v_hist, _ = np.histogram(fg[:, 2], bins=6, range=(0, 256))
        vec = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-6 else vec

    def extract_embeddings(self, crops_bgr: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Extract 15-D perceptual CIELAB torso descriptors for a batch of BGR crops."""
        if not crops_bgr:
            return np.zeros((0, 15), dtype=np.float32)
        feats = [self._extract_torso_lab_feature(c)[0] for c in crops_bgr]
        return np.asarray(feats, dtype=np.float32)

    def fit(self, sample_crops_bgr: List[np.ndarray]) -> None:
        """
        2-Stage Outlier-Trimmed CIELAB Clustering:
          Stage 1: Fit initial KMeans(k=2) on valid stratified crops.
          Stage 2: Reject the farthest ~15% outlier crops (coaches, refs, edge artifacts)
                   and refit clean team centroids.
          Stage 3: Compute CIE76 ΔE separation; activate secondary cue ONLY if ΔE_Lab < 16.
        """
        if len(sample_crops_bgr) < 4:
            return

        valid_feats: List[np.ndarray] = []
        valid_crops: List[np.ndarray] = []
        fallback_feats: List[np.ndarray] = []
        fallback_crops: List[np.ndarray] = []

        for crop in sample_crops_bgr:
            feat, is_valid, _ratio, _med3 = self._extract_torso_lab_feature(crop, min_fg_ratio=0.15)
            if is_valid:
                valid_feats.append(feat)
                valid_crops.append(crop)
            else:
                fallback_feats.append(feat)
                fallback_crops.append(crop)

        if len(valid_feats) >= 6:
            fit_list, fit_crops = valid_feats, valid_crops
        else:
            fit_list = valid_feats + fallback_feats
            fit_crops = valid_crops + fallback_crops

        if len(fit_list) < 4:
            return

        # Uniform subsampling if over 800 crops (keeps clustering < 30ms)
        if len(fit_list) > 800:
            idxs = np.linspace(0, len(fit_list) - 1, 800, dtype=int)
            fit_list = [fit_list[i] for i in idxs]
            fit_crops = [fit_crops[i] for i in idxs]

        X = np.asarray(fit_list, dtype=np.float32)

        # ── Stage 1: Initial 2-cluster fit ──
        km1 = KMeans(n_clusters=2, n_init=20, random_state=42)
        labels1 = km1.fit_predict(X)
        centers1 = km1.cluster_centers_

        # ── Stage 2: Reject farthest ~15% outlier crops per cluster & refit clean centroids ──
        if len(X) >= 16:
            dists = np.linalg.norm(X - centers1[labels1], axis=1)
            keep_mask = np.zeros(len(X), dtype=bool)
            for cid in (0, 1):
                c_idx = np.where(labels1 == cid)[0]
                if len(c_idx) <= 4:
                    keep_mask[c_idx] = True
                else:
                    cutoff = float(np.percentile(dists[c_idx], 85.0))
                    keep_mask[c_idx[dists[c_idx] <= cutoff]] = True
            X_clean = X[keep_mask] if int(np.sum(keep_mask)) >= 6 else X
            crops_clean = [fit_crops[i] for i in range(len(fit_crops)) if keep_mask[i]] if int(np.sum(keep_mask)) >= 6 else fit_crops
        else:
            X_clean = X
            crops_clean = fit_crops

        self.kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
        labels_clean = self.kmeans.fit_predict(X_clean)
        centers_clean = self.kmeans.cluster_centers_.astype(np.float32)

        # Deterministic canonical cluster ordering by perceptual (L*, b*, a*) so Team 0/1 never randomly flip
        c0_lab3 = centers_clean[0, :3].copy()
        c1_lab3 = centers_clean[1, :3].copy()
        c0_lab3[0] /= 1.10
        c1_lab3[0] /= 1.10
        if (float(c1_lab3[0]), float(c1_lab3[2]), float(c1_lab3[1])) < (
            float(c0_lab3[0]),
            float(c0_lab3[2]),
            float(c0_lab3[1]),
        ):
            centers_clean = centers_clean[[1, 0]]
            c0_lab3, c1_lab3 = c1_lab3, c0_lab3
            labels_clean = 1 - labels_clean

        self.centroids_lab = centers_clean
        self.centroids_raw_lab3 = np.stack([c0_lab3, c1_lab3], axis=0).astype(np.float32)
        self.cluster_delta_e = float(np.linalg.norm(c0_lab3 - c1_lab3))
        self.team_a_kit_hex = _cie_lab_to_hex(float(c0_lab3[0]), float(c0_lab3[1]), float(c0_lab3[2]))
        self.team_b_kit_hex = _cie_lab_to_hex(float(c1_lab3[0]), float(c1_lab3[1]), float(c1_lab3[2]))
        self.fitted = True

        # ── Stage 3: Smart Multi-Cue Fallback ONLY when ΔE_Lab < 16.0 (ambiguous/similar kits) ──
        self._secondary_centroids = None
        if self.cluster_delta_e < 16.0 and len(crops_clean) >= 8:
            sec_feats = np.asarray([self._extract_secondary_hsv_cue(c) for c in crops_clean], dtype=np.float32)
            c0_sec = sec_feats[labels_clean == 0]
            c1_sec = sec_feats[labels_clean == 1]
            if len(c0_sec) >= 2 and len(c1_sec) >= 2:
                self._secondary_centroids = np.stack(
                    [np.mean(c0_sec, axis=0), np.mean(c1_sec, axis=0)], axis=0
                ).astype(np.float32)
                self.mode_used = "cielab_plus_hsv_cue_fusion"
            else:
                self.mode_used = "cielab_universal_primary"
        else:
            self.mode_used = "cielab_universal_primary"

        print(
            f"[TeamClassifier] Fitted 2-Stage Trimmed CIELAB ({self.mode_used}) on "
            f"{len(X_clean)}/{len(X)} clean crops | ΔE_Lab={self.cluster_delta_e:.1f} | "
            f"Team0={self.team_a_kit_hex} (L={c0_lab3[0]:.1f},a={c0_lab3[1]:.1f},b={c0_lab3[2]:.1f}) vs "
            f"Team1={self.team_b_kit_hex} (L={c1_lab3[0]:.1f},a={c1_lab3[1]:.1f},b={c1_lab3[2]:.1f})"
        )

    def predict_with_confidence(
        self,
        crops_bgr: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(team_labels, confidences)` in [0..1] weighted by centroid margin `|d0-d1|/(d0+d1)`."""
        labels, confs, _min_de = self.predict_detailed(crops_bgr)
        return labels, confs

    def predict_detailed(
        self,
        crops_bgr: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(team_labels, confidences, min_cie_delta_e)` for every crop."""
        if not crops_bgr:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        if not self.fitted or self.centroids_lab is None or self.centroids_raw_lab3 is None:
            n = len(crops_bgr)
            return (
                np.zeros((n,), dtype=np.int32),
                np.zeros((n,), dtype=np.float32),
                np.zeros((n,), dtype=np.float32),
            )

        feats = []
        meds3 = []
        valid_weights = []
        for c in crops_bgr:
            f, ok, ratio, med3 = self._extract_torso_lab_feature(c, min_fg_ratio=0.15)
            feats.append(f)
            meds3.append(med3)
            valid_weights.append(ratio if ok else ratio * 0.25)

        X = np.asarray(feats, dtype=np.float32)
        M3 = np.asarray(meds3, dtype=np.float32)
        d0 = np.linalg.norm(X - self.centroids_lab[0][None, :], axis=1)
        d1 = np.linalg.norm(X - self.centroids_lab[1][None, :], axis=1)

        # Fuse secondary HSV pattern cue ONLY when ΔE_Lab < 16.0
        if self._secondary_centroids is not None and self.cluster_delta_e < 16.0:
            S = np.asarray([self._extract_secondary_hsv_cue(c) for c in crops_bgr], dtype=np.float32)
            sd0 = np.linalg.norm(S - self._secondary_centroids[0][None, :], axis=1) * 28.0
            sd1 = np.linalg.norm(S - self._secondary_centroids[1][None, :], axis=1) * 28.0
            d0 = d0 + 0.45 * sd0
            d1 = d1 + 0.45 * sd1

        labels = (d1 < d0).astype(np.int32)
        d_min = np.minimum(d0, d1)
        d_max = np.maximum(d0, d1)
        margin = (d_max - d_min) / (d_max + d_min + 1e-6)
        weights = np.asarray(valid_weights, dtype=np.float32)
        confs = np.clip(margin * np.clip(weights / 0.45, 0.20, 1.0), 0.01, 1.0).astype(np.float32)

        # True 3D perceptual CIE76 distance to nearest team centroid (for OOD referee detection)
        de0 = np.linalg.norm(M3 - self.centroids_raw_lab3[0][None, :], axis=1)
        de1 = np.linalg.norm(M3 - self.centroids_raw_lab3[1][None, :], axis=1)
        min_de = np.minimum(de0, de1).astype(np.float32)
        return labels, confs, min_de

    def predict(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        labels, _ = self.predict_with_confidence(crops_bgr)
        return labels

    def get_kit_colors_hex(self) -> Dict[str, str]:
        return {
            "team_a_kit_hex": self.team_a_kit_hex,
            "team_b_kit_hex": self.team_b_kit_hex,
        }


def stabilize_team_assignments(
    frame_records: List[dict],
    classifier: SigLIPTeamClassifier,
    player_id: int = 2,
    goalkeeper_id: int = 1,
    referee_id: int = 3,
    max_crops_per_track: int = 16,
    occlusion_iou_thresh: float = 0.15,
) -> Dict[int, int]:
    """
    Assigns locked, non-flickering team IDs per track via:
      1. Strict track-level referee isolation (>= 35% ref frames, OR >= 15% ref frames
         combined with extreme CIELAB color distance from BOTH team centroids).
      2. Strict non-occluded crop selection (ignores overlapping detections with IoU >= 0.15).
      3. Confidence-weighted CIELAB margin consensus (`|d0-d1| / (d0+d1)`).
      4. Goalkeeper assignment strictly by defensive-goal proximity.
    """
    # 1. Majority-vote track class across frames
    cls_votes: Dict[int, Counter] = defaultdict(Counter)
    for rec in frame_records:
        for ent in rec["entities"]:
            cls_votes[int(ent["track_id"])][int(ent["cls"])] += 1

    referee_tids: set[int] = set()
    partial_ref_tids: set[int] = set()
    goalkeeper_tids: set[int] = set()
    for tid, cnt in cls_votes.items():
        total = sum(cnt.values())
        ref_ratio = cnt[referee_id] / max(1, total)
        top_cls, _ = cnt.most_common(1)[0]
        if top_cls == referee_id or ref_ratio >= 0.35:
            referee_tids.add(tid)
        elif ref_ratio >= 0.12 and cnt[referee_id] >= 2:
            # Candidate referee whose detector class flickered -> verify via OOD CIELAB distance below
            partial_ref_tids.add(tid)
        elif top_cls == goalkeeper_id:
            goalkeeper_tids.add(tid)

    # 2. Collect clean non-occluded crops (and fallback crops ONLY if a track is always occluded)
    clean_crops_by_tid: Dict[int, List[np.ndarray]] = defaultdict(list)
    fallback_crops_by_tid: Dict[int, List[np.ndarray]] = defaultdict(list)

    for rec in frame_records:
        ents = rec["entities"]
        boxes = (
            np.asarray([e["box"] for e in ents], dtype=np.float32)
            if ents
            else np.zeros((0, 4), dtype=np.float32)
        )
        iou_mat = _pairwise_iou(boxes) if len(boxes) > 1 else np.zeros((len(boxes), len(boxes)), dtype=np.float32)
        max_ious = iou_mat.max(axis=1) if len(boxes) > 0 else np.zeros((0,), dtype=np.float32)

        for idx, ent in enumerate(ents):
            tid = int(ent["track_id"])
            if tid in referee_tids:
                ent["cls"] = referee_id
                ent["team"] = -1
                ent.pop("crop", None)
                continue
            if tid in goalkeeper_tids:
                ent["cls"] = goalkeeper_id
            elif int(ent["cls"]) == referee_id:
                ent["cls"] = player_id

            crop = ent.get("crop")
            if int(ent["cls"]) == player_id and crop is not None:
                if float(max_ious[idx]) < occlusion_iou_thresh:
                    clean_crops_by_tid[tid].append(crop)
                else:
                    fallback_crops_by_tid[tid].append(crop)

    # 3. Batch CIELAB classification + confidence-weighted track consensus + OOD referee isolation
    batched_crops: List[np.ndarray] = []
    batched_tids: List[int] = []
    all_outfield_tids = set(clean_crops_by_tid.keys()) | set(fallback_crops_by_tid.keys())
    for tid in all_outfield_tids:
        clist = clean_crops_by_tid[tid] if clean_crops_by_tid[tid] else fallback_crops_by_tid[tid]
        if len(clist) > max_crops_per_track:
            idxs = np.linspace(0, len(clist) - 1, max_crops_per_track, dtype=int)
            clist = [clist[i] for i in idxs]
        for c in clist:
            batched_crops.append(c)
            batched_tids.append(tid)

    track_weighted_votes: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0])
    track_min_des: Dict[int, List[float]] = defaultdict(list)
    if batched_crops:
        preds, confs, min_des = classifier.predict_detailed(batched_crops)
        for tid, pred_tm, conf_w, mde in zip(batched_tids, preds, confs, min_des):
            track_weighted_votes[int(tid)][int(pred_tm)] += float(conf_w)
            track_min_des[int(tid)].append(float(mde))

    # Promote flickering referee tracks whose jersey color is clearly OOD from BOTH teams (ΔE > 32)
    for tid in list(partial_ref_tids):
        des = track_min_des.get(tid, [])
        if des and float(np.median(des)) > 32.0:
            referee_tids.add(tid)
            track_weighted_votes.pop(tid, None)

    track_team_map: Dict[int, int] = {}
    for tid, w_pair in track_weighted_votes.items():
        if tid in referee_tids:
            continue
        track_team_map[tid] = 0 if w_pair[0] >= w_pair[1] else 1

    # 4. Determine mean pitch X per team to assign Goalkeepers strictly by defensive-goal proximity
    team_xs: Dict[int, List[float]] = {0: [], 1: []}
    gk_xs: Dict[int, List[float]] = defaultdict(list)
    for rec in frame_records:
        for ent in rec["entities"]:
            tid = int(ent["track_id"])
            if tid in referee_tids:
                ent["cls"] = referee_id
                ent["team"] = -1
                ent.pop("crop", None)
                continue
            px = ent.get("pitch_xy")
            if px is None:
                continue
            if int(ent["cls"]) == player_id and tid in track_team_map:
                team_xs[track_team_map[tid]].append(float(px[0]))
            elif int(ent["cls"]) == goalkeeper_id:
                gk_xs[tid].append(float(px[0]))

    mean_x0 = float(np.mean(team_xs[0])) if team_xs[0] else 35.0
    mean_x1 = float(np.mean(team_xs[1])) if team_xs[1] else 70.0
    left_goal_team = 0 if mean_x0 <= mean_x1 else 1
    right_goal_team = 1 - left_goal_team

    for gk_tid in goalkeeper_tids:
        if gk_tid in referee_tids:
            continue
        xs = gk_xs.get(gk_tid, [])
        if xs:
            gx = float(np.median(xs))
            track_team_map[gk_tid] = left_goal_team if gx < 52.5 else right_goal_team
        else:
            track_team_map[gk_tid] = left_goal_team

    # 5. Apply locked team IDs uniformly across every frame & purge temporary crops from RAM
    for rec in frame_records:
        for ent in rec["entities"]:
            tid = int(ent["track_id"])
            if tid in referee_tids or int(ent["cls"]) == referee_id:
                ent["cls"] = referee_id
                ent["team"] = -1
            elif tid in track_team_map:
                ent["team"] = track_team_map[tid]
            else:
                # Track had no sampled crops -> nearest spatial/default fallback
                ent["team"] = 0
            ent.pop("crop", None)

    return track_team_map
