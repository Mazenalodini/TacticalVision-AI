"""Broadcast-Grade Pitch Homography Calibration & Optical Flow Propagation.

Implements 4 industry-standard geometric invariants (SkillCorner / TRACAB / Second Spectrum):
  1. Broadcast Left-to-Right Monotonicity Law (`dX/du > 0`, `dY/dv > 0`, `det(J) > 0`):
     Automatically resolves 180-deg penalty-box flips by rotating ONLY when `dX/du < 0`
     (never rotating an already-oriented homography!).
  2. Image-Side vs Pitch-Half Consistency + Zero-DOF 4-Point Hallucination Guard:
     Rejects degenerate 4-inlier DLT fits that map a zoomed penalty box to midfield.
  3. Player In-Bounds & Non-Collapse Sanity Gate:
     Rejects any candidate H where >30% of detected on-pitch players project wildly OOB
     or collapse into a degenerate point.
  4. Inter-Frame Background Optical Flow Propagation (`H_t = H_{t-1} @ ΔH_back`):
     Tracks sparse Lucas-Kanade background features (masking out player boxes & HUD)
     so zoomed penalty-box sequences remain smoothly calibrated when keypoints drop out,
     and cleanly sets `homography_valid = False` during extreme close-ups/replays.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull, QhullError

from .config import FIFAPitchConfig


class StabilizedHomography:
    """2-tier oriented RANSAC homography with Player In-Bounds Gate and Optical Flow ΔH fallback."""

    def __init__(
        self,
        pitch_cfg: FIFAPitchConfig,
        kpt_conf_thresh: float = 0.30,
        ema_alpha: float = 0.22,
        max_reproj_m: float = 2.5,
        min_span_m: float = 15.0,
        min_convex_hull_area_m2: float = 150.0,
        temporal_hold_frames: int = 15,
        max_flow_frames: int = 150,
    ):
        self.pitch_cfg = pitch_cfg
        self.pitch_vertices = pitch_cfg.vertices
        self.kpt_conf_thresh = kpt_conf_thresh
        self.ema_alpha = ema_alpha
        self.max_reproj_m = max_reproj_m
        self.min_span_m = min_span_m
        self.min_convex_hull_area_m2 = min_convex_hull_area_m2
        self.temporal_hold_frames = temporal_hold_frames
        self.max_flow_frames = max_flow_frames

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.valid_updates = 0
        self.rejected_updates = 0
        self.flow_propagated_frames = 0
        self.reproj_errors_m: list[float] = []
        self._consecutive_failures = 0
        self._consecutive_flow_frames = 0
        self._anchor_history: list[float] = []
        self._anchor_mean_x: Optional[float] = None
        self.last_was_reset: bool = False
        self.homography_valid: bool = False

    # Tier-2 relaxed thresholds for zoomed/corner/penalty-box angles
    _T2_KPT_CONF = 0.25
    _T2_MIN_POINTS = 4
    _T2_MIN_SPAN_M = 5.0
    _T2_MIN_HULL_M2 = 25.0
    _T2_MAX_REPROJ_M = 3.0
    _T2_STD_X_MIN = 6.0
    _T2_STD_Y_MIN = 5.0

    # Broadcast scale sanity bounds (meters per pixel at 720p-4K)
    _MIN_DX_DU = 0.010
    _MAX_DX_DU = 0.300
    _MIN_DY_DV = 0.018
    _MAX_DY_DV = 0.480

    def reset_scene_cut(self) -> None:
        """Clear homography & half-pitch anchor strictly on a genuine hard scene cut."""
        self.H = None
        self.H_inv = None
        self._consecutive_failures = 0
        self._consecutive_flow_frames = 0
        self._anchor_history.clear()
        self._anchor_mean_x = None
        self.last_was_reset = True
        self.homography_valid = False

    def _rotate_180_pitch(self, H_norm: np.ndarray) -> Optional[np.ndarray]:
        """Compose 180-deg pitch rotation (X'=L-X, Y'=W-Y). Negates both dX/du and dY/dv."""
        L = float(self.pitch_cfg.length)
        W = float(self.pitch_cfg.width)
        R_180 = np.array(
            [[-1.0, 0.0, L], [0.0, -1.0, W], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        H_rot = R_180 @ H_norm.astype(np.float64)
        if not np.all(np.isfinite(H_rot)) or abs(H_rot[2, 2]) < 1e-8:
            return None
        return H_rot / H_rot[2, 2]

    def _evaluate_jacobian(
        self,
        H_mat: np.ndarray,
        ref_x: float,
        ref_y: float,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Evaluate (dx_du, dy_dv, jac_det, center_x_m, center_y_m) at image point (ref_x, ref_y)."""
        w_ref = float(H_mat[2, 0] * ref_x + H_mat[2, 1] * ref_y + H_mat[2, 2])
        if not np.isfinite(w_ref) or abs(w_ref) < 1e-8:
            return None
        H_pos = H_mat.astype(np.float64) / w_ref
        probe = np.array(
            [[[ref_x, ref_y]], [[ref_x + 10.0, ref_y]], [[ref_x, ref_y + 10.0]]],
            dtype=np.float32,
        )
        p_probe = cv2.perspectiveTransform(probe, H_pos).reshape(3, 2)
        if not np.all(np.isfinite(p_probe)):
            return None
        dx_m = p_probe[1] - p_probe[0]
        dy_m = p_probe[2] - p_probe[0]
        dx_du = float(dx_m[0] / 10.0)
        dy_dv = float(dy_m[1] / 10.0)
        jac_det = float((dx_m[0] * dy_m[1] - dx_m[1] * dy_m[0]) / 100.0)
        return dx_du, dy_dv, jac_det, float(p_probe[0, 0]), float(p_probe[0, 1])

    def _validate_player_inbounds(
        self,
        H_mat: np.ndarray,
        player_feet_px: Optional[np.ndarray],
        min_inbounds_ratio: float = 0.70,
    ) -> Tuple[bool, float, float]:
        """
        Player In-Bounds & Non-Collapse Sanity Gate.
        Returns `(ok, median_x, inbounds_ratio)`.
        Rejects projective explosions where >30% of players map wildly OOB or collapse to a point.
        """
        if player_feet_px is None or len(player_feet_px) == 0:
            return True, float(self._anchor_mean_x or 52.5), 1.0

        pts = np.asarray(player_feet_px, dtype=np.float32).reshape(-1, 2)
        n = len(pts)
        mapped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_mat.astype(np.float64)).reshape(-1, 2)
        finite = np.all(np.isfinite(mapped), axis=1)
        if int(np.sum(finite)) == 0:
            return False, 52.5, 0.0

        L = float(self.pitch_cfg.length)
        W = float(self.pitch_cfg.width)
        # Also verify positive projective depth w > 0.05 at player feet (no behind-horizon projection)
        w_vals = H_mat[2, 0] * pts[:, 0] + H_mat[2, 1] * pts[:, 1] + H_mat[2, 2]
        # Normalize sign using median w of players
        w_sign = 1.0 if float(np.median(w_vals)) >= 0.0 else -1.0
        w_pos = (w_vals * w_sign) > 0.05

        inb_mask = (
            finite
            & w_pos
            & (mapped[:, 0] >= -4.0)
            & (mapped[:, 0] <= L + 4.0)
            & (mapped[:, 1] >= -4.0)
            & (mapped[:, 1] <= W + 4.0)
        )
        inb_count = int(np.sum(inb_mask))
        inb_ratio = float(inb_count) / max(1, n)

        if n >= 3 and inb_ratio < min_inbounds_ratio:
            return False, float(np.median(mapped[finite, 0])), inb_ratio

        # Reject projective collapse (players widely spread in image collapsing to < 3m on pitch)
        if inb_count >= 4:
            u_span = float(np.ptp(pts[inb_mask, 0]))
            x_span = float(np.ptp(mapped[inb_mask, 0]))
            if u_span >= 180.0 and x_span < 3.0:
                return False, float(np.median(mapped[inb_mask, 0])), inb_ratio

        ref_xs = mapped[inb_mask, 0] if inb_count >= 1 else mapped[finite, 0]
        return True, float(np.median(ref_xs)), inb_ratio

    def _update_core(
        self,
        kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
        conf_thresh: float,
        min_points: int,
        std_x_min: float,
        std_y_min: float,
        min_span: float,
        min_hull: float,
        max_reproj: float,
        player_feet_px: Optional[np.ndarray] = None,
        is_scene_cut: bool = False,
        img_width: Optional[int] = None,
    ) -> bool:
        """Core homography fitting with Broadcast Monotonicity Law (dX/du > 0) & Player In-Bounds Gate."""
        valid_xy = (kpts_xy[:, 0] > 2.0) & (kpts_xy[:, 1] > 2.0)
        mask = valid_xy & (kpts_conf >= conf_thresh)
        if int(mask.sum()) < min_points:
            return False

        src_pts = kpts_xy[mask].astype(np.float32)
        dst_pts = self.pitch_vertices[mask].astype(np.float32)

        # Reject tight pixel clusters that blow up projective scale
        if np.std(src_pts[:, 0]) < std_x_min or np.std(src_pts[:, 1]) < std_y_min:
            return False

        # Geometric Pitch Span & Convex-Hull Area Gate
        span_diag_m = float(np.hypot(np.ptp(dst_pts[:, 0]), np.ptp(dst_pts[:, 1])))
        if span_diag_m < min_span:
            return False

        if len(dst_pts) >= 3:
            try:
                hull_area_m2 = float(ConvexHull(dst_pts).volume)
            except QhullError:
                hull_area_m2 = 0.0
            if hull_area_m2 < min_hull:
                return False

        H_cand, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_cand is None or inliers is None or int(inliers.sum()) < 4:
            return False
        if not np.all(np.isfinite(H_cand)):
            return False

        inl_mask = inliers.ravel().astype(bool)
        num_inliers = int(inl_mask.sum())
        if num_inliers < 4:
            return False

        inl_src = src_pts[inl_mask]
        inl_dst = dst_pts[inl_mask]
        if float(np.hypot(np.ptp(inl_dst[:, 0]), np.ptp(inl_dst[:, 1]))) < min_span:
            return False

        # Normalize projective depth at inlier centroid (solves cv2 +/- scale ambiguity)
        ref_x = float(np.median(inl_src[:, 0]))
        ref_y = float(np.median(inl_src[:, 1]))
        w_ref = float(H_cand[2, 0] * ref_x + H_cand[2, 1] * ref_y + H_cand[2, 2])
        if abs(w_ref) < 1e-8:
            return False
        H_pos = H_cand.astype(np.float64) / w_ref

        # Horizon-fold guard: all visible inliers strictly in front of camera horizon
        w_inl = H_pos[2, 0] * inl_src[:, 0] + H_pos[2, 1] * inl_src[:, 1] + H_pos[2, 2]
        if np.any(w_inl <= 0.08):
            return False

        # Reprojection error gate against inlier destination keypoints
        proj = cv2.perspectiveTransform(inl_src.reshape(-1, 1, 2), H_pos).reshape(-1, 2)
        err_m = float(np.mean(np.linalg.norm(proj - inl_dst, axis=1)))
        gate_m = max_reproj if self.H is not None else max(5.5, max_reproj * 2.0)
        if err_m > gate_m:
            return False

        if abs(H_pos[2, 2]) < 1e-8:
            return False
        H_norm = H_pos / H_pos[2, 2]

        # ── LAW 1: Broadcast Left-to-Right Monotonicity Law (dX/du > 0, dY/dv > 0, det(J) > 0) ──
        jac_info = self._evaluate_jacobian(H_norm, ref_x, ref_y)
        if jac_info is None:
            return False
        dx_du, dy_dv, jac_det, cx_m, cy_m = jac_info

        # If dX/du < 0, pitch_best.pt confused left & right penalty boxes -> auto-test 180° rotation!
        if dx_du < 0.0:
            H_rot = self._rotate_180_pitch(H_norm)
            if H_rot is None:
                return False
            rot_jac = self._evaluate_jacobian(H_rot, ref_x, ref_y)
            if rot_jac is None:
                return False
            r_dx, r_dy, r_det, r_cx, r_cy = rot_jac
            if r_dx > 0.0 and r_dy > 0.0 and r_det > 0.0:
                H_norm = H_rot
                dx_du, dy_dv, jac_det, cx_m, cy_m = r_dx, r_dy, r_det, r_cx, r_cy
            else:
                return False

        # Reject upside-down, flipped, collapsed, or exploded broadcast scales
        if (
            jac_det <= 0.0
            or dx_du < self._MIN_DX_DU
            or dx_du > self._MAX_DX_DU
            or dy_dv < self._MIN_DY_DV
            or dy_dv > self._MAX_DY_DV
        ):
            return False

        # Inlier anchor centroid must map reasonably on or adjacent to the pitch
        if not (-12.0 <= cx_m <= self.pitch_cfg.length + 12.0):
            return False
        if not (-12.0 <= cy_m <= self.pitch_cfg.width + 12.0):
            return False

        # ── LAW 2: Universal Image-Side vs Pitch-Half Consistency for Penalty Box Landmarks ──
        # Applies across ALL inlier counts: if a penalty box sits clearly on the RIGHT side of
        # the frame with open pitch/players to the left, it CANNOT map to the left half (< 50m),
        # and vice-versa. Prevents shifted-but-positive-gradient (dX/du > 0) left/right mislabels!
        if img_width is not None and img_width > 100:
            box_mask = (inl_dst[:, 0] <= 16.5) | (inl_dst[:, 0] >= self.pitch_cfg.length - 16.5)
            if int(np.sum(box_mask)) >= 2 and player_feet_px is not None and len(player_feet_px) >= 4:
                box_u_med = float(np.median(inl_src[box_mask, 0]))
                p_u_min = float(np.min(player_feet_px[:, 0]))
                p_u_max = float(np.max(player_feet_px[:, 0]))
                # Penalty box on right of frame with open pitch to the left -> must map to right half (> 50m)
                if box_u_med > 0.56 * img_width and p_u_min < 0.38 * img_width and cx_m < 50.0:
                    return False
                # Penalty box on left of frame with open pitch to the right -> must map to left half (< 55m)
                if box_u_med < 0.44 * img_width and p_u_max > 0.62 * img_width and cx_m > 55.0:
                    return False

        # ── LAW 3: Player In-Bounds & Non-Collapse Sanity Gate ──
        inb_ok, cand_med_x, _inb_ratio = self._validate_player_inbounds(H_norm, player_feet_px)
        if not inb_ok:
            return False

        # ── LAW 4: Continuity & Zero-DOF / Mid-Clip Jump Guard ──
        # Reject unmotivated pitch teleports when no scene cut occurred (e.g. penalty box suddenly jumping to midfield)
        if not is_scene_cut and self.H is not None:
            prev_ok, prev_med_x, _ = self._validate_player_inbounds(
                self.H, player_feet_px, min_inbounds_ratio=0.0
            )
            anchor_x = prev_med_x if (player_feet_px is not None and len(player_feet_px) >= 3 and prev_ok) else (
                self._anchor_mean_x if self._anchor_mean_x is not None else cand_med_x
            )
            jump_x = abs(cand_med_x - anchor_x)
            if num_inliers <= 5 and jump_x > 14.0:
                return False
            if num_inliers <= 8 and jump_x > 24.0:
                return False

        try:
            H_inv_cand = np.linalg.inv(H_norm)
        except np.linalg.LinAlgError:
            return False

        if self.H is None:
            self.H = H_norm.copy()
        else:
            # Blend smoothly; trust strong multi-inlier recoveries faster after optical-flow stretches
            alpha = min(0.45, self.ema_alpha * 1.6) if (self._consecutive_flow_frames > 0 and num_inliers >= 6) else self.ema_alpha
            blended = (1.0 - alpha) * self.H + alpha * H_norm
            if abs(blended[2, 2]) > 1e-8:
                self.H = blended / blended[2, 2]
            else:
                self.H = H_norm.copy()

        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H = H_norm
            self.H_inv = H_inv_cand

        self._anchor_history.append(cand_med_x)
        if len(self._anchor_history) > 10:
            self._anchor_history.pop(0)
        self._anchor_mean_x = float(np.median(self._anchor_history))
        self.last_was_reset = False
        self.homography_valid = True
        self._consecutive_flow_frames = 0

        self.valid_updates += 1
        self.reproj_errors_m.append(err_m)
        return True

    @staticmethod
    def is_genuine_closeup(
        person_boxes: Optional[np.ndarray],
        frame_h: int,
    ) -> bool:
        """
        Distinguish genuine non-tactical shots (extreme close-ups, player faces, crowd,
        celebrations) from wide tactical gameplay (which MUST NOT falsely pause).
        """
        if frame_h <= 0:
            return False
        if person_boxes is None or len(person_boxes) == 0:
            return True
        bh_ratios = (person_boxes[:, 3] - person_boxes[:, 1]) / float(frame_h)
        med_ratio = float(np.median(bh_ratios))
        # Wide tactical broadcast shots have player heights 0.05..0.18 of frame height;
        # extreme close-ups / celebrations have median player height > 0.30 or <= 2 giant boxes.
        if med_ratio > 0.30:
            return True
        if len(person_boxes) <= 2 and float(np.max(bh_ratios)) > 0.28:
            return True
        return False

    def apply_external_homography(
        self,
        H_cand: np.ndarray,
        player_feet_px: Optional[np.ndarray] = None,
        is_scene_cut: bool = False,
        img_width: int = 1280,
        img_height: int = 720,
        reproj_err_m: float = 0.0,
        ema_alpha_override: Optional[float] = None,
    ) -> bool:
        """
        Validate and blend an externally estimated homography (e.g. from PnLCalib 3D)
        through the exact same universal geometric sanity gates:
          - Broadcast Monotonicity Law (`dX/du > 0`, `dY/dv > 0`, `det(J) > 0`)
          - Realistic broadcast scale bounds
          - Player In-Bounds & Non-Collapse Gate
          - Smooth temporal EMA blending
        """
        if is_scene_cut:
            self.reset_scene_cut()
        if H_cand is None or H_cand.shape != (3, 3) or not np.all(np.isfinite(H_cand)):
            return False
        if abs(float(H_cand[2, 2])) < 1e-8:
            return False

        H_norm = H_cand.astype(np.float64) / float(H_cand[2, 2])
        ref_x = float(img_width) * 0.5
        ref_y = float(img_height) * 0.55

        jac_info = self._evaluate_jacobian(H_norm, ref_x, ref_y)
        if jac_info is None:
            return False
        dx_du, dy_dv, jac_det, cx_m, cy_m = jac_info

        # Auto-resolve 180-deg flip ONLY if dX/du < 0
        if dx_du < 0.0:
            H_rot = self._rotate_180_pitch(H_norm)
            if H_rot is None:
                return False
            rot_jac = self._evaluate_jacobian(H_rot, ref_x, ref_y)
            if rot_jac is None:
                return False
            r_dx, r_dy, r_det, r_cx, r_cy = rot_jac
            if r_dx > 0.0 and r_dy > 0.0 and r_det > 0.0:
                H_norm = H_rot
                dx_du, dy_dv, jac_det, cx_m, cy_m = r_dx, r_dy, r_det, r_cx, r_cy
            else:
                return False

        if (
            jac_det <= 0.0
            or dx_du < self._MIN_DX_DU
            or dx_du > self._MAX_DX_DU
            or dy_dv < self._MIN_DY_DV
            or dy_dv > self._MAX_DY_DV
        ):
            return False

        if not (-20.0 <= cx_m <= self.pitch_cfg.length + 20.0):
            return False
        if not (-20.0 <= cy_m <= self.pitch_cfg.width + 20.0):
            return False

        inb_ok, cand_med_x, _inb_ratio = self._validate_player_inbounds(
            H_norm, player_feet_px, min_inbounds_ratio=0.65
        )
        if not inb_ok:
            return False

        try:
            H_inv_cand = np.linalg.inv(H_norm)
        except np.linalg.LinAlgError:
            return False

        if self.H is None:
            self.H = H_norm.copy()
        else:
            alpha = (
                float(ema_alpha_override)
                if ema_alpha_override is not None
                else min(0.55, self.ema_alpha * 2.0)
            )
            blended = (1.0 - alpha) * self.H + alpha * H_norm
            if abs(float(blended[2, 2])) > 1e-8:
                self.H = blended / float(blended[2, 2])
            else:
                self.H = H_norm.copy()

        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H = H_norm
            self.H_inv = H_inv_cand

        self._anchor_history.append(cand_med_x)
        if len(self._anchor_history) > 10:
            self._anchor_history.pop(0)
        self._anchor_mean_x = float(np.median(self._anchor_history))
        self.last_was_reset = False
        self.homography_valid = True
        self._consecutive_flow_frames = 0
        self._consecutive_failures = 0
        self.valid_updates += 1
        if reproj_err_m >= 0.0:
            self.reproj_errors_m.append(float(reproj_err_m))
        return True

    def propagate_optical_flow(
        self,
        prev_gray: Optional[np.ndarray],
        cur_gray: Optional[np.ndarray],
        exclude_boxes: Optional[np.ndarray] = None,
        player_feet_px: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Propagate `H_{t-1} -> H_t = H_{t-1} @ ΔH_back` using sparse Lucas-Kanade optical flow
        strictly on background/pitch pixels (masking out all player bounding boxes & top HUD).
        """
        if self.H is None or prev_gray is None or cur_gray is None:
            return False
        if prev_gray.shape != cur_gray.shape:
            return False
        if self._consecutive_flow_frames >= self.max_flow_frames:
            return False

        h, w = prev_gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        # Mask out top 11% (broadcast scoreboards/HUD) and bottom 3%
        mask[:int(h * 0.11), :] = 0
        mask[int(h * 0.97):, :] = 0

        # Mask out all player/referee/goalkeeper bounding boxes (+10px safety margin)
        if exclude_boxes is not None and len(exclude_boxes) > 0:
            for box in exclude_boxes:
                bx1 = max(0, int(box[0]) - 10)
                by1 = max(0, int(box[1]) - 10)
                bx2 = min(w, int(box[2]) + 10)
                by2 = min(h, int(box[3]) + 10)
                if bx2 > bx1 and by2 > by1:
                    mask[by1:by2, bx1:bx2] = 0

        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=320,
            qualityLevel=0.015,
            minDistance=14,
            mask=mask,
            blockSize=7,
        )
        if pts_prev is None or len(pts_prev) < 18:
            return False

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        pts_cur, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts_prev, None, **lk_params)
        if pts_cur is None or st_fwd is None:
            return False

        # Forward-backward consistency filter eliminates drifting/occluded flow tracks
        pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, pts_cur, None, **lk_params)
        if pts_back is None or st_bwd is None:
            return False

        fb_err = np.linalg.norm(pts_prev.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
        good = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_err < 1.5)
        if int(np.sum(good)) < 16:
            return False

        src_cur = pts_cur[good].reshape(-1, 1, 2)
        dst_prev = pts_prev[good].reshape(-1, 1, 2)

        # Fit backward homography ΔH_back mapping cur_img -> prev_img directly (avoids inverting ΔH)
        dH_back, inl = cv2.findHomography(src_cur, dst_prev, cv2.RANSAC, 2.5)
        if dH_back is None or inl is None or not np.all(np.isfinite(dH_back)):
            return False
        inlier_cnt = int(inl.sum())
        inlier_ratio = float(inlier_cnt) / max(1, len(src_cur))
        if inlier_cnt < 14 or inlier_ratio < 0.45:
            return False

        H_prop = self.H.astype(np.float64) @ dH_back.astype(np.float64)
        if not np.all(np.isfinite(H_prop)) or abs(H_prop[2, 2]) < 1e-8:
            return False
        H_prop = H_prop / H_prop[2, 2]

        # Validate propagated H against Broadcast Monotonicity + Player In-Bounds Gate
        jac_info = self._evaluate_jacobian(H_prop, w * 0.5, h * 0.55)
        if jac_info is None:
            return False
        dx_du, dy_dv, jac_det, _cx, _cy = jac_info
        if (
            jac_det <= 0.0
            or dx_du < self._MIN_DX_DU
            or dx_du > self._MAX_DX_DU
            or dy_dv < self._MIN_DY_DV
            or dy_dv > self._MAX_DY_DV
        ):
            return False

        inb_ok, prop_med_x, _ = self._validate_player_inbounds(H_prop, player_feet_px, min_inbounds_ratio=0.65)
        if not inb_ok:
            return False

        try:
            H_inv_prop = np.linalg.inv(H_prop)
        except np.linalg.LinAlgError:
            return False

        self.H = H_prop
        self.H_inv = H_inv_prop
        self.homography_valid = True
        self._consecutive_flow_frames += 1
        self._consecutive_failures = 0
        self.flow_propagated_frames += 1
        if player_feet_px is not None and len(player_feet_px) >= 2:
            self._anchor_history.append(prop_med_x)
            if len(self._anchor_history) > 10:
                self._anchor_history.pop(0)
            self._anchor_mean_x = float(np.median(self._anchor_history))
        return True

    def update(
        self,
        kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
        player_feet_px: Optional[np.ndarray] = None,
        is_scene_cut: bool = False,
        prev_gray: Optional[np.ndarray] = None,
        cur_gray: Optional[np.ndarray] = None,
        exclude_boxes: Optional[np.ndarray] = None,
        img_width: Optional[int] = None,
    ) -> bool:
        """Try Tier-1 -> Tier-1.5 -> Tier-2 -> Background Optical Flow ΔH -> Brief Hold."""
        self.last_was_reset = False
        if is_scene_cut:
            self.reset_scene_cut()

        # First propagate existing H smoothly by inter-frame camera flow so even if pitch_best.pt
        # runs on a strided frame or gets rejected, the anchor tracks camera motion accurately!
        flow_ok = False
        if not is_scene_cut and self.H is not None and prev_gray is not None and cur_gray is not None:
            flow_ok = self.propagate_optical_flow(
                prev_gray, cur_gray, exclude_boxes=exclude_boxes, player_feet_px=player_feet_px
            )

        ok = self._update_core(
            kpts_xy,
            kpts_conf,
            conf_thresh=self.kpt_conf_thresh,
            min_points=4,
            std_x_min=12.0,
            std_y_min=10.0,
            min_span=self.min_span_m,
            min_hull=self.min_convex_hull_area_m2,
            max_reproj=self.max_reproj_m,
            player_feet_px=player_feet_px,
            is_scene_cut=is_scene_cut,
            img_width=img_width,
        )
        if not ok:
            ok = self._update_core(
                kpts_xy,
                kpts_conf,
                conf_thresh=min(0.25, self.kpt_conf_thresh),
                min_points=4,
                std_x_min=12.0,
                std_y_min=10.0,
                min_span=self.min_span_m,
                min_hull=self.min_convex_hull_area_m2,
                max_reproj=self.max_reproj_m,
                player_feet_px=player_feet_px,
                is_scene_cut=is_scene_cut,
                img_width=img_width,
            )
        if not ok:
            ok = self._update_core(
                kpts_xy,
                kpts_conf,
                conf_thresh=self._T2_KPT_CONF,
                min_points=self._T2_MIN_POINTS,
                std_x_min=self._T2_STD_X_MIN,
                std_y_min=self._T2_STD_Y_MIN,
                min_span=self._T2_MIN_SPAN_M,
                min_hull=self._T2_MIN_HULL_M2,
                max_reproj=self._T2_MAX_REPROJ_M,
                player_feet_px=player_feet_px,
                is_scene_cut=is_scene_cut,
                img_width=img_width,
            )

        if ok:
            self._consecutive_failures = 0
            self.homography_valid = True
            return True

        if flow_ok:
            # Optical flow ΔH cleanly tracked camera pan/zoom while keypoints were absent/rejected
            self.homography_valid = True
            return True

        self._consecutive_failures += 1
        self.rejected_updates += 1
        if self._consecutive_failures > self.temporal_hold_frames and self.H is not None:
            # Sustained loss (close-up / celebration / replay) -> clear stale H honestly
            self.H = None
            self.H_inv = None
            self.homography_valid = False
            self.last_was_reset = True
        self.homography_valid = self.H is not None
        return self.H is not None

    def image_to_pitch(self, pts_xy: np.ndarray) -> Optional[np.ndarray]:
        if self.H is None or len(pts_xy) == 0:
            return None
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.H.astype(np.float64)).reshape(-1, 2)
        L, W = float(self.pitch_cfg.length), float(self.pitch_cfg.width)
        near_x = (mapped[:, 0] >= -4.0) & (mapped[:, 0] <= L + 4.0)
        near_y = (mapped[:, 1] >= -4.0) & (mapped[:, 1] <= W + 4.0)
        near_mask = near_x & near_y
        mapped[near_mask, 0] = np.clip(mapped[near_mask, 0], 0.3, L - 0.3)
        mapped[near_mask, 1] = np.clip(mapped[near_mask, 1], 0.3, W - 0.3)
        mapped[~near_mask, 0] = np.clip(mapped[~near_mask, 0], -2.0, L + 2.0)
        mapped[~near_mask, 1] = np.clip(mapped[~near_mask, 1], -2.0, W + 2.0)
        return mapped

    def pitch_to_image(self, pts_m: np.ndarray) -> Optional[np.ndarray]:
        if self.H_inv is None or len(pts_m) == 0:
            return None
        pts = np.asarray(pts_m, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv.astype(np.float64)).reshape(-1, 2)

    def summary(self) -> dict:
        total = self.valid_updates + self.rejected_updates
        return {
            "valid_frames": self.valid_updates,
            "rejected_frames": self.rejected_updates,
            "flow_propagated_frames": self.flow_propagated_frames,
            "stability_rate_pct": round(100.0 * self.valid_updates / max(1, total), 2),
            "mean_inlier_reproj_error_m": round(float(np.mean(self.reproj_errors_m)), 3) if self.reproj_errors_m else None,
        }


def smooth_shot_homographies(
    frame_records: List[dict],
    pitch_cfg: FIFAPitchConfig,
    img_w: int,
    img_h: int,
    fps: float = 25.0,
    outlier_thresh_m: float = 6.0,
    outlier_window_radius: int = 7,
) -> Dict[str, Any]:
    """
    Offline Pass 1.5 — Zero-Phase Shot-Level Homography Outlier Rejection & Smoothing.

    Segments `frame_records` into continuous camera shots split at genuine scene cuts
    (`is_scene_cut == True`) or invalid/paused spans (`H is None` or `homography_valid == False`).
    Within each continuous shot:
      1. Normalizes `H_t` (`H_t / H_t[2,2]`) and evaluates image-center pitch point `(cx_t, cy_t)`
         plus local scale `dx_du_t`.
      2. Flags isolated 1-3 frame calibration glitches deviating `> outlier_thresh_m` (6.0m)
         from the rolling `+-7`-frame median (or scale deviating > 45%), replacing them via
         linear interpolation of normalized `H` between nearest clean neighbors.
      3. Applies zero-phase Savitzky-Golay smoothing (window ~0.6s odd, polyorder=2;
         moving-average fallback on very short shots) across the shot.
      4. Validates every smoothed `H_t` against Broadcast Monotonicity (`dX/du > 0`, `det(J) > 0`)
         and Player In-Bounds before committing `H`, `H_inv`, and re-projecting all player `pitch_xy`.
    """
    n = len(frame_records)
    if n == 0:
        return {"shots_smoothed": 0, "outliers_replaced": 0, "frames_smoothed": 0}

    validator = StabilizedHomography(pitch_cfg)
    ref_x = float(img_w) * 0.5
    ref_y = float(img_h) * 0.55

    # 1. Partition into continuous valid camera shots
    shots: List[List[int]] = []
    cur_shot: List[int] = []
    for idx, rec in enumerate(frame_records):
        H = rec.get("H")
        hv = bool(rec.get("homography_valid", H is not None))
        is_cut = bool(rec.get("is_scene_cut", False))
        if is_cut and cur_shot:
            shots.append(cur_shot)
            cur_shot = []
        if hv and H is not None and np.all(np.isfinite(H)) and abs(float(H[2, 2])) > 1e-8:
            cur_shot.append(idx)
        else:
            if cur_shot:
                shots.append(cur_shot)
                cur_shot = []
    if cur_shot:
        shots.append(cur_shot)

    shots_smoothed = 0
    outliers_replaced = 0
    frames_smoothed = 0

    for shot in shots:
        m = len(shot)
        if m < 3:
            continue

        # Normalize H matrices along the shot
        H_seq = np.zeros((m, 3, 3), dtype=np.float64)
        cx_arr = np.zeros(m, dtype=np.float64)
        cy_arr = np.zeros(m, dtype=np.float64)
        dx_arr = np.zeros(m, dtype=np.float64)
        valid_jac = np.ones(m, dtype=bool)

        for k, f_idx in enumerate(shot):
            Hk = np.asarray(frame_records[f_idx]["H"], dtype=np.float64)
            Hk = Hk / float(Hk[2, 2])
            H_seq[k] = Hk
            jinfo = validator._evaluate_jacobian(Hk, ref_x, ref_y)
            if jinfo is None:
                valid_jac[k] = False
            else:
                dx_du, _dy_dv, _det, cx_m, cy_m = jinfo
                cx_arr[k] = cx_m
                cy_arr[k] = cy_m
                dx_arr[k] = dx_du

        # 2. Rolling-median outlier keyframe detection (catches 1-3 frame PnLCalib spikes!)
        is_outlier = ~valid_jac
        rad = max(2, int(outlier_window_radius))
        for k in range(m):
            if is_outlier[k]:
                continue
            lo = max(0, k - rad)
            hi = min(m, k + rad + 1)
            win_valid = valid_jac[lo:hi]
            if int(np.sum(win_valid)) < 3:
                continue
            med_cx = float(np.median(cx_arr[lo:hi][win_valid]))
            med_cy = float(np.median(cy_arr[lo:hi][win_valid]))
            med_dx = float(np.median(dx_arr[lo:hi][win_valid]))
            dist_c = float(np.hypot(cx_arr[k] - med_cx, cy_arr[k] - med_cy))
            scale_dev = abs(dx_arr[k] - med_dx) / max(1e-5, abs(med_dx))
            if dist_c > outlier_thresh_m or scale_dev > 0.45:
                is_outlier[k] = True

        # Replace flagged outliers via linear interpolation between nearest clean frames
        clean_indices = np.where(~is_outlier)[0]
        H_clean = H_seq.copy()
        if len(clean_indices) >= 2 and np.any(is_outlier):
            for k in np.where(is_outlier)[0]:
                pos = int(np.searchsorted(clean_indices, k))
                if pos == 0:
                    H_clean[k] = H_seq[clean_indices[0]]
                elif pos >= len(clean_indices):
                    H_clean[k] = H_seq[clean_indices[-1]]
                else:
                    i0 = int(clean_indices[pos - 1])
                    i1 = int(clean_indices[pos])
                    alpha = float(k - i0) / float(max(1, i1 - i0))
                    H_interp = (1.0 - alpha) * H_seq[i0] + alpha * H_seq[i1]
                    if abs(float(H_interp[2, 2])) > 1e-8:
                        H_interp = H_interp / float(H_interp[2, 2])
                    H_clean[k] = H_interp
                outliers_replaced += 1

        # 3. Zero-phase Savitzky-Golay smoothing across the continuous shot
        target_win = max(5, int(round(0.60 * max(5.0, float(fps)))))
        if target_win % 2 == 0:
            target_win += 1
        max_odd = m if (m % 2 == 1) else (m - 1)
        win_len = min(max_odd, target_win)

        H_flat = H_clean.reshape(m, 9)
        if win_len >= 5:
            H_smooth_flat = savgol_filter(H_flat, window_length=win_len, polyorder=2, axis=0, mode="interp")
        elif m >= 3:
            # Short shot (3-4 frames): gentle 3-tap zero-phase moving average
            kern = np.array([0.25, 0.50, 0.25], dtype=np.float64)
            H_smooth_flat = np.empty_like(H_flat)
            for col in range(9):
                padded = np.pad(H_flat[:, col], (1, 1), mode="edge")
                H_smooth_flat[:, col] = np.convolve(padded, kern, mode="valid")
        else:
            H_smooth_flat = H_flat

        H_smooth_seq = H_smooth_flat.reshape(m, 3, 3)
        shots_smoothed += 1

        # 4. Validate each smoothed H against Monotonicity + Player In-Bounds, then re-project players
        for k, f_idx in enumerate(shot):
            rec = frame_records[f_idx]
            Hs = H_smooth_seq[k]
            if not np.all(np.isfinite(Hs)) or abs(float(Hs[2, 2])) < 1e-8:
                Hs = H_clean[k]
            else:
                Hs = Hs / float(Hs[2, 2])

            ents = rec.get("entities", [])
            feet_px = (
                np.asarray(
                    [[(float(e["box"][0]) + float(e["box"][2])) * 0.5, float(e["box"][3])] for e in ents],
                    dtype=np.float32,
                )
                if ents
                else None
            )

            # Verify smoothed H still preserves broadcast monotonicity & player in-bounds
            use_H = Hs
            j_s = validator._evaluate_jacobian(Hs, ref_x, ref_y)
            ok_s = False
            if j_s is not None:
                dx_s, dy_s, det_s, _cx_s, _cy_s = j_s
                if (
                    det_s > 0.0
                    and validator._MIN_DX_DU <= dx_s <= validator._MAX_DX_DU
                    and validator._MIN_DY_DV <= dy_s <= validator._MAX_DY_DV
                ):
                    inb_ok, _, _ = validator._validate_player_inbounds(Hs, feet_px, min_inbounds_ratio=0.60)
                    ok_s = inb_ok
            if not ok_s:
                use_H = H_clean[k]

            try:
                use_H_inv = np.linalg.inv(use_H)
            except np.linalg.LinAlgError:
                continue

            rec["H"] = use_H.copy()
            rec["H_inv"] = use_H_inv.copy()
            rec["homography_valid"] = True
            # Inside a continuous smooth shot (k > 0), clear transient reset flags so sprints aren't falsely wiped
            if k > 0 and not rec.get("is_scene_cut", False):
                rec["homography_reset"] = False

            if feet_px is not None and len(feet_px) > 0:
                validator.H = use_H
                mapped = validator.image_to_pitch(feet_px)
                if mapped is not None:
                    for e, pt in zip(ents, mapped):
                        e["pitch_xy"] = pt.copy()
            frames_smoothed += 1

    return {
        "shots_smoothed": shots_smoothed,
        "outliers_replaced": outliers_replaced,
        "frames_smoothed": frames_smoothed,
    }
