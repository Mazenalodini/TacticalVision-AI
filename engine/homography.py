from typing import Optional
import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .config import FIFAPitchConfig


class StabilizedHomography:
    """2-tier adaptive RANSAC homography with temporal hold, Jacobian validation, and scene-cut reset.

    Tier 1 (wide-angle): Standard gates (min_span >= 15m, hull >= 150m^2, reproj <= 2.5m).
    Tier 2 (zoomed/corner-kick): Relaxed gates (>= 4 non-collinear keypoints with
        conf >= 0.25, RANSAC inliers >= 4, reproj <= 3.0m, positive local Jacobian det,
        all inliers positive projective depth w > 0.08).

    When both tiers fail, the last valid H is held for up to `temporal_hold_frames`
    consecutive failures, then reset (scene cut / replay detection).
    """

    def __init__(
        self,
        pitch_cfg: FIFAPitchConfig,
        kpt_conf_thresh: float = 0.30,
        ema_alpha: float = 0.20,
        max_reproj_m: float = 2.5,
        min_span_m: float = 15.0,
        min_convex_hull_area_m2: float = 150.0,
        temporal_hold_frames: int = 20,
    ):
        self.pitch_cfg = pitch_cfg
        self.pitch_vertices = pitch_cfg.vertices
        self.kpt_conf_thresh = kpt_conf_thresh
        self.ema_alpha = ema_alpha
        self.max_reproj_m = max_reproj_m
        self.min_span_m = min_span_m
        self.min_convex_hull_area_m2 = min_convex_hull_area_m2
        self.temporal_hold_frames = temporal_hold_frames

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.valid_updates = 0
        self.rejected_updates = 0
        self.reproj_errors_m: list[float] = []
        self._consecutive_failures = 0
        self._anchor_history: list[float] = []
        self._anchor_mean_x: Optional[float] = None
        self.last_was_reset: bool = False

    def reset_scene_cut(self) -> None:
        """Reset half-pitch anchor and homography state ONLY on a genuine hard scene cut."""
        self.H = None
        self.H_inv = None
        self._consecutive_failures = 0
        self._anchor_history.clear()
        self._anchor_mean_x = None
        self.last_was_reset = True

    def _rotate_180_pitch(self, H_norm: np.ndarray) -> Optional[np.ndarray]:
        """Compose 180-degree pitch symmetry transform (x'=L-x, y'=W-y) preserving +Jacobian."""
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

    def _project_cloud_stats(
        self,
        H_mat: np.ndarray,
        player_feet_px: Optional[np.ndarray],
        fallback_x: float,
    ) -> tuple[float, int]:
        """Compute (median_x, inbounds_count) for visible players under candidate H."""
        L = float(self.pitch_cfg.length)
        W = float(self.pitch_cfg.width)
        if player_feet_px is None or len(player_feet_px) < 2:
            inb = 1 if (0.0 <= fallback_x <= L) else 0
            return float(fallback_x), inb

        pts = np.asarray(player_feet_px, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H_mat.astype(np.float64)).reshape(-1, 2)
        finite = np.all(np.isfinite(mapped), axis=1)
        if not np.any(finite):
            return float(fallback_x), 0
        m_fin = mapped[finite]
        inb_mask = (
            (m_fin[:, 0] >= -2.0)
            & (m_fin[:, 0] <= L + 2.0)
            & (m_fin[:, 1] >= -2.0)
            & (m_fin[:, 1] <= W + 2.0)
        )
        inb_count = int(np.sum(inb_mask))
        ref_xs = m_fin[inb_mask, 0] if inb_count >= 2 else np.clip(m_fin[:, 0], -10.0, L + 10.0)
        return float(np.median(ref_xs)), inb_count

    # Tier-2 relaxed thresholds for zoomed/corner/penalty-box angles
    _T2_KPT_CONF = 0.25
    _T2_MIN_POINTS = 4
    _T2_MIN_SPAN_M = 5.0
    _T2_MIN_HULL_M2 = 25.0
    _T2_MAX_REPROJ_M = 3.0
    _T2_STD_X_MIN = 6.0
    _T2_STD_Y_MIN = 5.0

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
    ) -> bool:
        """Core homography fitting with Jacobian orientation and horizon-fold validation."""
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
        if int(inl_mask.sum()) < 4:
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
        H_pos = H_cand / w_ref  # now projective depth w(ref_x, ref_y) == +1.0

        # Horizon-fold guard: all visible inliers must remain strictly in front of the horizon
        w_inl = H_pos[2, 0] * inl_src[:, 0] + H_pos[2, 1] * inl_src[:, 1] + H_pos[2, 2]
        if np.any(w_inl <= 0.08):
            return False

        # Orientation-preserving local 2D Jacobian check at inlier centroid
        probe = np.array(
            [[[ref_x, ref_y]], [[ref_x + 10.0, ref_y]], [[ref_x, ref_y + 10.0]]],
            dtype=np.float32,
        )
        p_probe = cv2.perspectiveTransform(probe, H_pos).reshape(3, 2)
        if not np.all(np.isfinite(p_probe)):
            return False
        dx_m = p_probe[1] - p_probe[0]
        dy_m = p_probe[2] - p_probe[0]
        jac_det = float(dx_m[0] * dy_m[1] - dx_m[1] * dy_m[0])
        if jac_det <= 0.0:
            return False

        # Inlier anchor centroid must map reasonably on or adjacent to the pitch
        if not (-12.0 <= p_probe[0, 0] <= self.pitch_cfg.length + 12.0):
            return False
        if not (-12.0 <= p_probe[0, 1] <= self.pitch_cfg.width + 12.0):
            return False

        proj = cv2.perspectiveTransform(inl_src.reshape(-1, 1, 2), H_pos).reshape(-1, 2)
        err_m = float(np.mean(np.linalg.norm(proj - inl_dst, axis=1)))
        gate_m = max_reproj if self.H is not None else max(5.5, max_reproj * 2.0)
        if err_m > gate_m:
            return False

        if abs(H_pos[2, 2]) < 1e-8:
            return False
        H_norm = H_pos / H_pos[2, 2]

        # Half-Pitch Symmetry Lock: prevent 180-deg penalty-box flips during continuous pan/zoom
        fallback_x = float(np.median(inl_dst[:, 0]))
        cand_mean_x, cand_inb = self._project_cloud_stats(H_norm, player_feet_px, fallback_x)
        if not is_scene_cut and self._anchor_mean_x is not None:
            if abs(cand_mean_x - self._anchor_mean_x) > 32.0:
                H_rot = self._rotate_180_pitch(H_norm)
                if H_rot is not None:
                    rot_fallback_x = float(self.pitch_cfg.length) - fallback_x
                    rot_mean_x, rot_inb = self._project_cloud_stats(H_rot, player_feet_px, rot_fallback_x)
                    if abs(rot_mean_x - self._anchor_mean_x) < 18.0 and rot_inb >= cand_inb:
                        H_norm = H_rot
                        cand_mean_x = rot_mean_x
                    else:
                        return False
                else:
                    return False

        try:
            H_inv_cand = np.linalg.inv(H_norm)
        except np.linalg.LinAlgError:
            return False

        if self.H is None:
            self.H = H_norm.copy()
        else:
            blended = (1.0 - self.ema_alpha) * self.H + self.ema_alpha * H_norm
            if abs(blended[2, 2]) > 1e-8:
                self.H = blended / blended[2, 2]
            else:
                self.H = H_norm.copy()

        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H = H_norm
            self.H_inv = H_inv_cand

        self._anchor_history.append(cand_mean_x)
        if len(self._anchor_history) > 10:
            self._anchor_history.pop(0)
        self._anchor_mean_x = float(np.median(self._anchor_history))
        self.last_was_reset = False

        self.valid_updates += 1
        self.reproj_errors_m.append(err_m)
        return True

    def update(
        self,
        kpts_xy: np.ndarray,
        kpts_conf: np.ndarray,
        player_feet_px: Optional[np.ndarray] = None,
        is_scene_cut: bool = False,
    ) -> bool:
        """Try Tier-1 (wide), Tier-1.5 (sparse midfield), then Tier-2 (corner/zoom). Hold briefly on loss."""
        self.last_was_reset = False
        if is_scene_cut:
            self.reset_scene_cut()

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
            )

        if ok:
            self._consecutive_failures = 0
            return True

        self._consecutive_failures += 1
        self.rejected_updates += 1
        if self._consecutive_failures > self.temporal_hold_frames and self.H is not None:
            # Sustained loss (close-up / replay) -> clear stale H
            self.H = None
            self.H_inv = None
            self.last_was_reset = True
        return self.H is not None

    def image_to_pitch(self, pts_xy: np.ndarray) -> Optional[np.ndarray]:
        if self.H is None or len(pts_xy) == 0:
            return None
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        L, W = float(self.pitch_cfg.length), float(self.pitch_cfg.width)
        # Soft-clamp near-touchline / corner-flag apron projections inside [0.3..104.7, 0.3..67.7]
        # while clamping extreme wild outliers (>4m outside bounds) to sentinel OOB [-2..107, -2..70].
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
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def summary(self) -> dict:
        total = self.valid_updates + self.rejected_updates
        return {
            "valid_frames": self.valid_updates,
            "rejected_frames": self.rejected_updates,
            "stability_rate_pct": round(100.0 * self.valid_updates / max(1, total), 2),
            "mean_inlier_reproj_error_m": round(float(np.mean(self.reproj_errors_m)), 3) if self.reproj_errors_m else None,
        }
