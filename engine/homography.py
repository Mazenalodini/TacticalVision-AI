from typing import Optional
import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .config import FIFAPitchConfig


class StabilizedHomography:
    """RANSAC planar homography with geometric spread/convex-hull gates and EMA smoothing."""

    def __init__(
        self,
        pitch_cfg: FIFAPitchConfig,
        kpt_conf_thresh: float = 0.30,
        ema_alpha: float = 0.20,
        max_reproj_m: float = 2.5,
        min_span_m: float = 15.0,
        min_convex_hull_area_m2: float = 150.0,
    ):
        self.pitch_cfg = pitch_cfg
        self.pitch_vertices = pitch_cfg.vertices
        self.kpt_conf_thresh = kpt_conf_thresh
        self.ema_alpha = ema_alpha
        self.max_reproj_m = max_reproj_m
        self.min_span_m = min_span_m
        self.min_convex_hull_area_m2 = min_convex_hull_area_m2

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.valid_updates = 0
        self.rejected_updates = 0
        self.reproj_errors_m = []

    def update(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> bool:
        valid_xy = (kpts_xy[:, 0] > 2.0) & (kpts_xy[:, 1] > 2.0)
        mask = valid_xy & (kpts_conf >= self.kpt_conf_thresh)
        # Adaptive relaxation when visible landmarks are sparse in tight midfield zooms
        if int(mask.sum()) < 5:
            mask = valid_xy & (kpts_conf >= min(0.25, self.kpt_conf_thresh))
        if int(mask.sum()) < 4:
            self.rejected_updates += 1
            return self.H is not None

        src_pts = kpts_xy[mask].astype(np.float32)
        dst_pts = self.pitch_vertices[mask].astype(np.float32)

        # Reject tight pixel clusters that blow up projective scale
        if np.std(src_pts[:, 0]) < 12.0 or np.std(src_pts[:, 1]) < 10.0:
            self.rejected_updates += 1
            return self.H is not None

        # Geometric Pitch Span & Convex-Hull Area Gate (prevents collinear corner blowups)
        span_diag_m = float(np.hypot(np.ptp(dst_pts[:, 0]), np.ptp(dst_pts[:, 1])))
        if span_diag_m < self.min_span_m:
            self.rejected_updates += 1
            return self.H is not None

        try:
            hull_area_m2 = float(ConvexHull(dst_pts).volume)  # volume = area in 2D
        except QhullError:
            hull_area_m2 = 0.0
        if hull_area_m2 < self.min_convex_hull_area_m2:
            self.rejected_updates += 1
            return self.H is not None

        H_cand, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_cand is None or inliers is None or int(inliers.sum()) < 4:
            self.rejected_updates += 1
            return self.H is not None

        if abs(H_cand[2, 2]) < 1e-8 or not np.all(np.isfinite(H_cand)):
            self.rejected_updates += 1
            return self.H is not None

        H_cand = H_cand / H_cand[2, 2]

        inl_mask = inliers.ravel().astype(bool)
        if int(inl_mask.sum()) < 4:
            self.rejected_updates += 1
            return self.H is not None

        # Also check geometric span across RANSAC inliers specifically
        inl_dst = dst_pts[inl_mask]
        if float(np.hypot(np.ptp(inl_dst[:, 0]), np.ptp(inl_dst[:, 1]))) < self.min_span_m:
            self.rejected_updates += 1
            return self.H is not None

        proj = cv2.perspectiveTransform(src_pts[inl_mask].reshape(-1, 1, 2), H_cand).reshape(-1, 2)
        err_m = float(np.mean(np.linalg.norm(proj - inl_dst, axis=1)))
        gate_m = self.max_reproj_m if self.H is not None else max(5.5, self.max_reproj_m * 2.0)
        if err_m > gate_m:
            self.rejected_updates += 1
            return self.H is not None

        try:
            H_inv_cand = np.linalg.inv(H_cand)
        except np.linalg.LinAlgError:
            self.rejected_updates += 1
            return self.H is not None

        if self.H is None:
            self.H = H_cand.copy()
        else:
            blended = (1.0 - self.ema_alpha) * self.H + self.ema_alpha * H_cand
            if abs(blended[2, 2]) > 1e-8:
                self.H = blended / blended[2, 2]
            else:
                self.H = H_cand.copy()

        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H = H_cand
            self.H_inv = H_inv_cand

        self.valid_updates += 1
        self.reproj_errors_m.append(err_m)
        return True

    def image_to_pitch(self, pts_xy: np.ndarray) -> Optional[np.ndarray]:
        if self.H is None or len(pts_xy) == 0:
            return None
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        mapped[:, 0] = np.clip(mapped[:, 0], -2.0, self.pitch_cfg.length + 2.0)
        mapped[:, 1] = np.clip(mapped[:, 1], -2.0, self.pitch_cfg.width + 2.0)
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
