from typing import Optional
import cv2
import numpy as np

from .config import FIFAPitchConfig


class StabilizedHomography:
    def __init__(
        self,
        pitch_cfg: FIFAPitchConfig,
        kpt_conf_thresh: float = 0.45,
        ema_alpha: float = 0.30,
        max_reproj_m: float = 5.0,
    ):
        self.pitch_cfg = pitch_cfg
        self.pitch_vertices = pitch_cfg.vertices
        self.kpt_conf_thresh = kpt_conf_thresh
        self.ema_alpha = ema_alpha
        self.max_reproj_m = max_reproj_m

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.valid_updates = 0
        self.rejected_updates = 0
        self.reproj_errors_m = []

    def update(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> bool:
        valid_xy = (kpts_xy[:, 0] > 2.0) & (kpts_xy[:, 1] > 2.0)
        mask = valid_xy & (kpts_conf >= self.kpt_conf_thresh)
        # Adaptive fallback on tight midfield camera zoom
        if int(mask.sum()) < 5:
            mask = valid_xy & (kpts_conf >= 0.30)
        if int(mask.sum()) < 4:
            self.rejected_updates += 1
            return self.H is not None

        src_pts = kpts_xy[mask].astype(np.float32)
        dst_pts = self.pitch_vertices[mask].astype(np.float32)

        if np.std(src_pts[:, 0]) < 12.0 or np.std(src_pts[:, 1]) < 10.0:
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
        proj = cv2.perspectiveTransform(src_pts[inl_mask].reshape(-1, 1, 2), H_cand).reshape(-1, 2)
        err_m = float(np.mean(np.linalg.norm(proj - dst_pts[inl_mask], axis=1)))
        gate_m = self.max_reproj_m if self.H is not None else max(7.0, self.max_reproj_m)
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
