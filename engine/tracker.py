from typing import Optional, Tuple, List
import numpy as np
from scipy.optimize import linear_sum_assignment


class _LinearKalmanFilter:
    """Lightweight pure-NumPy linear Kalman Filter (zero external dependency)."""

    def __init__(self, dim_x: int, dim_z: int):
        self.x = np.zeros((dim_x, 1), dtype=np.float32)
        self.F = np.eye(dim_x, dtype=np.float32)
        self.H = np.zeros((dim_z, dim_x), dtype=np.float32)
        self.P = np.eye(dim_x, dtype=np.float32)
        self.R = np.eye(dim_z, dtype=np.float32)
        self.Q = np.eye(dim_x, dtype=np.float32)
        self._I = np.eye(dim_x, dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        z = np.asarray(z, dtype=np.float32).reshape(-1, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (self._I - K @ self.H) @ self.P


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class _Tracklet:
    """Single player Kalman-smoothed bounding box tracklet for 2-stage ByteTrack."""

    def __init__(self, track_id: int, box: np.ndarray, conf: float, cls_id: int):
        self.track_id = track_id
        self.box = box.astype(np.float32).copy()
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.time_since_update = 0
        self.hits = 1

        self.kf = _LinearKalmanFilter(dim_x=6, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0.85, 0],
            [0, 0, 0, 0, 0, 0.85],
        ], dtype=np.float32)
        self.kf.H[0, 0] = self.kf.H[1, 1] = self.kf.H[2, 2] = self.kf.H[3, 3] = 1.0
        self.kf.R *= 4.0
        self.kf.P *= 20.0
        self.kf.Q = np.diag([1.0, 1.0, 2.0, 2.0, 4.0, 4.0]).astype(np.float32)

        cx, cy, w, h = self._xyxy_to_cxcywh(box)
        self.kf.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32)

    @staticmethod
    def _xyxy_to_cxcywh(b: np.ndarray) -> Tuple[float, float, float, float]:
        w = max(2.0, float(b[2] - b[0]))
        h = max(2.0, float(b[3] - b[1]))
        cx = float(b[0]) + w * 0.5
        cy = float(b[1]) + h * 0.5
        return cx, cy, w, h

    def predict(self) -> np.ndarray:
        self.kf.predict()
        cx, cy, w, h = self.kf.x[:4, 0]
        w, h = max(4.0, float(w)), max(4.0, float(h))
        self.time_since_update += 1
        return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)

    def update(self, box: np.ndarray, conf: float, cls_id: int):
        cx, cy, w, h = self._xyxy_to_cxcywh(box)
        self.kf.update(np.array([[cx], [cy], [w], [h]], dtype=np.float32))
        self.box = box.astype(np.float32).copy()
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.time_since_update = 0
        self.hits += 1


class PlayerByteTracker:
    """Self-contained 2-stage ByteTrack (Kalman + Hungarian IoU association)."""

    def __init__(
        self,
        fps: float = 25.0,
        high_thresh: float = 0.45,
        low_thresh: float = 0.15,
        iou_thresh: float = 0.22,
    ):
        self.max_age = max(20, int(fps * 1.5))
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.tracks: List[_Tracklet] = []
        self.next_id = 1

    def _associate(
        self,
        pred_boxes: np.ndarray,
        det_boxes: np.ndarray,
        min_iou: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(pred_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(pred_boxes))), list(range(len(det_boxes)))

        iou = _iou_matrix(pred_boxes, det_boxes)
        row_ind, col_ind = linear_sum_assignment(1.0 - iou)

        matched = []
        unmatched_trks = set(range(len(pred_boxes)))
        unmatched_dets = set(range(len(det_boxes)))
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= min_iou:
                matched.append((int(r), int(c)))
                unmatched_trks.discard(int(r))
                unmatched_dets.discard(int(c))
        return matched, sorted(unmatched_trks), sorted(unmatched_dets)

    def update(
        self,
        boxes: np.ndarray,
        confs: np.ndarray,
        clss: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred_boxes = np.array([t.predict() for t in self.tracks], dtype=np.float32).reshape(-1, 4)

        if len(boxes) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        high_idx = np.where(confs >= self.high_thresh)[0]
        low_idx = np.where((confs >= self.low_thresh) & (confs < self.high_thresh))[0]

        matched_1, unm_trks_1, unm_high = self._associate(pred_boxes, boxes[high_idx], self.iou_thresh)
        for t_i, d_rel in matched_1:
            d_i = int(high_idx[d_rel])
            self.tracks[t_i].update(boxes[d_i], confs[d_i], clss[d_i])

        if len(unm_trks_1) > 0 and len(low_idx) > 0:
            rem_preds = pred_boxes[unm_trks_1]
            matched_2, _, _ = self._associate(rem_preds, boxes[low_idx], max(0.15, self.iou_thresh - 0.05))
            for rel_t, d_rel in matched_2:
                t_i = unm_trks_1[rel_t]
                d_i = int(low_idx[d_rel])
                self.tracks[t_i].update(boxes[d_i], confs[d_i], clss[d_i])

        for d_rel in unm_high:
            d_i = int(high_idx[d_rel])
            self.tracks.append(_Tracklet(self.next_id, boxes[d_i], confs[d_i], clss[d_i]))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        active = [t for t in self.tracks if t.time_since_update == 0]
        if not active:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        out_boxes = np.array([t.box for t in active], dtype=np.float32)
        out_confs = np.array([t.conf for t in active], dtype=np.float32)
        out_clss = np.array([t.cls_id for t in active], dtype=np.int32)
        out_tids = np.array([t.track_id for t in active], dtype=np.int32)
        return out_boxes, out_confs, out_clss, out_tids


class KalmanBallTracker:
    """Constant-Velocity 2D Kalman Filter for high-speed football tracking."""

    def __init__(self, max_jump_px: float = 160.0, max_coast_frames: int = 8):
        self.max_jump_px = max_jump_px
        self.max_coast_frames = max_coast_frames
        self.kf = self._create_kf()
        self.initialized = False
        self.missing_count = 0

    def _create_kf(self) -> _LinearKalmanFilter:
        kf = _LinearKalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0.92, 0],
            [0, 0, 0, 0.92],
        ], dtype=np.float32)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.R *= 6.0
        kf.P *= 50.0
        kf.Q = np.diag([2.0, 2.0, 12.0, 12.0]).astype(np.float32)
        return kf

    def update(
        self,
        ball_boxes: np.ndarray,
        ball_confs: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], bool]:
        pred_xy = None
        if self.initialized:
            self.kf.predict()
            pred_xy = self.kf.x[:2, 0].copy()

        chosen_xy = None
        if len(ball_boxes) > 0:
            centers = np.stack([
                (ball_boxes[:, 0] + ball_boxes[:, 2]) * 0.5,
                (ball_boxes[:, 1] + ball_boxes[:, 3]) * 0.5,
            ], axis=-1)

            if not self.initialized:
                best_idx = int(np.argmax(ball_confs))
                chosen_xy = centers[best_idx]
            else:
                dists = np.linalg.norm(centers - pred_xy[None, :], axis=1)
                gate = self.max_jump_px * (1.0 + 0.35 * self.missing_count)
                valid = np.where(dists <= gate)[0]
                if len(valid) > 0:
                    best_idx = valid[int(np.argmin(dists[valid]))]
                    chosen_xy = centers[best_idx]
                elif self.missing_count >= self.max_coast_frames:
                    best_idx = int(np.argmax(ball_confs))
                    if ball_confs[best_idx] >= 0.35:
                        self.kf = self._create_kf()
                        chosen_xy = centers[best_idx]
                        self.initialized = False

        if chosen_xy is not None:
            if not self.initialized:
                self.kf.x = np.array([[chosen_xy[0]], [chosen_xy[1]], [0.0], [0.0]], dtype=np.float32)
                self.initialized = True
            else:
                self.kf.update(chosen_xy.reshape(2, 1))
            self.missing_count = 0
            return self.kf.x[:2, 0].copy(), True

        if self.initialized and self.missing_count < self.max_coast_frames:
            self.missing_count += 1
            return pred_xy, False

        self.missing_count += 1
        return None, False


def interpolate_ball_track(
    raw_positions: List[Optional[np.ndarray]],
    max_gap: int = 12,
) -> List[Optional[np.ndarray]]:
    out = [p.copy() if p is not None else None for p in raw_positions]
    known_idx = [i for i, p in enumerate(out) if p is not None]
    if len(known_idx) < 2:
        return out

    for k in range(len(known_idx) - 1):
        i0, i1 = known_idx[k], known_idx[k + 1]
        gap = i1 - i0 - 1
        if 0 < gap <= max_gap:
            p0, p1 = out[i0], out[i1]
            for step in range(1, gap + 1):
                alpha = step / (gap + 1.0)
                out[i0 + step] = (1.0 - alpha) * p0 + alpha * p1
    return out
