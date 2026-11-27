import importlib.util
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO

from .config import EngineConfig


class DualModelDetector:
    """
    Runs our custom-trained models per frame:
      1. Match Detector (YOLOv11-m @ 1280px) with per-class calibrated thresholds.
         IMPORTANT: Runs single-pass `predict()` so 100% of raw low-conf/fast-moving
         `ball` detections are preserved untouched, then feeds ONLY person detections
         (players, goalkeepers, referees) into BoT-SORT + sparseOptFlow GMC when `lap`
         is installed (avoiding BoT-SORT discarding fast-moving ball boxes).
      2. Pitch Keypoints (YOLOv11-m-pose @ 640px) detecting 32 geometric pitch landmarks
         (skipped completely when PnLCalib 3D is active on GPU!).
    """

    def __init__(self, cfg: EngineConfig, load_pitch_model: bool = True):
        self.cfg = cfg
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.botsort_available = bool(
            importlib.util.find_spec("lap") or importlib.util.find_spec("lapx")
        )
        self._botsort: Any = None
        self._botsort_fps: float = 25.0

        # Load calibrated thresholds from training artifacts if present
        if cfg.thresholds_path and Path(cfg.thresholds_path).exists():
            with open(cfg.thresholds_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            ct = calib.get("class_thresholds", {})
            # Keep ball slightly more sensitive for video (high 95.4% precision + Kalman outlier rejection)
            self.cfg.ball_conf = min(cfg.ball_conf, float(ct.get("ball", cfg.ball_conf)))
            self.cfg.goalkeeper_conf = float(ct.get("goalkeeper", cfg.goalkeeper_conf))
            self.cfg.player_conf = min(0.55, float(ct.get("player", cfg.player_conf)))
            self.cfg.referee_conf = float(ct.get("referee", cfg.referee_conf))

        self.players_model = YOLO(cfg.players_model_path)
        self.pitch_model = YOLO(cfg.pitch_model_path) if (load_pitch_model and cfg.pitch_model_path) else None

        self.cls_thresholds = {
            cfg.ball_id: cfg.ball_conf,
            cfg.goalkeeper_id: cfg.goalkeeper_conf,
            cfg.player_id: cfg.player_conf,
            cfg.referee_id: cfg.referee_conf,
        }
        self.min_conf = min(self.cls_thresholds.values())

    def reset_tracker(self, fps: float = 25.0) -> None:
        """Initialize or reset standalone person-only BoT-SORT + sparseOptFlow GMC."""
        self._botsort = None
        self._botsort_fps = max(1.0, float(fps))
        if not self.botsort_available:
            return
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.utils import YAML, IterableSimpleNamespace
            from ultralytics.utils.checks import check_yaml

            tcfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
            tcfg.with_reid = False
            tcfg.gmc_method = "sparseOptFlow"
            self._botsort = BOTSORT(args=tcfg, frame_rate=int(round(self._botsort_fps)))
        except Exception as e:
            print(f"[Detector] BoT-SORT init skipped ({e}); using ByteTrack fallback.")
            self.botsort_available = False
            self._botsort = None

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        run_pitch: bool = True,
        use_botsort: bool = False,
    ) -> Dict[str, Any]:
        # 1. Single-pass YOLO predict(): preserves 100% of raw ball detections!
        det_res = self.players_model.predict(
            frame_bgr,
            imgsz=self.cfg.det_imgsz,
            conf=self.min_conf,
            iou=self.cfg.nms_iou,
            device=self.device,
            verbose=False,
        )[0]

        if len(det_res.boxes) > 0:
            boxes = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = det_res.boxes.conf.cpu().numpy().astype(np.float32)
            clss = det_res.boxes.cls.cpu().numpy().astype(np.int32)

            keep = np.array(
                [cf >= self.cls_thresholds.get(int(c), 0.45) for c, cf in zip(clss, confs)],
                dtype=bool,
            )
            boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
            clss = np.zeros((0,), dtype=np.int32)

        ball_mask = clss == self.cfg.ball_id
        person_mask = ~ball_mask

        person_boxes = boxes[person_mask]
        person_confs = confs[person_mask]
        person_clss = clss[person_mask]
        person_tids: Optional[np.ndarray] = None

        # Run BoT-SORT + sparseOptFlow strictly on person boxes (never filtering ball!)
        if use_botsort and self.botsort_available:
            if self._botsort is None:
                self.reset_tracker(self._botsort_fps)
            if self._botsort is not None and len(person_boxes) > 0:
                try:
                    from ultralytics.engine.results import Boxes

                    # Build clean Ultralytics Boxes view for persons only: [x1, y1, x2, y2, conf, cls]
                    p_data = np.column_stack(
                        [person_boxes, person_confs[:, None], person_clss.astype(np.float32)[:, None]]
                    ).astype(np.float32)
                    p_boxes_obj = Boxes(torch.from_numpy(p_data), orig_shape=frame_bgr.shape[:2]).cpu().numpy()
                    tracks = self._botsort.update(p_boxes_obj, frame_bgr)
                    if tracks is not None and len(tracks) > 0:
                        person_boxes = tracks[:, 0:4].astype(np.float32)
                        person_tids = tracks[:, 4].astype(np.int32)
                        person_confs = tracks[:, 5].astype(np.float32)
                        person_clss = tracks[:, 6].astype(np.int32)
                    else:
                        person_boxes = np.zeros((0, 4), dtype=np.float32)
                        person_tids = np.zeros((0,), dtype=np.int32)
                        person_confs = np.zeros((0,), dtype=np.float32)
                        person_clss = np.zeros((0,), dtype=np.int32)
                except Exception as e:
                    print(f"[Detector] BoT-SORT runtime fallback ({e}); switching to ByteTrack.")
                    self.botsort_available = False
                    self._botsort = None
                    person_tids = None

        # 2. Pitch 32-Keypoints Detection @ 640px (only when fallback pose calibrator is active)
        kpts_xy = np.zeros((32, 2), dtype=np.float32)
        kpts_conf = np.zeros((32,), dtype=np.float32)
        if run_pitch and self.pitch_model is not None:
            pitch_res = self.pitch_model.predict(
                frame_bgr,
                imgsz=self.cfg.pitch_imgsz,
                conf=0.15,
                max_det=1,
                device=self.device,
                verbose=False,
            )[0]

            if pitch_res.keypoints is not None and len(pitch_res.keypoints.xy) > 0:
                kpts_xy = pitch_res.keypoints.xy[0].cpu().numpy().astype(np.float32)
                if pitch_res.keypoints.conf is not None:
                    kpts_conf = pitch_res.keypoints.conf[0].cpu().numpy().astype(np.float32)
                else:
                    kpts_conf = ((kpts_xy[:, 0] > 0) & (kpts_xy[:, 1] > 0)).astype(np.float32)

        return {
            "person_boxes": person_boxes,
            "person_confs": person_confs,
            "person_clss": person_clss,
            "person_tids": person_tids,
            "ball_boxes": boxes[ball_mask],
            "ball_confs": confs[ball_mask],
            "pitch_kpts_xy": kpts_xy,
            "pitch_kpts_conf": kpts_conf,
        }
