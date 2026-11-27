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
    Runs our two custom-trained models per frame:
      1. Match Detector (YOLOv11-m @ 1280px) with BoT-SORT + sparseOptFlow GMC tracking
         and per-class calibrated thresholds.
      2. Pitch Keypoints (YOLOv11-m-pose @ 640px) detecting 32 geometric pitch landmarks.
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.botsort_available = bool(
            importlib.util.find_spec("lap") or importlib.util.find_spec("lapx")
        )
        self.tracker_cfg = "botsort.yaml"

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
        self.pitch_model = YOLO(cfg.pitch_model_path)

        self.cls_thresholds = {
            cfg.ball_id: cfg.ball_conf,
            cfg.goalkeeper_id: cfg.goalkeeper_conf,
            cfg.player_id: cfg.player_conf,
            cfg.referee_id: cfg.referee_conf,
        }
        self.min_conf = min(self.cls_thresholds.values())

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        run_pitch: bool = True,
        use_botsort: bool = False,
    ) -> Dict[str, Any]:
        # 1. Match Object Detection / BoT-SORT+GMC Tracking @ 1280px (or 640px in cpu_quick)
        person_tids: Optional[np.ndarray] = None
        if use_botsort and self.botsort_available:
            try:
                det_res = self.players_model.track(
                    frame_bgr,
                    imgsz=self.cfg.det_imgsz,
                    conf=self.min_conf,
                    iou=self.cfg.nms_iou,
                    device=self.device,
                    persist=True,
                    tracker=self.tracker_cfg,
                    verbose=False,
                )[0]
            except Exception:
                self.botsort_available = False
                det_res = self.players_model.predict(
                    frame_bgr,
                    imgsz=self.cfg.det_imgsz,
                    conf=self.min_conf,
                    iou=self.cfg.nms_iou,
                    device=self.device,
                    verbose=False,
                )[0]
        else:
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
            raw_ids = (
                det_res.boxes.id.cpu().numpy().astype(np.int32)
                if (use_botsort and det_res.boxes.id is not None)
                else None
            )

            keep = np.array(
                [cf >= self.cls_thresholds.get(int(c), 0.45) for c, cf in zip(clss, confs)],
                dtype=bool,
            )
            boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
            if raw_ids is not None:
                raw_ids = raw_ids[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
            clss = np.zeros((0,), dtype=np.int32)
            raw_ids = None

        # Split into entities
        ball_mask = clss == self.cfg.ball_id
        person_mask = ~ball_mask
        if raw_ids is not None:
            person_tids = raw_ids[person_mask]

        # 2. Pitch 32-Keypoints Detection @ 640px (skipped when stride > 1 and homography locked)
        kpts_xy = np.zeros((32, 2), dtype=np.float32)
        kpts_conf = np.zeros((32,), dtype=np.float32)
        if run_pitch:
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
            "person_boxes": boxes[person_mask],
            "person_confs": confs[person_mask],
            "person_clss": clss[person_mask],
            "person_tids": person_tids,
            "ball_boxes": boxes[ball_mask],
            "ball_confs": confs[ball_mask],
            "pitch_kpts_xy": kpts_xy,
            "pitch_kpts_conf": kpts_conf,
        }
