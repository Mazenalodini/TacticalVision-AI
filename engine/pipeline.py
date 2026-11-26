import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
from tqdm import tqdm

from .config import EngineConfig, FIFAPitchConfig
from .detector import DualModelDetector
from .tracker import PlayerByteTracker, KalmanBallTracker, interpolate_ball_track
from .team_classifier import SigLIPTeamClassifier, stabilize_team_assignments
from .jersey_recognizer import JerseyRecognizer
from .homography import StabilizedHomography
from .tactics import TacticalAnalyticsEngine
from .renderer import BroadcastRenderer, transcode_to_h264


class MatchAnalysisPipeline:
    """
    End-to-End 2-Pass Football Tactical Analysis Engine:
      Pass 1: Dual-YOLOv11 Detection + ByteTrack + Kalman Ball + Calibrated Homography + Person Crops
      Stage 2: SigLIP + UMAP/PCA Team Fitting + Model 4 Jersey Legibility/PARSeq Voting
      Stage 3: Ball Interpolation + Kinematics Smoothing + Tactical Telemetry Extraction
      Pass 2: Broadcast Video + 2D Tactical Radar Rendering + Web H.264 Encoding + JSON Export
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.pitch = FIFAPitchConfig()
        self.detector = DualModelDetector(cfg)
        self.team_classifier = SigLIPTeamClassifier()
        self.jersey_recognizer = JerseyRecognizer(
            onnx_path=cfg.jersey_legibility_onnx_path or None,
            pt_path=cfg.jersey_legibility_pt_path or None,
            legibility_threshold=cfg.jersey_legibility_threshold,
            min_track_ocr_conf=cfg.jersey_min_track_ocr_conf,
            parseq_min_crop_conf=cfg.jersey_parseq_min_crop_conf,
            max_crops_per_track=cfg.jersey_max_crops_per_track,
            torso_crop_box=cfg.jersey_torso_crop_box,
            enable_ocr=cfg.enable_jersey_ocr,
        )

    @staticmethod
    def _crop_full_person(frame_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 16:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        return crop.copy() if crop.size > 0 else None

    def run(
        self,
        video_path: str,
        output_dir: str,
        max_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        vpath = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vpath))
        assert cap.isOpened(), f"Cannot open video: {vpath}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_seconds is not None:
            total_frames = min(total_frames, int(fps * max_seconds))

        print(f"\n[Pipeline] Analyzing: {vpath.name} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")

        player_tracker = PlayerByteTracker(fps=fps)
        ball_tracker = KalmanBallTracker(max_jump_px=max(120.0, width * 0.10))
        homography = StabilizedHomography(
            self.pitch,
            kpt_conf_thresh=self.cfg.pitch_kpt_conf,
            ema_alpha=self.cfg.homography_ema_alpha,
            max_reproj_m=self.cfg.max_homography_reproj_m,
            min_span_m=self.cfg.min_span_m,
            min_convex_hull_area_m2=self.cfg.min_convex_hull_area_m2,
        )

        # ======================================================================
        # PASS 1: Detection, Tracking, Homography & Person Sampling
        # ======================================================================
        frame_records = []
        raw_ball_xy = []
        jersey_samples = []
        sample_stride = max(1, int(fps // 2))       # Sample team jersey colors ~2 Hz
        full_crop_stride = max(1, int(fps // 4))    # Sample full person crops ~4 Hz for Model 4
        pitch_stride = max(1, int(self.cfg.pitch_stride))

        pbar = tqdm(range(total_frames), desc="Pass 1/2 (CV Detection + Tracking)")
        for f_idx in pbar:
            ok, frame = cap.read()
            if not ok:
                break

            run_pitch = (homography.H is None) or (f_idx % pitch_stride == 0)
            det = self.detector.infer_frame(frame, run_pitch=run_pitch)
            boxes, confs, clss, tids = player_tracker.update(
                det["person_boxes"], det["person_confs"], det["person_clss"]
            )
            ball_xy, ball_det_ok = ball_tracker.update(det["ball_boxes"], det["ball_confs"])
            raw_ball_xy.append(ball_xy)

            if run_pitch:
                homography.update(det["pitch_kpts_xy"], det["pitch_kpts_conf"])

            # Project player foot-points (bottom center of bounding box) onto FIFA pitch (meters)
            entities = []
            if len(boxes) > 0:
                feet_px = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, boxes[:, 3]], axis=-1)
                feet_m = homography.image_to_pitch(feet_px)
            else:
                feet_m = None

            for i in range(len(boxes)):
                crop = None
                full_crop = None
                c_cls = int(clss[i])
                if c_cls == self.cfg.player_id:
                    crop = self.team_classifier.crop_jersey(frame, boxes[i])
                    if crop is not None and (f_idx % sample_stride == 0):
                        jersey_samples.append(crop)

                if (
                    self.cfg.enable_jersey_ocr
                    and c_cls in (self.cfg.player_id, self.cfg.goalkeeper_id)
                    and (f_idx % full_crop_stride == 0)
                ):
                    full_crop = self._crop_full_person(frame, boxes[i])

                entities.append({
                    "track_id": int(tids[i]),
                    "cls": c_cls,
                    "conf": float(confs[i]),
                    "box": boxes[i],
                    "pitch_xy": feet_m[i].copy() if feet_m is not None else None,
                    "crop": crop,
                    "full_crop": full_crop,
                })

            frame_records.append({
                "frame_idx": f_idx,
                "entities": entities,
                "H": homography.H.copy() if homography.H is not None else None,
                "H_inv": homography.H_inv.copy() if homography.H_inv is not None else None,
                "ball_detected": ball_det_ok,
            })

        cap.release()

        # Backfill any early frames before the first valid Homography so radar is populated
        first_valid = next((r for r in frame_records if r["H"] is not None), None)
        if first_valid is not None:
            H0, H0_inv = first_valid["H"], first_valid["H_inv"]
            for rec in frame_records:
                if rec["H"] is None:
                    rec["H"], rec["H_inv"] = H0.copy(), H0_inv.copy()
                    for ent in rec["entities"]:
                        b = ent["box"]
                        foot = np.array([[[(b[0] + b[2]) * 0.5, b[3]]]], dtype=np.float32)
                        m = cv2.perspectiveTransform(foot, H0).reshape(2)
                        m[0] = np.clip(m[0], -2.0, self.pitch.length + 2.0)
                        m[1] = np.clip(m[1], -2.0, self.pitch.width + 2.0)
                        ent["pitch_xy"] = m

        # ======================================================================
        # STAGE 2: Ball Interpolation, Team Classification & Model 4 Jersey OCR
        # ======================================================================
        print("[Stage 2] Fitting SigLIP Team Classifier & Running Model 4 Jersey Recognition...")
        self.team_classifier.fit(jersey_samples)
        stabilize_team_assignments(
            frame_records,
            self.team_classifier,
            player_id=self.cfg.player_id,
            goalkeeper_id=self.cfg.goalkeeper_id,
        )

        if self.cfg.enable_jersey_ocr:
            jersey_assignments, jersey_diag = self.jersey_recognizer.recognize_tracks(
                frame_records,
                player_id=self.cfg.player_id,
                goalkeeper_id=self.cfg.goalkeeper_id,
            )
        else:
            # Clear temporary crops if OCR is explicitly disabled
            for rec in frame_records:
                for ent in rec["entities"]:
                    ent.pop("full_crop", None)
            jersey_assignments = {}
            jersey_diag = {
                "gate_enabled": False,
                "gate_backend": "disabled",
                "ocr_available": False,
                "ocr_status": "disabled_by_user",
                "tracks_evaluated": 0,
                "legible_crops": 0,
                "verified_players": 0,
            }

        smooth_ball_xy = interpolate_ball_track(raw_ball_xy, max_gap=12)
        ball_tracked_frames = 0
        for rec, bxy in zip(frame_records, smooth_ball_xy):
            rec["ball_xy"] = bxy
            if bxy is not None:
                ball_tracked_frames += 1
                if rec["H"] is not None:
                    bp = cv2.perspectiveTransform(bxy.reshape(1, 1, 2).astype(np.float32), rec["H"]).reshape(2)
                    bp[0] = np.clip(bp[0], 0.0, self.pitch.length)
                    bp[1] = np.clip(bp[1], 0.0, self.pitch.width)
                    rec["ball_pitch_xy"] = bp
                else:
                    rec["ball_pitch_xy"] = None
            else:
                rec["ball_pitch_xy"] = None

        # ======================================================================
        # STAGE 3: Physical & Tactical Telemetry Extraction
        # ======================================================================
        print("[Stage 3] Computing Speeds, Sprints, Passes, Voronoi Control, Convex Hull & xT...")
        tactics_engine = TacticalAnalyticsEngine(self.cfg, self.pitch, fps=fps)
        telemetry = tactics_engine.analyze(frame_records, jersey_assignments=jersey_assignments)

        telemetry["match_info"] = {
            "source_video": vpath.name,
            "resolution": [width, height],
            "fps": round(fps, 2),
            "pitch_dimensions_m": [self.pitch.length, self.pitch.width],
            "team_colors": self.cfg.team_colors_hex,
        }
        telemetry["engine_diagnostics"] = {
            "homography": homography.summary(),
            "ball_coverage_pct": round(100.0 * ball_tracked_frames / max(1, len(frame_records)), 2),
            "processing_time_s": 0.0,
            "jersey_recognition": jersey_diag,
        }

        # ======================================================================
        # PASS 2: Render Broadcast Video + 2D Tactical Radar Video
        # ======================================================================
        renderer = BroadcastRenderer(self.cfg, self.pitch)
        raw_broadcast = out_dir / "_raw_broadcast.mp4"
        raw_radar = out_dir / "_raw_radar.mp4"
        final_broadcast = out_dir / "broadcast_analyzed.mp4"
        final_radar = out_dir / "tactical_radar.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w_bcast = cv2.VideoWriter(str(raw_broadcast), fourcc, fps, (width, height))
        w_radar = cv2.VideoWriter(str(raw_radar), fourcc, fps, (renderer.radar_w, renderer.radar_h))

        cap = cv2.VideoCapture(str(vpath))
        preview_frames = []
        preview_indices = set(np.linspace(0, max(0, len(frame_records) - 1), 4, dtype=int).tolist())

        for rec in tqdm(frame_records, desc="Pass 2/2 (Broadcast & Radar Rendering)"):
            ok, frame = cap.read()
            if not ok:
                break
            b_frame = renderer.draw_broadcast_frame(frame, rec, fps=fps)
            r_frame = renderer.draw_radar_frame(rec)
            w_bcast.write(b_frame)
            w_radar.write(r_frame)

            if rec["frame_idx"] in preview_indices:
                preview_frames.append((rec["frame_idx"], b_frame.copy(), r_frame.copy()))

        cap.release()
        w_bcast.release()
        w_radar.release()

        print("[Encoding] Transcoding videos to browser-ready H.264...")
        transcode_to_h264(raw_broadcast, final_broadcast)
        transcode_to_h264(raw_radar, final_radar)

        if preview_frames:
            _, bf, rf = preview_frames[len(preview_frames) // 2]
            rf_scaled = cv2.resize(rf, (int(rf.shape[1] * (bf.shape[0] / rf.shape[0])), bf.shape[0]))
            side_by_side = np.hstack([bf, rf_scaled])
            cv2.imwrite(str(out_dir / "preview_snapshot.png"), side_by_side)

        elapsed = round(time.perf_counter() - t_start, 2)
        telemetry["engine_diagnostics"]["processing_time_s"] = elapsed

        telemetry_path = out_dir / "telemetry.json"
        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, ensure_ascii=False)

        print(f"[Done] Saved outputs to: {out_dir} (Total time: {elapsed}s)")
        return {
            "broadcast_video": str(final_broadcast),
            "radar_video": str(final_radar),
            "telemetry_json": str(telemetry_path),
            "preview_image": str(out_dir / "preview_snapshot.png"),
            "telemetry": telemetry,
        }
