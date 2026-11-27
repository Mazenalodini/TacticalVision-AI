import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
import torch
from tqdm import tqdm

from .config import EngineConfig, FIFAPitchConfig
from .detector import DualModelDetector
from .tracker import PlayerByteTracker, KalmanBallTracker, interpolate_ball_track
from .team_classifier import SigLIPTeamClassifier, stabilize_team_assignments
from .jersey_recognizer import JerseyRecognizer
from .homography import StabilizedHomography
from .tactics import TacticalAnalyticsEngine
from .renderer import BroadcastRenderer, transcode_to_h264
from .video_feature_extractor import SoccerNetVideoFeatureExtractor
from .action_spotter import ActionSpotter


class MatchAnalysisPipeline:
    """
    End-to-End Unified 4-Model Football Tactical Analysis Engine:
      Pass 1: Dual-YOLOv11 Detection + ByteTrack + Kalman Ball + Homography + Crops + 2FPS ResNet Sampling
      Stage 1b: Model 3 Raw-Video Broadcast Action Spotting (ResNet-152 ONNX + PCA-512 -> ActionSpotter INT8)
      Stage 2: Team Clustering + Model 4 Jersey Legibility/OCR Voting
      Stage 3: Ball Interpolation + Kinematics Smoothing + Tactical Telemetry Extraction
      Pass 2: Broadcast Video + 2D Tactical Radar Rendering + Web H.264 Encoding + JSON Export
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.pitch = FIFAPitchConfig()
        print("[Init] Loading Dual-YOLOv11 detector & 32-keypoint pose weights...")
        self.detector: Optional[DualModelDetector] = DualModelDetector(cfg)
        # Stage 1b & Stage 2 models are initialized lazily AFTER detector unload
        # to prevent peak RAM OOM on long CPU clips.
        self.team_classifier: Optional[SigLIPTeamClassifier] = None
        self.jersey_recognizer: Optional[JerseyRecognizer] = None
        self.feature_extractor: Optional[SoccerNetVideoFeatureExtractor] = None
        self.action_spotter: Optional[ActionSpotter] = None

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

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
        start_seconds: float = 0.0,
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
        raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, int(round(max(0.0, float(start_seconds or 0.0)) * fps)))
        if start_frame > 0 and start_frame < raw_total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        remaining_frames = max(0, raw_total_frames - start_frame)
        # max_seconds=0 or None means "Full Video" (from start_seconds to end) — no truncation
        if max_seconds is not None and max_seconds > 0:
            total_frames = min(remaining_frames, int(round(fps * max_seconds)))
        else:
            total_frames = remaining_frames

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

        if self.detector is None:
            print("[Init] Loading Dual-YOLOv11 detector & 32-keypoint pose weights...")
            self.detector = DualModelDetector(self.cfg)

        # ======================================================================
        # PASS 1: Detection, Tracking, Homography & Bounded Person Sampling
        # ======================================================================
        frame_records = []
        raw_ball_xy = []
        jersey_samples = []
        spotting_frames_bgr = []
        sample_stride = max(1, int(fps // 2))       # Sample team jersey colors ~2 Hz
        spotting_stride = max(1, int(round(fps / 2.0)))  # Sample 2FPS frames for Model 3 ResNet-152
        spotting_eff_fps = fps / float(spotting_stride)
        full_crop_stride = max(1, int(fps // 4))    # Sample full person crops ~4 Hz for Model 4
        pitch_stride = max(1, int(self.cfg.pitch_stride))

        # Strict per-track memory caps so 30s-180s+ videos never accumulate unbounded crops in RAM
        max_team_crops_per_track = 16
        max_full_crops_per_track = max(16, int(self.cfg.jersey_max_crops_per_track) * 2)
        max_total_jersey_samples = 240
        team_crop_counts: Dict[int, int] = defaultdict(int)
        full_crop_counts: Dict[int, int] = defaultdict(int)
        prev_thumb_hsv: Optional[np.ndarray] = None
        prev_thumb_gray: Optional[np.ndarray] = None

        pbar = tqdm(range(total_frames), desc="Pass 1/2 (CV Detection + Tracking)", mininterval=0.25)
        for f_idx in pbar:
            ok, frame = cap.read()
            if not ok:
                break

            # Lightweight hard scene-cut detection (HSV histogram L1 + luminance MAD on 96x54 thumb)
            thumb = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
            cur_hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
            cur_gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            is_scene_cut = False
            if prev_thumb_hsv is not None and prev_thumb_gray is not None:
                h_cur = cv2.calcHist([cur_hsv], [0, 2], None, [16, 8], [0, 180, 0, 256]).ravel()
                h_prev = cv2.calcHist([prev_thumb_hsv], [0, 2], None, [16, 8], [0, 180, 0, 256]).ravel()
                h_cur = h_cur / max(1e-6, float(h_cur.sum()))
                h_prev = h_prev / max(1e-6, float(h_prev.sum()))
                hist_l1 = float(np.sum(np.abs(h_cur - h_prev))) * 0.5
                mad = float(np.mean(np.abs(cur_gray.astype(np.float32) - prev_thumb_gray.astype(np.float32))))
                if hist_l1 > 0.45 and mad > 28.0:
                    is_scene_cut = True
            prev_thumb_hsv = cur_hsv
            prev_thumb_gray = cur_gray

            if self.cfg.enable_action_spotting and (f_idx % spotting_stride == 0):
                spotting_frames_bgr.append(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR))

            run_pitch = is_scene_cut or (homography.H is None) or (f_idx % pitch_stride == 0)
            det = self.detector.infer_frame(frame, run_pitch=run_pitch)
            boxes, confs, clss, tids = player_tracker.update(
                det["person_boxes"], det["person_confs"], det["person_clss"]
            )
            ball_xy, ball_det_ok = ball_tracker.update(det["ball_boxes"], det["ball_confs"])
            raw_ball_xy.append(ball_xy)

            feet_px = (
                np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, boxes[:, 3]], axis=-1)
                if len(boxes) > 0
                else None
            )
            if run_pitch:
                homography.update(
                    det["pitch_kpts_xy"],
                    det["pitch_kpts_conf"],
                    player_feet_px=feet_px,
                    is_scene_cut=is_scene_cut,
                )

            # Project player foot-points (bottom center of bounding box) onto FIFA pitch (meters)
            entities = []
            if feet_px is not None:
                feet_m = homography.image_to_pitch(feet_px)
            else:
                feet_m = None

            for i in range(len(boxes)):
                crop = None
                full_crop = None
                c_cls = int(clss[i])
                tid_i = int(tids[i])
                if c_cls == self.cfg.player_id and (f_idx % sample_stride == 0):
                    need_track_crop = team_crop_counts[tid_i] < max_team_crops_per_track
                    need_global_sample = len(jersey_samples) < max_total_jersey_samples
                    if need_track_crop or need_global_sample:
                        cropped = SigLIPTeamClassifier.crop_jersey(frame, boxes[i])
                        if cropped is not None:
                            if need_track_crop:
                                crop = cropped
                                team_crop_counts[tid_i] += 1
                            if need_global_sample:
                                jersey_samples.append(cropped)

                if (
                    self.cfg.enable_jersey_ocr
                    and c_cls in (self.cfg.player_id, self.cfg.goalkeeper_id)
                    and (f_idx % full_crop_stride == 0)
                    and full_crop_counts[tid_i] < max_full_crops_per_track
                ):
                    full_crop = self._crop_full_person(frame, boxes[i])
                    if full_crop is not None:
                        full_crop_counts[tid_i] += 1

                entities.append({
                    "track_id": tid_i,
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
                "homography_reset": bool(is_scene_cut or homography.last_was_reset),
            })

        cap.release()

        # Sequential Stage Memory Release: unload heavy YOLO models BEFORE Stage 1b ResNet-152
        if self.detector is not None:
            if hasattr(self.detector, "players_model"):
                del self.detector.players_model
            if hasattr(self.detector, "pitch_model"):
                del self.detector.pitch_model
            del self.detector
            self.detector = None
        self._release_memory()

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
        # STAGE 1b: Model 3 Raw-Video Broadcast Action Spotting
        # ======================================================================
        broadcast_events = []
        broadcast_spotting_ms: Optional[float] = None
        broadcast_spotting_backend: Optional[str] = None
        if self.cfg.enable_action_spotting and spotting_frames_bgr:
            extractor = SoccerNetVideoFeatureExtractor(
                resnet_onnx_path=self.cfg.resnet_onnx_path or None,
                pca_weights_path=self.cfg.pca_weights_path or None,
                target_fps=2.0,
            )
            if extractor.is_available():
                print("[Stage 1b] Extracting ResNet-152 + PCA-512 features & running Model 3 ONNX...")
                t_spot = time.perf_counter()
                spotter: Optional[ActionSpotter] = None
                try:
                    feats_512 = extractor.extract_from_frames(spotting_frames_bgr)
                    # Release 224x224 sampled frames & ResNet backbone immediately after extraction
                    spotting_frames_bgr.clear()
                    extractor.close()
                    del extractor
                    extractor = None
                    self._release_memory()

                    if len(feats_512) > 0:
                        spotter = ActionSpotter()
                        spot_res = spotter.spot_events(
                            feats_512,
                            half=1,
                            fps=spotting_eff_fps,
                            min_conf=None,
                            nms_window_sec=15.0,
                            use_calibrated_thresholds=True,
                        )
                        clip_dur = round(len(frame_records) / max(1.0, fps), 2)
                        broadcast_events = [
                            ev for ev in spot_res.get("events", [])
                            if float(ev.get("time_s", 0.0)) <= clip_dur + 0.25
                        ]
                        broadcast_spotting_backend = str(spot_res.get("backend_used") or "onnx_int8")
                    del feats_512
                    broadcast_spotting_ms = round((time.perf_counter() - t_spot) * 1000.0, 2)
                except Exception as e:
                    print(f"[Stage 1b] Action Spotting skipped due to error: {e}")
                    broadcast_events = []
                    broadcast_spotting_backend = "error_fallback"
                finally:
                    if extractor is not None:
                        extractor.close()
                    if spotter is not None:
                        spotter.close()
                    spotting_frames_bgr.clear()
                    self._release_memory()
            else:
                broadcast_spotting_backend = "extractor_unavailable"
                spotting_frames_bgr.clear()
        elif self.cfg.enable_action_spotting:
            broadcast_spotting_backend = "extractor_unavailable"

        # ======================================================================
        # STAGE 2: Ball Interpolation, Team Classification & Model 4 Jersey OCR
        # ======================================================================
        print("[Stage 2] Clustering Teams & Running Model 4 Jersey Recognition...")
        team_classifier = SigLIPTeamClassifier(enable_siglip=self.cfg.enable_siglip)
        team_classifier.fit(jersey_samples)
        stabilize_team_assignments(
            frame_records,
            team_classifier,
            player_id=self.cfg.player_id,
            goalkeeper_id=self.cfg.goalkeeper_id,
            referee_id=self.cfg.referee_id,
        )
        jersey_samples.clear()
        team_classifier.close()
        del team_classifier
        self._release_memory()

        if self.cfg.enable_jersey_ocr:
            print("[Stage 2] Loading Model 4 Jersey Gate & OCR...")
            jersey_recognizer = JerseyRecognizer(
                onnx_path=self.cfg.jersey_legibility_onnx_path or None,
                pt_path=self.cfg.jersey_legibility_pt_path or None,
                legibility_threshold=self.cfg.jersey_legibility_threshold,
                min_track_ocr_conf=self.cfg.jersey_min_track_ocr_conf,
                parseq_min_crop_conf=self.cfg.jersey_parseq_min_crop_conf,
                max_crops_per_track=self.cfg.jersey_max_crops_per_track,
                torso_crop_box=self.cfg.jersey_torso_crop_box,
                enable_ocr=self.cfg.enable_jersey_ocr,
            )
            jersey_assignments, jersey_diag = jersey_recognizer.recognize_tracks(
                frame_records,
                player_id=self.cfg.player_id,
                goalkeeper_id=self.cfg.goalkeeper_id,
            )
            jersey_recognizer.close()
            del jersey_recognizer
            self._release_memory()
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
        telemetry["broadcast_events"] = broadcast_events

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
            "broadcast_spotting_ms": broadcast_spotting_ms,
            "broadcast_spotting_backend": broadcast_spotting_backend,
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
        if start_frame > 0 and start_frame < raw_total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
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
