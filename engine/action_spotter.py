from __future__ import annotations

import gc
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ACTION_SPOTTING_CLASSES: List[str] = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card",
]

EVENT_CATEGORY_MAP: Dict[str, str] = {
    "Goal": "scoring",
    "Penalty": "scoring",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Corner": "set_piece",
    "Direct free-kick": "set_piece",
    "Indirect free-kick": "set_piece",
    "Throw-in": "set_piece",
    "Kick-off": "set_piece",
    "Foul": "foul",
    "Yellow card": "card",
    "Red card": "card",
    "Yellow->red card": "card",
    "Offside": "tactical",
    "Substitution": "tactical",
    "Clearance": "tactical",
    "Ball out of play": "flow",
}

# Per-class calibrated precision thresholds (prevents celebration/close-up false positives)
PER_CLASS_THRESHOLDS: Dict[str, float] = {
    # High-risk rare events prone to close-up / celebration hallucinations
    "Red card": 0.85,
    "Yellow->red card": 0.85,
    "Penalty": 0.80,
    "Substitution": 0.78,
    "Yellow card": 0.75,
    "Offside": 0.72,
    # Set-pieces & restarts
    "Ball out of play": 0.70,
    "Kick-off": 0.70,
    "Direct free-kick": 0.68,
    "Indirect free-kick": 0.68,
    "Foul": 0.68,
    "Corner": 0.65,
    "Throw-in": 0.65,
    "Clearance": 0.65,
    # Core attacking events
    "Shots off target": 0.58,
    "Goal": 0.55,
    "Shots on target": 0.55,
}

# Events suppressed when Goal or Shots on target fires within +-5.0s (celebration / replay cut window)
ATTACK_SUPPRESSED_CLASSES = {
    "Substitution",
    "Yellow card",
    "Red card",
    "Yellow->red card",
    "Corner",
    "Direct free-kick",
    "Indirect free-kick",
    "Throw-in",
    "Kick-off",
    "Ball out of play",
}

# Mutually conflicting restart/discipline/sub classes within +-5.0s (keep highest-confidence only)
MUTUALLY_CONFLICTING_CLASSES = {
    "Substitution",
    "Yellow card",
    "Red card",
    "Yellow->red card",
    "Corner",
    "Direct free-kick",
    "Indirect free-kick",
    "Throw-in",
    "Kick-off",
}


class ActionSpotter:
    """
    Model 3: Standalone CPU Temporal Action Spotting Inference Engine.

    Executes on 2FPS ResNet-PCA512 half-match feature arrays `(T, 512)` using:
      1. Primary backend: `action_spotting_v1_int8.onnx` (or FP32 `.onnx`) via ONNX Runtime CPU.
      2. Automatic PyTorch fallback: `TacticalActionSpotter` + `action_spotting_v1.pt` when ORT unavailable.

    Strictly preserves ONNX sigmoid parity (never double-applies sigmoid to ONNX outputs).
    """

    def __init__(
        self,
        onnx_path: Optional[str | Path] = None,
        pt_path: Optional[str | Path] = None,
        registry_path: Optional[str | Path] = None,
    ):
        root = Path(__file__).resolve().parent.parent
        env_models = os.environ.get("MODELS_DIR", "").strip()
        models_dir = Path(env_models) if env_models else (root / "models")
        reg_file = Path(registry_path) if registry_path else (root / "artifacts" / "models_meta" / "models_registry.json")

        self.fps: float = 2.0
        self.window_frames: int = 80
        self.stride_frames: int = 40
        self.default_nms_sec: float = 15.0
        self.default_min_conf: float = 0.55
        self.classes: List[str] = list(ACTION_SPOTTING_CLASSES)

        preferred_onnx = "action_spotting_v1_int8.onnx"
        if reg_file.exists():
            try:
                reg = json.loads(reg_file.read_text(encoding="utf-8"))
                m3_meta = reg.get("models", {}).get("action_spotting", {})
                if isinstance(m3_meta.get("classes"), list) and len(m3_meta["classes"]) == 17:
                    self.classes = m3_meta["classes"]
                tuning = reg.get("production_tuning", {}).get("model_3_action_spotting", {})
                preferred_onnx = str(tuning.get("preferred_file", preferred_onnx))
                self.fps = float(tuning.get("fps", self.fps))
                self.window_frames = int(tuning.get("window_frames", self.window_frames))
                self.stride_frames = int(tuning.get("stride_frames", self.stride_frames))
                self.default_nms_sec = max(15.0, float(tuning.get("nms_window_sec", self.default_nms_sec)))
            except Exception:
                pass

        if onnx_path:
            self.onnx_path = Path(onnx_path)
        else:
            cand = models_dir / preferred_onnx
            if not cand.exists():
                cand = models_dir / "action_spotting_v1.onnx"
            self.onnx_path = cand

        self.pt_path = Path(pt_path) if pt_path else (models_dir / "action_spotting_v1.pt")

        self._ort_session: Any = None
        self._ort_input_name: str = "features"
        self._pt_model: Any = None
        self.backend_used: Optional[str] = None

    def close(self) -> None:
        self._ort_session = None
        self._pt_model = None
        gc.collect()

    def _ensure_loaded(self) -> None:
        if self._ort_session is not None or self._pt_model is not None:
            return

        if self.onnx_path.exists():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.enable_cpu_mem_arena = False
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._ort_session = sess
                self._ort_input_name = sess.get_inputs()[0].name
                self.backend_used = "onnx_int8" if "int8" in self.onnx_path.name else "onnx_fp32"
                return
            except Exception as e:
                print(f"[ActionSpotter] ONNX Runtime unavailable ({e}); trying PyTorch fallback.")

        if self.pt_path.exists():
            self._pt_model = self._load_pytorch_fallback(self.pt_path)
            self.backend_used = "pytorch_cpu"
            return

        raise FileNotFoundError(
            f"No Model 3 weights found at '{self.onnx_path}' or '{self.pt_path}'."
        )

    @staticmethod
    def _load_pytorch_fallback(pt_path: Path) -> Any:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _TemporalSEBlock(nn.Module):
            def __init__(self, channels: int, reduction: int = 8):
                super().__init__()
                mid = max(16, channels // reduction)
                self.fc1 = nn.Linear(channels, mid)
                self.fc2 = nn.Linear(mid, channels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w = x.mean(dim=-1)
                w = torch.sigmoid(self.fc2(F.gelu(self.fc1(w)))).unsqueeze(-1)
                return x * w

        class _DilatedResTCNBlock(nn.Module):
            def __init__(self, channels: int, dilation: int, dropout: float = 0.0):
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
                self.gn1 = nn.GroupNorm(8, channels)
                self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
                self.gn2 = nn.GroupNorm(8, channels)
                self.se = _TemporalSEBlock(channels)
                self.drop = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.drop(F.gelu(self.gn1(self.conv1(x))))
                h = self.drop(F.gelu(self.gn2(self.conv2(h))))
                h = self.se(h)
                return x + h

        class _TacticalActionSpotter(nn.Module):
            def __init__(self, feature_dim=512, hidden_dim=256, num_classes=17, dilations=(1, 2, 4, 8), gru_layers=2, attn_heads=4):
                super().__init__()
                self.in_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.0))
                self.tcn = nn.Sequential(*[_DilatedResTCNBlock(hidden_dim, d, dropout=0.0) for d in dilations])
                self.bigru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=gru_layers, batch_first=True, bidirectional=True)
                self.gru_norm = nn.LayerNorm(hidden_dim)
                self.attn_norm1 = nn.LayerNorm(hidden_dim)
                self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, dropout=0.0, batch_first=True)
                self.attn_norm2 = nn.LayerNorm(hidden_dim)
                self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(0.0))
                self.frame_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden_dim // 2, num_classes))
                self.clip_attn = nn.Linear(hidden_dim, 1)
                self.clip_head = nn.Linear(hidden_dim, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.in_proj(x)
                h_tcn = self.tcn(h.transpose(1, 2)).transpose(1, 2)
                h_gru, _ = self.bigru(h_tcn)
                h = self.gru_norm(h_tcn + h_gru)
                qkv = self.attn_norm1(h)
                attn_out, _ = self.mha(qkv, qkv, qkv, need_weights=False)
                h = h + attn_out
                h = h + self.ffn(self.attn_norm2(h))
                return torch.sigmoid(self.frame_head(h))

        blob = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        cfg = blob.get("config", {})
        model = _TacticalActionSpotter(
            feature_dim=int(cfg.get("feature_dim", 512)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_classes=int(cfg.get("num_classes", 17)),
            dilations=tuple(cfg.get("tcn_dilations", (1, 2, 4, 8))),
            gru_layers=int(cfg.get("gru_layers", 2)),
            attn_heads=int(cfg.get("attn_heads", 4)),
        ).eval()
        model.load_state_dict(blob["model_state_dict"])
        return model

    @staticmethod
    def validate_features(features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix (T, 512), got {arr.ndim}D shape {arr.shape}.")
        if arr.shape[0] < 1 or arr.shape[1] != 512:
            raise ValueError(f"Expected feature shape (T >= 1, 512), got {arr.shape}.")
        arr32 = np.ascontiguousarray(arr, dtype=np.float32)
        if not np.all(np.isfinite(arr32)):
            raise ValueError("Feature matrix contains NaN or Inf values.")
        return arr32

    def predict_probabilities(
        self,
        features: np.ndarray,
        window_frames: Optional[int] = None,
        stride_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Runs Hanning-blended overlapping sliding-window inference over `(T, 512)` features.
        Returns `(T, 17)` float32 probabilities strictly bounded in `[0.0, 1.0]`.
        """
        arr = self.validate_features(features)
        self._ensure_loaded()

        T = len(arr)
        W = int(window_frames or self.window_frames)
        S = int(stride_frames or self.stride_frames)

        probs_sum = np.zeros((T, len(self.classes)), dtype=np.float32)
        weight_sum = np.zeros((T, 1), dtype=np.float32)

        hann = np.clip(np.hanning(W).astype(np.float32), 0.15, 1.0)[:, None]

        starts = list(range(0, max(1, T - W + 1), S))
        if T > W and starts[-1] + W < T:
            starts.append(T - W)

        windows = []
        for st in starts:
            if T < W:
                # Edge-pad short clips (3s-30s) rather than zero-padding to avoid empty-frame bias
                padded = np.pad(arr, ((0, W - T), (0, 0)), mode="edge")
                windows.append(np.ascontiguousarray(padded[:W], dtype=np.float32))
            else:
                buf = np.zeros((W, 512), dtype=np.float32)
                end = min(T, st + W)
                buf[:end - st] = arr[st:end]
                windows.append(buf)

        batch_arr = np.ascontiguousarray(np.stack(windows, axis=0), dtype=np.float32)

        preds: List[np.ndarray] = []
        chunk = 64
        if self._ort_session is not None:
            for i in range(0, len(batch_arr), chunk):
                # ActionSpottingONNXWrapper ALREADY applies torch.sigmoid internally!
                pb = self._ort_session.run(None, {self._ort_input_name: batch_arr[i:i + chunk]})[0]
                preds.append(np.asarray(pb, dtype=np.float32))
        else:
            import torch

            with torch.no_grad():
                for i in range(0, len(batch_arr), chunk):
                    pb = self._pt_model(torch.from_numpy(batch_arr[i:i + chunk])).cpu().numpy()
                    preds.append(np.asarray(pb, dtype=np.float32))

        all_pb = np.concatenate(preds, axis=0)
        for st, pb in zip(starts, all_pb):
            end = min(T, st + W)
            L = end - st
            probs_sum[st:end] += pb[:L] * hann[:L]
            weight_sum[st:end] += hann[:L]

        return np.clip(probs_sum / np.maximum(weight_sum, 1e-6), 0.0, 1.0)

    @staticmethod
    def smooth_probabilities(probs: np.ndarray) -> np.ndarray:
        """Apply 3-step zero-phase temporal smoothing [0.25, 0.50, 0.25] per class."""
        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return arr.copy()
        padded = np.pad(arr, ((1, 1), (0, 0)), mode="edge")
        smoothed = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    def postprocess_probabilities(
        self,
        probs: np.ndarray,
        half: int = 1,
        fps: Optional[float] = None,
        min_conf: Optional[float] = None,
        nms_window_sec: Optional[float] = None,
        conflict_window_sec: float = 5.0,
        use_calibrated_thresholds: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Post-process `(T, 17)` probabilities into high-precision broadcast events:
          1. 3-step temporal smoothing [0.25, 0.5, 0.25]
          2. Per-class calibrated thresholds (eliminates celebration/close-up card/sub hallucinations)
          3. Strict per-class 1D Temporal NMS (default 15.0s)
          4. Cross-class conflict suppression within +-5.0s
          5. Realistic density cap (~1 event per 12s, min 3, never dropping a valid Goal)
        """
        eff_fps = float(fps or self.fps)
        eff_nms_sec = float(self.default_nms_sec if nms_window_sec is None else nms_window_sec)
        floor_conf = float(min_conf) if min_conf is not None else 0.0

        smoothed = self.smooth_probabilities(probs)
        T, C = smoothed.shape
        rad = max(1, int(round((eff_nms_sec / 2.0) * eff_fps)))
        raw_dets: List[Tuple[int, int, float]] = []

        for c in range(C):
            label = self.classes[c] if c < len(self.classes) else f"Class-{c}"
            cls_thresh = (
                max(floor_conf, PER_CLASS_THRESHOLDS.get(label, 0.65))
                if use_calibrated_thresholds
                else max(floor_conf, self.default_min_conf)
            )
            scores = smoothed[:, c].copy()
            while True:
                idx = int(np.argmax(scores))
                val = float(scores[idx])
                if val < cls_thresh:
                    break
                raw_dets.append((c, idx, val))
                lo = max(0, idx - rad)
                hi = min(T, idx + rad + 1)
                scores[lo:hi] = -1.0

        # Build candidate dicts
        candidates: List[Dict[str, Any]] = []
        for cid, f_idx, conf in raw_dets:
            label = self.classes[cid] if cid < len(self.classes) else f"Class-{cid}"
            category = EVENT_CATEGORY_MAP.get(label, "tactical")
            time_s = round(f_idx / eff_fps, 2)
            mins = int(time_s // 60)
            secs = int(time_s % 60)
            candidates.append({
                "class_id": cid,
                "label": label,
                "category": category,
                "half": int(half),
                "frame_idx": f_idx,
                "time_s": time_s,
                "minute_str": f"{mins:02d}:{secs:02d}",
                "confidence": round(conf, 4),
            })

        # Cross-class conflict suppression within +-conflict_window_sec
        # Priority: Goal first, then Shots on target, then highest confidence
        def _rank_key(ev: Dict[str, Any]) -> Tuple[int, float]:
            lbl = ev["label"]
            prio = 2 if lbl == "Goal" else (1 if lbl == "Shots on target" else 0)
            return (prio, float(ev["confidence"]))

        ranked = sorted(candidates, key=_rank_key, reverse=True)
        kept: List[Dict[str, Any]] = []
        for cand in ranked:
            suppressed = False
            c_lbl = cand["label"]
            c_t = float(cand["time_s"])
            for existing in kept:
                if abs(c_t - float(existing["time_s"])) > conflict_window_sec:
                    continue
                e_lbl = existing["label"]
                # Goal or Shots on target suppresses lower-priority cards/subs/restarts in same window
                if e_lbl in ("Goal", "Shots on target") and c_lbl in ATTACK_SUPPRESSED_CLASSES:
                    suppressed = True
                    break
                # Mutually conflicting cards/subs/restarts cannot co-fire within +-5s
                if e_lbl in MUTUALLY_CONFLICTING_CLASSES and c_lbl in MUTUALLY_CONFLICTING_CLASSES:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(cand)

        # Realistic density cap: ~1 event per 12s (minimum 3), never dropping a valid Goal
        duration_s = T / max(1e-6, eff_fps)
        max_events = max(3, int(round(duration_s / 12.0)))
        if len(kept) > max_events:
            goals = [e for e in kept if e["label"] == "Goal"]
            others = sorted(
                [e for e in kept if e["label"] != "Goal"],
                key=lambda e: float(e["confidence"]),
                reverse=True,
            )
            rem = max(0, max_events - len(goals))
            kept = goals + others[:rem]

        kept.sort(key=lambda e: (e["frame_idx"], -float(e["confidence"])))
        return kept

    def spot_events(
        self,
        features: np.ndarray,
        half: int = 1,
        fps: Optional[float] = None,
        min_conf: Optional[float] = None,
        nms_window_sec: Optional[float] = None,
        use_calibrated_thresholds: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end Model 3 analysis on a `(T, 512)` sequence:
          - Probability extraction + 3-step temporal smoothing
          - Per-class calibrated thresholds + per-class 1D Temporal NMS + cross-class suppression
          - 5-minute timeline density bucketing
          - Measured wall-clock `inference_ms`
        """
        eff_fps = float(fps or self.fps)
        eff_min_conf = float(self.default_min_conf if min_conf is None else min_conf)
        eff_nms_sec = float(self.default_nms_sec if nms_window_sec is None else nms_window_sec)

        t0 = time.perf_counter()
        probs = self.predict_probabilities(features)
        inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        T, _C = probs.shape
        events = self.postprocess_probabilities(
            probs,
            half=half,
            fps=eff_fps,
            min_conf=min_conf,
            nms_window_sec=eff_nms_sec,
            use_calibrated_thresholds=use_calibrated_thresholds,
        )

        class_counts: Counter = Counter()
        category_counts: Counter = Counter()
        for ev in events:
            class_counts[ev["label"]] += 1
            category_counts[ev["category"]] += 1

        duration_s = round(T / eff_fps, 1)
        num_bins = max(1, int(np.ceil(duration_s / 300.0)))
        density_bins = []
        for b in range(num_bins):
            t_start = b * 300.0
            t_end = (b + 1) * 300.0
            in_bin = [e for e in events if t_start <= e["time_s"] < t_end]
            density_bins.append({
                "bin_index": b,
                "label": f"{b * 5}-{(b + 1) * 5}'",
                "start_s": t_start,
                "end_s": t_end,
                "count": len(in_bin),
                "high_impact_count": sum(1 for e in in_bin if e["category"] in ("scoring", "card", "shot")),
            })

        return {
            "half": int(half),
            "num_frames": T,
            "fps": eff_fps,
            "duration_s": duration_s,
            "inference_ms": inference_ms,
            "backend_used": self.backend_used or "unknown",
            "min_conf": eff_min_conf,
            "nms_window_sec": eff_nms_sec,
            "events": events,
            "class_counts": dict(class_counts),
            "category_counts": dict(category_counts),
            "density_bins": density_bins,
        }

    def spot_video(
        self,
        video_path: str | Path,
        max_seconds: Optional[float] = None,
        min_conf: float = 0.25,
        nms_window_sec: float = 5.0,
        resnet_onnx_path: Optional[str | Path] = None,
        pca_weights_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end Model 3 raw-video inference (ResNet-152 ONNX + PCA-512 -> ActionSpotter INT8).
        Degrades gracefully returning empty events if feature extractor weights are unavailable.
        """
        from .video_feature_extractor import SoccerNetVideoFeatureExtractor

        extractor = SoccerNetVideoFeatureExtractor(
            resnet_onnx_path=resnet_onnx_path,
            pca_weights_path=pca_weights_path,
            target_fps=self.fps,
        )
        if not extractor.is_available():
            return {
                "events": [],
                "inference_ms": 0.0,
                "backend_used": "unavailable",
            }

        t0 = time.perf_counter()
        feats, eff_fps = extractor.extract_from_video(video_path, max_seconds=max_seconds)
        if len(feats) == 0:
            return {
                "events": [],
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "backend_used": self.backend_used or "unavailable",
            }

        res = self.spot_events(
            feats,
            half=1,
            fps=eff_fps,
            min_conf=min_conf,
            nms_window_sec=nms_window_sec,
        )
        res["inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return res
