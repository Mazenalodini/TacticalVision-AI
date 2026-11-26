from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
DEFAULT_LETTERBOX_FILL: Tuple[int, int, int] = (20, 26, 22)
DEFAULT_TORSO_CROP_BOX: Tuple[float, float, float, float] = (0.20, 0.18, 0.80, 0.56)


def letterbox_square_bgr(
    crop_bgr: np.ndarray,
    size: int = 160,
    fill_rgb: Tuple[int, int, int] = DEFAULT_LETTERBOX_FILL,
) -> np.ndarray:
    """
    Aspect-ratio-preserving square resize (RGB uint8) with zero horizontal flip,
    matching training LetterboxSquare(160) transform in ml_pipeline_reference/trainer_jersey.py.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.full((size, size, 3), fill_rgb, dtype=np.uint8)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = size / float(max(w, h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), fill_rgb, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def preprocess_legibility_batch(crops_bgr: List[np.ndarray], size: int = 160) -> np.ndarray:
    """Converts BGR person crops into normalized (B, 3, size, size) float32 tensor."""
    if not crops_bgr:
        return np.zeros((0, 3, size, size), dtype=np.float32)

    tensors = []
    for crop in crops_bgr:
        sq = letterbox_square_bgr(crop, size=size).astype(np.float32) / 255.0
        norm = (sq - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(np.transpose(norm, (2, 0, 1)))
    return np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)


def crop_torso_pil(
    crop_bgr: np.ndarray,
    torso_box: Tuple[float, float, float, float] = DEFAULT_TORSO_CROP_BOX,
) -> Optional[Image.Image]:
    """Slices torso ROI [x1, y1, x2, y2] fraction strictly for PARSeq OCR."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if w < 10 or h < 12:
        return None
    fx1, fy1, fx2, fy2 = torso_box
    x1 = max(0, min(w - 1, int(round(w * fx1))))
    y1 = max(0, min(h - 1, int(round(h * fy1))))
    x2 = max(x1 + 1, min(w, int(round(w * fx2))))
    y2 = max(y1 + 1, min(h, int(round(h * fy2))))
    roi = crop_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))


def extract_jersey_digits(raw_text: str) -> Optional[int]:
    """Parses valid jersey numbers in [1, 99] from OCR text."""
    digits = re.findall(r"\d+", raw_text or "")
    if not digits:
        return None
    val = int("".join(digits)[:2])
    return val if 1 <= val <= 99 else None


class JerseyRecognizer:
    """
    Two-Stage Production Jersey Recognizer (Model 4):
      Stage 4a: MobileNetV3-Large Legibility Gate (INT8 ONNX primary, PyTorch .pt fallback)
                fed full bounding box via LetterboxSquare(160) — sigmoid included in ONNX graph.
      Stage 4b: Torso Crop ([0.20, 0.18, 0.80, 0.56]) -> Pretrained PARSeq OCR -> Confidence-Weighted Voting.
                Degrades gracefully to legibility-only when PARSeq/strhub is unavailable or offline.
    """

    def __init__(
        self,
        onnx_path: Optional[str | Path] = None,
        pt_path: Optional[str | Path] = None,
        legibility_threshold: float = 0.45,
        min_track_ocr_conf: float = 0.18,
        parseq_min_crop_conf: float = 0.35,
        max_crops_per_track: int = 12,
        torso_crop_box: Tuple[float, float, float, float] = DEFAULT_TORSO_CROP_BOX,
        enable_ocr: bool = True,
    ):
        self.legibility_threshold = float(legibility_threshold)
        self.min_track_ocr_conf = float(min_track_ocr_conf)
        self.parseq_min_crop_conf = float(parseq_min_crop_conf)
        self.max_crops_per_track = max(1, int(max_crops_per_track))
        self.torso_crop_box = tuple(float(v) for v in torso_crop_box)  # type: ignore
        self.enable_ocr = bool(enable_ocr)

        root = Path(__file__).resolve().parent.parent
        models_dir = root / "models"
        if onnx_path is not None:
            self.onnx_path = Path(onnx_path)
        else:
            cand = models_dir / "jersey_legibility_v1_int8.onnx"
            if not cand.exists() and (models_dir / "jersey_legibility_v1.onnx").exists():
                cand = models_dir / "jersey_legibility_v1.onnx"
            self.onnx_path = cand
        self.pt_path = Path(pt_path) if pt_path else (models_dir / "jersey_legibility_v1.pt")

        self.gate_backend: str = "unavailable"
        self._ort_session: Any = None
        self._ort_input_name: str = "image"
        self._pt_model: Any = None

        self.ocr_available: bool = False
        self.ocr_status: str = "disabled"
        self._parseq: Any = None
        self._rapidocr: Any = None
        self._ocr_tf: Any = None

        self._init_gate()
        if self.enable_ocr:
            self._init_parseq()

    def _init_gate(self) -> None:
        # Primary: ONNX Runtime CPU (INT8 / FP32)
        if self.onnx_path and self.onnx_path.exists():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._ort_session = sess
                self._ort_input_name = sess.get_inputs()[0].name
                self.gate_backend = "onnx_int8" if "int8" in self.onnx_path.name else "onnx_fp32"
                return
            except Exception as e:
                print(f"[JerseyRecognizer] ONNX Runtime unavailable ({e}); trying PyTorch fallback.")

        # Fallback: PyTorch .pt checkpoint (logits -> requires sigmoid in score_legibility)
        if self.pt_path and self.pt_path.exists():
            try:
                import timm
                import torch
                import torch.nn as nn

                blob = torch.load(str(self.pt_path), map_location="cpu", weights_only=False)
                cfg_d = blob.get("config", {})
                backbone_name = cfg_d.get("backbone", "mobilenetv3_large_100")

                class _LegibilityPT(nn.Module):
                    def __init__(self, bb_name: str):
                        super().__init__()
                        self.backbone = timm.create_model(bb_name, pretrained=False, num_classes=0)
                        with torch.no_grad():
                            fdim = int(self.backbone(torch.zeros(1, 3, 160, 160)).shape[-1])
                        self.head = nn.Sequential(nn.Dropout(0.0), nn.Linear(fdim, 1))

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.head(self.backbone(x)).squeeze(-1)

                model = _LegibilityPT(backbone_name).eval()
                model.load_state_dict(blob["model_state_dict"])
                self._pt_model = model
                self.gate_backend = "pytorch_cpu"
            except Exception as e:
                print(f"[JerseyRecognizer] PyTorch legibility fallback failed ({e}).")
                self.gate_backend = "unavailable"

    def _init_parseq(self) -> None:
        """Attempt offline-safe local PARSeq first; fall back cleanly to offline RapidOCR ONNX CPU."""
        try:
            import torch
            from torchvision import transforms

            hub_dir = Path(torch.hub.get_dir())
            has_local_cache = any(hub_dir.glob("baudm_parseq*")) if hub_dir.exists() else False
            if has_local_cache:
                parseq = torch.hub.load(
                    "baudm/parseq",
                    "parseq",
                    pretrained=True,
                    trust_repo=True,
                ).eval()
                img_sz = parseq.hparams.img_size
                self._ocr_tf = transforms.Compose([
                    transforms.Resize(img_sz, transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ])
                self._parseq = parseq
                self.ocr_available = True
                self.ocr_status = "parseq_active"
                return
        except Exception:
            pass

        # Offline CPU Fallback: RapidOCR ONNX Runtime (fast, deterministic, zero internet needed)
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapidocr = RapidOCR()
            self.ocr_available = True
            self.ocr_status = "rapidocr_onnx_active"
        except Exception as e:
            self.ocr_available = False
            self.ocr_status = f"unavailable:{type(e).__name__}"

    @staticmethod
    def _enhance_torso_bgr(
        crop_bgr: np.ndarray,
        torso_box: Tuple[float, float, float, float] = DEFAULT_TORSO_CROP_BOX,
        extra_scale: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Crop torso region, reject ultra-tiny (<14px) crops, and upscale/contrast-enhance for digit OCR."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if w < 14 or h < 14:
            return None
        fx1, fy1, fx2, fy2 = torso_box
        x1 = max(0, min(w - 1, int(round(w * fx1))))
        y1 = max(0, min(h - 1, int(round(h * fy1))))
        x2 = max(x1 + 1, min(w, int(round(w * fx2))))
        y2 = max(y1 + 1, min(h, int(round(h * fy2))))
        roi = crop_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        rh, rw = roi.shape[:2]
        if rw < 8 or rh < 8:
            return None
        min_side = min(rh, rw)
        if min_side < 64 or extra_scale > 1.0:
            base_scale = max(2.0, min(4.0, 72.0 / float(max(min_side, 1)))) if min_side < 64 else 1.0
            scale = base_scale * max(1.0, float(extra_scale))
            roi = cv2.resize(
                roi,
                (max(1, int(round(rw * scale))), max(1, int(round(rh * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )

        # Gentle LAB CLAHE luminance contrast enhancement helps worn/shadowed jersey prints
        try:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l_ch = clahe.apply(l_ch)
            roi = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        except Exception:
            pass
        return roi

    def _is_accepted_digit_read(self, num: Optional[int], conf: float) -> bool:
        """Require stronger OCR confidence for single digits (1..9) to reject tiny noise artifacts."""
        if num is None or not (1 <= int(num) <= 99):
            return False
        req = max(0.55, self.parseq_min_crop_conf) if int(num) < 10 else self.parseq_min_crop_conf
        return float(conf) >= req

    def score_legibility(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Returns float32 array of probabilities strictly in [0.0, 1.0]."""
        if not crops_bgr:
            return np.zeros((0,), dtype=np.float32)
        if self.gate_backend == "unavailable":
            return np.zeros((len(crops_bgr),), dtype=np.float32)

        xb = preprocess_legibility_batch(crops_bgr, size=160)
        if self._ort_session is not None:
            # ONNX graph already applies torch.sigmoid internally — do NOT sigmoid again!
            probs = self._ort_session.run(None, {self._ort_input_name: xb})[0]
            return np.clip(np.asarray(probs, dtype=np.float32).reshape(-1), 0.0, 1.0)

        import torch

        with torch.no_grad():
            logits = self._pt_model(torch.from_numpy(xb))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.clip(np.asarray(probs, dtype=np.float32).reshape(-1), 0.0, 1.0)

    def recognize_tracklet(self, crops_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """Run Stage 4a legibility gate + optional Stage 4b PARSeq weighted voting for one player tracklet."""
        if not crops_bgr:
            return {
                "jersey_number": None,
                "jersey_confidence": None,
                "jersey_verified": False,
                "legibility_score": None,
                "legible_crops_count": 0,
            }

        if len(crops_bgr) > self.max_crops_per_track:
            idxs = np.linspace(0, len(crops_bgr) - 1, self.max_crops_per_track, dtype=int)
            sampled = [crops_bgr[i] for i in idxs]
        else:
            sampled = crops_bgr

        scores = self.score_legibility(sampled)
        max_leg = round(float(np.max(scores)), 3) if len(scores) > 0 else 0.0
        keep_idx = [i for i, s in enumerate(scores) if float(s) >= self.legibility_threshold]

        if not keep_idx or not self.ocr_available or (self._parseq is None and self._rapidocr is None):
            return {
                "jersey_number": None,
                "jersey_confidence": None,
                "jersey_verified": False,
                "legibility_score": max_leg,
                "legible_crops_count": len(keep_idx),
            }

        keep_idx.sort(key=lambda i: float(scores[i]), reverse=True)
        chosen = keep_idx[:5]

        votes: Dict[int, float] = defaultdict(float)
        valid_count = 0

        if self._parseq is not None and self._ocr_tf is not None:
            import torch

            ocr_tensors = []
            valid_chosen_scores: List[float] = []
            for idx in chosen:
                torso_pil = crop_torso_pil(sampled[idx], self.torso_crop_box)
                if torso_pil is not None:
                    ocr_tensors.append(self._ocr_tf(torso_pil))
                    valid_chosen_scores.append(float(scores[idx]))

            if not ocr_tensors:
                return {
                    "jersey_number": None,
                    "jersey_confidence": None,
                    "jersey_verified": False,
                    "legibility_score": max_leg,
                    "legible_crops_count": len(keep_idx),
                }

            valid_count = len(valid_chosen_scores)
            with torch.no_grad():
                ob = torch.stack(ocr_tensors)
                logits = self._parseq(ob)
                probs = logits.softmax(-1)
                preds, p_confs = self._parseq.tokenizer.decode(probs)

            for i_loc, (txt, conf_t) in enumerate(zip(preds, p_confs)):
                num = extract_jersey_digits(txt)
                if num is None:
                    continue
                c_val = float(conf_t[:-1].prod().item()) if len(conf_t) > 1 else float(conf_t.mean().item())
                if self._is_accepted_digit_read(num, c_val):
                    votes[num] += c_val * valid_chosen_scores[i_loc]
        else:
            # Stage 4b Offline Fallback: Multi-scale RapidOCR ONNX on contrast-enhanced torso crops
            for idx in chosen:
                torso_bgr = self._enhance_torso_bgr(sampled[idx], self.torso_crop_box, extra_scale=1.0)
                if torso_bgr is None:
                    continue
                valid_count += 1
                res_list, _ = self._rapidocr(torso_bgr)
                if not res_list:
                    torso_hi = self._enhance_torso_bgr(sampled[idx], self.torso_crop_box, extra_scale=1.5)
                    res_list, _ = self._rapidocr(
                        torso_hi if torso_hi is not None else torso_bgr,
                        use_det=False,
                        use_cls=False,
                        use_rec=True,
                    )
                if not res_list:
                    continue
                for item in res_list:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    txt = str(item[-2]).strip()
                    if any(ch.isalpha() for ch in txt) and not txt.startswith("#"):
                        continue
                    try:
                        c_val = float(item[-1])
                    except (TypeError, ValueError):
                        continue
                    num = extract_jersey_digits(txt)
                    if self._is_accepted_digit_read(num, c_val):
                        votes[num] += c_val * float(scores[idx])

        if not votes or valid_count == 0:
            return {
                "jersey_number": None,
                "jersey_confidence": None,
                "jersey_verified": False,
                "legibility_score": max_leg,
                "legible_crops_count": len(keep_idx),
            }

        best_num, best_weight = max(votes.items(), key=lambda kv: kv[1])
        norm_conf = round(min(1.0, best_weight / max(1, valid_count)), 3)
        verified = norm_conf >= self.min_track_ocr_conf

        return {
            "jersey_number": int(best_num) if verified else None,
            "jersey_confidence": norm_conf if verified else None,
            "jersey_verified": bool(verified),
            "legibility_score": max_leg,
            "legible_crops_count": len(keep_idx),
        }

    def recognize_tracks(
        self,
        frame_records: List[dict],
        player_id: int = 2,
        goalkeeper_id: int = 1,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        """
        Aggregates full person crops per tracked entity across Pass 1 frame records,
        evaluates Model 4 per tracklet, annotates entities cleanly, and returns:
          (per_track_summary_dict, engine_diagnostics_summary)
        """
        crops_by_tid: Dict[int, List[np.ndarray]] = defaultdict(list)
        for rec in frame_records:
            for ent in rec.get("entities", []):
                full_crop = ent.pop("full_crop", None)
                if int(ent.get("cls", -1)) in (player_id, goalkeeper_id) and full_crop is not None:
                    crops_by_tid[int(ent["track_id"])].append(full_crop)

        assignments: Dict[int, Dict[str, Any]] = {}
        total_legible_crops = 0
        verified_count = 0

        for tid, clist in crops_by_tid.items():
            res = self.recognize_tracklet(clist)
            assignments[tid] = res
            total_legible_crops += int(res.get("legible_crops_count", 0))
            if res.get("jersey_verified"):
                verified_count += 1

        # Stamp resolved identity onto frame entities for BroadcastRenderer
        for rec in frame_records:
            for ent in rec.get("entities", []):
                tid = int(ent["track_id"])
                info = assignments.get(tid)
                if info:
                    ent["jersey_number"] = info.get("jersey_number")
                    ent["jersey_verified"] = bool(info.get("jersey_verified", False))

        diag = {
            "gate_enabled": self.gate_backend != "unavailable",
            "gate_backend": self.gate_backend,
            "ocr_available": self.ocr_available,
            "ocr_status": self.ocr_status,
            "tracks_evaluated": len(assignments),
            "legible_crops": total_legible_crops,
            "verified_players": verified_count,
        }
        return assignments, diag
