from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.jersey_recognizer import (
    DEFAULT_TORSO_CROP_BOX,
    JerseyRecognizer,
    crop_torso_pil,
    extract_jersey_digits,
    letterbox_square_bgr,
    preprocess_legibility_batch,
)

ROOT = Path(__file__).resolve().parent.parent


def test_letterbox_square_shape_and_orientation_preserved():
    # Vertical crop with asymmetric left (red) vs right (blue) halves to prove zero horizontal flip
    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    crop[:, :25] = (0, 0, 255)   # BGR Red -> RGB[0]=255 on LEFT
    crop[:, 25:] = (255, 0, 0)  # BGR Blue -> RGB[2]=255 on RIGHT

    out = letterbox_square_bgr(crop, size=160)
    assert out.shape == (160, 160, 3)
    assert out.dtype == np.uint8

    # Scaled width = round(50 * (160/100)) = 80, centered in 160 => x in [40, 120)
    left_pixel = out[80, 50]
    right_pixel = out[80, 110]
    assert int(left_pixel[0]) > 200 and int(left_pixel[2]) < 50
    assert int(right_pixel[2]) > 200 and int(right_pixel[0]) < 50

    batch = preprocess_legibility_batch([crop], size=160)
    assert batch.shape == (1, 3, 160, 160)
    assert batch.dtype == np.float32


def test_torso_crop_coordinates_and_digit_parser():
    assert DEFAULT_TORSO_CROP_BOX == (0.20, 0.18, 0.80, 0.56)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    pil_crop = crop_torso_pil(img)
    assert pil_crop is not None
    # x: [20, 80] -> w=60, y: [18, 56] -> h=38
    assert pil_crop.size == (60, 38)

    assert extract_jersey_digits("10") == 10
    assert extract_jersey_digits("#7A") == 7
    assert extract_jersey_digits("0") is None
    assert extract_jersey_digits("abc") is None


def test_legibility_gate_onnx_vs_pytorch_parity_and_bounds():
    fp32_onnx = ROOT / "models" / "jersey_legibility_v1.onnx"
    int8_onnx = ROOT / "models" / "jersey_legibility_v1_int8.onnx"
    pt_file = ROOT / "models" / "jersey_legibility_v1.pt"
    assert pt_file.exists(), "jersey_legibility_v1.pt missing"

    rng = np.random.default_rng(7)
    crops = [rng.integers(0, 255, size=(120, 60, 3), dtype=np.uint8) for _ in range(4)]

    rec_pt = JerseyRecognizer(
        onnx_path=ROOT / "models" / "__nonexistent__.onnx",
        pt_path=pt_file,
        enable_ocr=False,
    )
    scores_pt = rec_pt.score_legibility(crops)
    assert rec_pt.gate_backend == "pytorch_cpu"
    assert scores_pt.shape == (4,)
    assert np.all((scores_pt >= 0.0) & (scores_pt <= 1.0))

    if fp32_onnx.exists():
        rec_fp32 = JerseyRecognizer(onnx_path=fp32_onnx, pt_path=pt_file, enable_ocr=False)
        scores_fp32 = rec_fp32.score_legibility(crops)
        assert rec_fp32.gate_backend == "onnx_fp32"
        assert np.max(np.abs(scores_fp32 - scores_pt)) < 1e-4

    if int8_onnx.exists():
        rec_int8 = JerseyRecognizer(onnx_path=int8_onnx, pt_path=pt_file, enable_ocr=False)
        scores_int8 = rec_int8.score_legibility(crops)
        assert rec_int8.gate_backend == "onnx_int8"
        assert np.all((scores_int8 >= 0.0) & (scores_int8 <= 1.0))
        assert np.mean(np.abs(scores_int8 - scores_pt)) < 0.08


def test_weighted_tracklet_voting_and_offline_fallback():
    class _FakeDecodedTokenizer:
        def __init__(self, texts, confs):
            self._texts = texts
            self._confs = confs

        def decode(self, probs):
            return self._texts, self._confs

    class _FakeParseq:
        def __init__(self, texts, confs):
            self.tokenizer = _FakeDecodedTokenizer(texts, confs)

        def __call__(self, xb):
            import torch
            return torch.ones((xb.shape[0], 3, 10), dtype=torch.float32)

    import torch

    rec = JerseyRecognizer(enable_ocr=False, legibility_threshold=0.0)
    crops = [np.zeros((100, 50, 3), dtype=np.uint8) for _ in range(3)]

    # Verify offline / OCR-disabled fallback returns legible score cleanly without crash
    fallback = rec.recognize_tracklet(crops)
    assert fallback["jersey_verified"] is False
    assert fallback["jersey_number"] is None
    assert fallback["legibility_score"] is not None

    # Inject mocked PARSeq and fixed high legibility score to isolate voting math
    rec.score_legibility = lambda cr: np.full((len(cr),), 0.9, dtype=np.float32)  # type: ignore[method-assign]
    rec.ocr_available = True
    rec._ocr_tf = lambda pil_img: torch.zeros((3, 32, 128), dtype=torch.float32)
    rec._parseq = _FakeParseq(
        texts=["10", "10", "7"],
        confs=[
            torch.tensor([0.9, 0.9, 0.99]),
            torch.tensor([0.85, 0.85, 0.99]),
            torch.tensor([0.5, 0.99]),
        ],
    )
    voted = rec.recognize_tracklet(crops)
    assert voted["jersey_verified"] is True
    assert voted["jersey_number"] == 10
    assert voted["jersey_confidence"] is not None and voted["jersey_confidence"] > 0.0

    # Ultra-tiny (<14px) crop should be rejected safely by _enhance_torso_bgr
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    assert rec._enhance_torso_bgr(tiny) is None

    # Single-digit read requires stronger confidence (>=0.55) to prevent noisy hallucinations
    assert rec._is_accepted_digit_read(8, 0.40) is False
    assert rec._is_accepted_digit_read(8, 0.65) is True
    assert rec._is_accepted_digit_read(22, 0.40) is True
