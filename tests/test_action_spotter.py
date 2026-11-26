from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.action_spotter import ACTION_SPOTTING_CLASSES, ActionSpotter

ROOT = Path(__file__).resolve().parent.parent
DEMO_NPZ = ROOT / "artifacts" / "demo_features" / "test_000.npz"


def test_validate_features_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        ActionSpotter.validate_features(np.zeros((512,), dtype=np.float32))
    with pytest.raises(ValueError):
        ActionSpotter.validate_features(np.zeros((10, 256), dtype=np.float32))
    with pytest.raises(ValueError):
        ActionSpotter.validate_features(np.zeros((0, 512), dtype=np.float32))

    with_nan = np.zeros((16, 512), dtype=np.float32)
    with_nan[3, 10] = np.nan
    with pytest.raises(ValueError):
        ActionSpotter.validate_features(with_nan)

    with_inf = np.zeros((16, 512), dtype=np.float32)
    with_inf[5, 0] = np.inf
    with pytest.raises(ValueError):
        ActionSpotter.validate_features(with_inf)


def test_onnx_fp32_vs_pytorch_and_int8_probability_parity():
    fp32_onnx = ROOT / "models" / "action_spotting_v1.onnx"
    int8_onnx = ROOT / "models" / "action_spotting_v1_int8.onnx"
    pt_file = ROOT / "models" / "action_spotting_v1.pt"
    assert pt_file.exists(), "action_spotting_v1.pt missing"

    rng = np.random.default_rng(42)
    sample = rng.normal(0.0, 0.5, size=(120, 512)).astype(np.float32)

    spot_pt = ActionSpotter(onnx_path=ROOT / "models" / "__nonexistent__.onnx", pt_path=pt_file)
    probs_pt = spot_pt.predict_probabilities(sample)
    assert spot_pt.backend_used == "pytorch_cpu"
    assert probs_pt.shape == (120, 17)
    assert float(probs_pt.min()) >= 0.0 and float(probs_pt.max()) <= 1.0

    if fp32_onnx.exists():
        spot_fp32 = ActionSpotter(onnx_path=fp32_onnx, pt_path=pt_file)
        probs_fp32 = spot_fp32.predict_probabilities(sample)
        assert spot_fp32.backend_used == "onnx_fp32"
        max_diff = float(np.max(np.abs(probs_fp32 - probs_pt)))
        assert max_diff < 1e-3, f"FP32 ONNX vs PyTorch diverged (max_diff={max_diff})"

    if int8_onnx.exists():
        spot_int8 = ActionSpotter(onnx_path=int8_onnx, pt_path=pt_file)
        probs_int8 = spot_int8.predict_probabilities(sample)
        assert spot_int8.backend_used == "onnx_int8"
        assert float(probs_int8.min()) >= 0.0 and float(probs_int8.max()) <= 1.0
        mean_diff = float(np.mean(np.abs(probs_int8 - probs_pt)))
        assert mean_diff < 0.05, f"INT8 quantization drift too large (mean_diff={mean_diff})"


def test_spot_events_on_demo_test_000_half1():
    assert DEMO_NPZ.exists(), f"Missing demo NPZ at {DEMO_NPZ}"
    with np.load(str(DEMO_NPZ), allow_pickle=False) as data:
        half1 = np.array(data["half1"])

    assert half1.shape == (5400, 512)

    spotter = ActionSpotter()
    probs = spotter.predict_probabilities(half1)
    assert probs.shape == (5400, 17)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0

    result = spotter.spot_events(half1, half=1, min_conf=0.35, nms_window_sec=10.0)
    print(
        f"\n[Model 3 test_000 half1] backend={result['backend_used']} "
        f"inference_ms={result['inference_ms']}ms events={len(result['events'])}"
    )

    assert result["backend_used"] in ("onnx_int8", "onnx_fp32", "pytorch_cpu")
    assert result["inference_ms"] > 0.0
    assert result["num_frames"] == 5400
    assert result["fps"] == 2.0
    assert len(result["events"]) > 0
    assert len(result["density_bins"]) >= 9

    by_class: dict[str, list[float]] = {}
    for ev in result["events"]:
        assert ev["label"] in ACTION_SPOTTING_CLASSES
        assert 0.35 <= ev["confidence"] <= 1.0
        assert ev["half"] == 1
        assert 0 <= ev["frame_idx"] < 5400
        by_class.setdefault(ev["label"], []).append(ev["time_s"])

    # Verify per-class 1D Temporal NMS enforces >= 5.0s separation (radius = nms_window_sec / 2)
    for label, times in by_class.items():
        times.sort()
        for t1, t2 in zip(times, times[1:]):
            assert (t2 - t1) >= 5.0, f"NMS violation for '{label}': {t1}s vs {t2}s"
