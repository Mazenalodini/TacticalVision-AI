from __future__ import annotations

import importlib.util
import io
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import settings

MAX_UPLOAD_FEATURE_BYTES = 100 * 1024 * 1024  # 100 MB hard cap

_spotter_instance: Any = None


def _artifacts_dir() -> Path:
    return settings.project_root / "artifacts"


def _meta_dir() -> Path:
    return _artifacts_dir() / "models_meta"


def _demo_features_dir() -> Path:
    return _artifacts_dir() / "demo_features"


def _results_dir() -> Path:
    return _artifacts_dir() / "action_spotting_results"


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_spotter():
    """Lazy-load ActionSpotter singleton so FastAPI boot stays lightweight."""
    global _spotter_instance
    if _spotter_instance is None:
        from engine.action_spotter import ActionSpotter

        _spotter_instance = ActionSpotter(
            registry_path=_meta_dir() / "models_registry.json",
        )
    return _spotter_instance


def _check_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_models_suite() -> dict[str, Any]:
    """
    Aggregate verified 4-model metadata, evaluation metrics, CPU/ONNX benchmarks,
    on-disk artifact inventory, and active runtime backend diagnostics.
    """
    registry = _read_json(_meta_dir() / "models_registry.json") or {}
    metrics = _read_json(_meta_dir() / "master_metrics_report.json") or {}
    benchmarks = _read_json(_meta_dir() / "cpu_onnx_benchmarks.json") or {}
    thresholds = _read_json(_meta_dir() / "optimal_thresholds_v1.json") or {}

    disk_inventory: dict[str, dict[str, Any]] = {}
    for model_key, model_meta in (registry.get("models") or {}).items():
        files_meta = model_meta.get("files") or {
            k: model_meta[k]
            for k in ("pt", "onnx", "onnx_cpu_int8")
            if k in model_meta
        }
        entries: dict[str, Any] = {}
        for variant, fname in files_meta.items():
            fpath = settings.models_dir / str(fname)
            exists = fpath.exists()
            size_mb = round(fpath.stat().st_size / (1024 * 1024), 2) if exists else None
            entries[variant] = {
                "filename": fname,
                "exists": exists,
                "size_mb": size_mb,
            }
        disk_inventory[model_key] = entries

    has_ort = _check_pkg("onnxruntime")
    has_torch = _check_pkg("torch")
    has_timm = _check_pkg("timm")
    has_strhub = _check_pkg("strhub")

    demo_npz = _demo_features_dir() / "test_000.npz"

    return {
        "platform": registry.get("platform", "TacticalVision AI"),
        "schema_version": registry.get("schema_version", "2.0"),
        "generated_at": registry.get("generated_at"),
        "models": registry.get("models", {}),
        "production_tuning": registry.get("production_tuning", {}),
        "metrics": metrics,
        "cpu_onnx_benchmarks": benchmarks,
        "optimal_thresholds": thresholds,
        "disk_inventory": disk_inventory,
        "runtime": {
            "onnxruntime_available": has_ort,
            "torch_available": has_torch,
            "timm_available": has_timm,
            "strhub_ocr_available": has_strhub,
            "model_3_preferred_backend": "onnx_int8" if has_ort else ("pytorch_cpu" if has_torch else "unavailable"),
            "model_4_gate_preferred_backend": "onnx_int8" if has_ort else ("pytorch_cpu" if has_torch else "unavailable"),
            "demo_features_available": demo_npz.exists(),
        },
    }


def list_spotting_matches() -> dict[str, Any]:
    """List pre-extracted SoccerNet demo matches, precomputed sample summary, and saved runs."""
    meta = _read_json(_meta_dir() / "demo_match_meta.json")
    sample = _read_json(_meta_dir() / "sample_action_spotting_timeline.json")

    demo_matches: list[dict[str, Any]] = []
    if meta:
        game_id = str(meta.get("game_id", "test_000"))
        npz_path = _demo_features_dir() / f"{game_id}.npz"
        num_frames = meta.get("num_frames") or {}
        gt_events = meta.get("events") or {}
        demo_matches.append({
            "match_id": game_id,
            "title": str(meta.get("game_rel") or game_id).split("/")[-1],
            "league_season_path": meta.get("game_rel"),
            "fps": meta.get("fps", 2.0),
            "feature_dim": meta.get("feature_dim", 512),
            "halves": {
                "half1_frames": int(num_frames.get("1", 0)),
                "half2_frames": int(num_frames.get("2", 0)),
            },
            "gt_event_counts": {
                "half1": len(gt_events.get("1", [])),
                "half2": len(gt_events.get("2", [])),
                "total": len(gt_events.get("1", [])) + len(gt_events.get("2", [])),
            },
            "features_ready": npz_path.exists(),
            "features_size_mb": round(npz_path.stat().st_size / (1024 * 1024), 2) if npz_path.exists() else None,
        })

    saved_runs: list[dict[str, Any]] = []
    res_dir = _results_dir()
    if res_dir.exists():
        for path in sorted(res_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = _read_json(path)
            if not data:
                continue
            saved_runs.append({
                "run_id": data.get("run_id", path.stem),
                "match_id": data.get("match_id"),
                "match_title": data.get("match_title"),
                "half_mode": data.get("half_mode"),
                "backend_used": data.get("backend_used"),
                "total_inference_ms": data.get("total_inference_ms"),
                "events_count": len(data.get("events") or []),
                "created_at": data.get("created_at"),
            })

    sample_summary = None
    if sample:
        sample_summary = {
            "match": sample.get("match"),
            "half": sample.get("half"),
            "cpu_inference_sec": sample.get("cpu_inference_sec"),
            "detected_events_count": len(sample.get("detected_events") or []),
        }

    return {
        "demo_matches": demo_matches,
        "sample_timeline": sample_summary,
        "saved_runs": saved_runs[:25],
    }


def get_spotting_result(run_id: str) -> Optional[dict[str, Any]]:
    """Fetch a previously persisted Model 3 action spotting result by run_id."""
    safe_id = Path(run_id).name
    if safe_id != run_id or not safe_id:
        return None
    return _read_json(_results_dir() / f"{safe_id}.json")


def _compute_gt_comparison(
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    tolerance_sec: float = 10.0,
) -> dict[str, Any]:
    """Greedy one-to-one matching within `±tolerance_sec` on same half & class label."""
    candidates: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_events):
        for g_idx, gt in enumerate(gt_events):
            if int(pred.get("half", 1)) != int(gt.get("half", 1)):
                continue
            if str(pred.get("label")) != str(gt.get("label")):
                continue
            dt = abs(float(pred.get("time_s", 0.0)) - float(gt.get("time_s", 0.0)))
            if dt <= tolerance_sec:
                candidates.append((dt, p_idx, g_idx))

    candidates.sort(key=lambda item: item[0])
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for _, p_idx, g_idx in candidates:
        if p_idx not in used_pred and g_idx not in used_gt:
            used_pred.add(p_idx)
            used_gt.add(g_idx)

    matched = len(used_gt)
    gt_total = len(gt_events)
    pred_total = len(pred_events)
    precision = round((matched / pred_total) * 100.0, 2) if pred_total > 0 else 0.0
    recall = round((matched / gt_total) * 100.0, 2) if gt_total > 0 else 0.0

    return {
        "tolerance_sec": float(tolerance_sec),
        "gt_total": gt_total,
        "matched_gt": matched,
        "predicted_total": pred_total,
        "precision_pct": precision,
        "recall_pct": recall,
    }


def _aggregate_halves_response(
    match_id: str,
    match_title: str,
    half_mode: str,
    half_arrays: list[tuple[int, np.ndarray]],
    min_conf: float,
    nms_window_sec: float,
    gt_by_half: Optional[dict[int, list[dict[str, Any]]]] = None,
    persist: bool = True,
) -> dict[str, Any]:
    spotter = _get_spotter()

    halves_summary: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    class_counts: Counter = Counter()
    category_counts: Counter = Counter()
    all_bins: list[dict[str, Any]] = []
    total_inference_ms = 0.0
    backend_used = "unknown"
    eff_fps = float(spotter.fps)

    bin_offset = 0
    for half_num, feat_arr in half_arrays:
        res = spotter.spot_events(
            feat_arr,
            half=half_num,
            min_conf=min_conf,
            nms_window_sec=nms_window_sec,
        )
        backend_used = res["backend_used"]
        eff_fps = res["fps"]
        total_inference_ms += float(res["inference_ms"])
        halves_summary.append({
            "half": int(res["half"]),
            "num_frames": int(res["num_frames"]),
            "fps": float(res["fps"]),
            "duration_s": float(res["duration_s"]),
            "inference_ms": float(res["inference_ms"]),
            "backend_used": str(res["backend_used"]),
            "events_count": len(res["events"]),
        })
        all_events.extend(res["events"])
        for k, v in (res.get("class_counts") or {}).items():
            class_counts[k] += int(v)
        for k, v in (res.get("category_counts") or {}).items():
            category_counts[k] += int(v)

        for b in res.get("density_bins") or []:
            bin_copy = dict(b)
            bin_copy["bin_index"] = bin_offset + int(b["bin_index"])
            if len(half_arrays) > 1:
                bin_copy["label"] = f"H{half_num} {b['label']}"
            all_bins.append(bin_copy)
        bin_offset += len(res.get("density_bins") or [])

    all_events.sort(key=lambda e: (int(e["half"]), float(e["time_s"]), -float(e["confidence"])))

    gt_flat: Optional[list[dict[str, Any]]] = None
    gt_comparison: Optional[dict[str, Any]] = None
    if gt_by_half:
        gt_flat = []
        for half_num, _ in half_arrays:
            for g in gt_by_half.get(half_num, []):
                g_copy = dict(g)
                g_copy["half"] = half_num
                gt_flat.append(g_copy)
        if gt_flat:
            gt_comparison = _compute_gt_comparison(
                all_events,
                gt_flat,
                tolerance_sec=nms_window_sec,
            )

    run_id = f"as_{match_id}_{half_mode}_{uuid.uuid4().hex[:8]}"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "match_id": match_id,
        "match_title": match_title,
        "half_mode": half_mode,
        "fps": eff_fps,
        "min_conf": float(min_conf),
        "nms_window_sec": float(nms_window_sec),
        "total_inference_ms": round(total_inference_ms, 2),
        "backend_used": backend_used,
        "halves": halves_summary,
        "events": all_events,
        "class_counts": dict(class_counts),
        "category_counts": dict(category_counts),
        "density_bins": all_bins,
        "ground_truth_events": gt_flat,
        "gt_comparison": gt_comparison,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if persist:
        out_dir = _results_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{run_id}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    return payload


def _parse_half_mode(half: str) -> str:
    norm = (half or "half1").strip().lower()
    mapping = {
        "1": "half1",
        "half1": "half1",
        "2": "half2",
        "half2": "half2",
        "both": "both",
        "full": "both",
    }
    if norm not in mapping:
        raise ValueError(f"Invalid half '{half}'. Expected 'half1', 'half2', or 'both'.")
    return mapping[norm]


def analyze_demo_match(
    match_id: str,
    half: str = "half1",
    min_conf: float = 0.35,
    nms_window_sec: float = 10.0,
    persist: bool = True,
) -> dict[str, Any]:
    """Run live Model 3 CPU inference on pre-extracted SoccerNet demo match features."""
    half_mode = _parse_half_mode(half)
    npz_path = _demo_features_dir() / f"{Path(match_id).name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Demo feature file not found for match_id='{match_id}'.")

    meta = _read_json(_meta_dir() / "demo_match_meta.json") or {}
    match_title = str(meta.get("game_rel") or match_id).split("/")[-1]

    gt_by_half: dict[int, list[dict[str, Any]]] = {}
    if meta.get("game_id") == match_id and isinstance(meta.get("events"), dict):
        for k, ev_list in meta["events"].items():
            try:
                gt_by_half[int(k)] = list(ev_list)
            except (TypeError, ValueError):
                pass

    half_arrays: list[tuple[int, np.ndarray]] = []
    with np.load(str(npz_path), allow_pickle=False) as data:
        if half_mode in ("half1", "both"):
            if "half1" not in data:
                raise ValueError("Demo NPZ is missing key 'half1'.")
            half_arrays.append((1, np.array(data["half1"])))
        if half_mode in ("half2", "both"):
            if "half2" not in data:
                raise ValueError("Demo NPZ is missing key 'half2'.")
            half_arrays.append((2, np.array(data["half2"])))

    return _aggregate_halves_response(
        match_id=match_id,
        match_title=match_title,
        half_mode=half_mode,
        half_arrays=half_arrays,
        min_conf=min_conf,
        nms_window_sec=nms_window_sec,
        gt_by_half=gt_by_half or None,
        persist=persist,
    )


def analyze_uploaded_features(
    filename: str,
    raw_bytes: bytes,
    half: str = "half1",
    min_conf: float = 0.35,
    nms_window_sec: float = 10.0,
    persist: bool = True,
) -> dict[str, Any]:
    """Validate and run live Model 3 CPU inference on uploaded `.npy` or `.npz` `(T, 512)` features."""
    if not raw_bytes:
        raise ValueError("Uploaded feature file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_FEATURE_BYTES:
        raise ValueError(
            f"Uploaded file exceeds 100 MB limit ({len(raw_bytes) / (1024 * 1024):.1f} MB)."
        )

    safe_name = Path(filename or "upload.npy").name
    ext = Path(safe_name).suffix.lower()
    if ext not in (".npy", ".npz"):
        raise ValueError(f"Unsupported feature extension '{ext}'. Upload a .npy or .npz file.")

    half_mode = _parse_half_mode(half)
    buf = io.BytesIO(raw_bytes)
    try:
        loaded = np.load(buf, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Failed to load NumPy feature file safely (allow_pickle=False): {exc}") from exc

    half_arrays: list[tuple[int, np.ndarray]] = []
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            keys = list(loaded.files)
            if not keys:
                raise ValueError("Uploaded .npz archive contains no arrays.")
            if half_mode == "both" and "half1" in keys and "half2" in keys:
                half_arrays.append((1, np.array(loaded["half1"])))
                half_arrays.append((2, np.array(loaded["half2"])))
            elif half_mode in keys:
                h_num = 2 if half_mode == "half2" else 1
                half_arrays.append((h_num, np.array(loaded[half_mode])))
            elif len(keys) == 1:
                h_num = 2 if half_mode == "half2" else 1
                half_arrays.append((h_num, np.array(loaded[keys[0]])))
            else:
                raise ValueError(
                    f"Requested '{half_mode}' not found in .npz keys {keys}."
                )
    else:
        h_num = 2 if half_mode == "half2" else 1
        half_arrays.append((h_num, np.asarray(loaded)))

    stem = Path(safe_name).stem or "uploaded"
    return _aggregate_halves_response(
        match_id=f"upload_{stem}",
        match_title=safe_name,
        half_mode=half_mode if len(half_arrays) == 2 else f"half{half_arrays[0][0]}",
        half_arrays=half_arrays,
        min_conf=min_conf,
        nms_window_sec=nms_window_sec,
        gt_by_half=None,
        persist=persist,
    )
