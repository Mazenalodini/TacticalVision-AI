"""Standalone Worker Script for Video Inference.

Spawned by backend.services.job_runner in an isolated subprocess to keep
the main FastAPI process completely free of heavy PyTorch / Ultralytics imports.

Usage:
    python -m backend.services.worker --help
    python -m backend.services.worker --job-id <id> --video <path> --output-dir <path> --title <title> --max-seconds 5.0
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def ensure_ffmpeg_in_path(backend_dir: Path) -> str | None:
    """Ensure `ffmpeg` is discoverable via shutil.which('ffmpeg') for H.264 encoding.

    If system ffmpeg is not in PATH, resolves imageio-ffmpeg's bundled binary
    and exposes it as `ffmpeg.exe` (Windows) / `ffmpeg` inside `backend/bin`,
    prepending that directory to PATH so engine/renderer.py works untouched.
    """
    existing = shutil.which("ffmpeg")
    if existing:
        return existing

    try:
        import imageio_ffmpeg

        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not src.exists():
            return None

        bin_dir = backend_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        dst = bin_dir / exe_name

        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        path_entries = [str(bin_dir), str(src.parent), os.environ.get("PATH", "")]
        os.environ["PATH"] = os.pathsep.join(p for p in path_entries if p)
        return shutil.which("ffmpeg")
    except Exception as e:
        sys.stderr.write(f"[Worker Warning] Could not configure imageio-ffmpeg fallback: {e}\n")
        return None


def update_job_status(
    jobs_dir: Path,
    job_id: str,
    status: str,
    progress_pct: float,
    stage_label: str,
    start_time: float,
    error: str | None = None,
    match_id: str | None = None,
    video_path: str | None = None,
) -> None:
    """Atomically write job status JSON (preserving cancellation if triggered externally)."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    target_path = jobs_dir / f"{job_id}.json"

    # Do not overwrite an already-cancelled job
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as existing_f:
                existing = json.load(existing_f)
            if existing.get("status") == "cancelled":
                return
        except Exception:
            pass

    elapsed = round(time.time() - start_time, 1)
    payload = {
        "job_id": job_id,
        "status": status,
        "progress_pct": round(progress_pct, 1),
        "stage_label": stage_label,
        "elapsed_s": elapsed,
        "error": error,
        "match_id": match_id,
        "pid": os.getpid() if status not in ("completed", "failed", "cancelled") else None,
        "video_path": video_path,
    }

    temp_path = jobs_dir / f"{job_id}.json.tmp"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        temp_path.replace(target_path)
    except Exception as e:
        sys.stderr.write(f"Failed to update job status: {e}\n")


class ProgressParserStream:
    """Tee stream that inspects output from tqdm / pipeline and triggers status callbacks."""

    def __init__(self, original_stream, on_progress_callback):
        self.original_stream = original_stream
        self.on_progress = on_progress_callback
        self.buffer = ""
        self.current_pass = 1

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()
        self.buffer += s

        lines = self.buffer.split("\r")
        if len(lines) > 1:
            for line in lines[:-1]:
                self._parse_line(line)
            self.buffer = lines[-1]

        lines_nl = self.buffer.split("\n")
        if len(lines_nl) > 1:
            for line in lines_nl[:-1]:
                self._parse_line(line)
            self.buffer = lines_nl[-1]

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

    def _parse_line(self, line: str):
        text = line.strip()
        if not text:
            return

        # Stage 2 marker
        if "[Stage 2]" in text:
            self.on_progress("classifying", 55.0, "Fitting SigLIP & Stabilizing Team IDs...")
            return

        # Stage 3 marker
        if "[Stage 3]" in text:
            self.on_progress("computing", 68.0, "Extracting Tactical & Kinematic Telemetry...")
            return

        # Pass 2 marker
        if "Pass 2/2" in text:
            self.current_pass = 2
            # Try parsing percentage from tqdm e.g. " 45%|"
            pct_match = re.search(r"(\d+)%\|", text)
            if pct_match:
                pct = float(pct_match.group(1))
                # Map 0..100% of pass 2 to 78%..90%
                overall = 78.0 + (pct / 100.0) * 12.0
                self.on_progress("encoding", overall, f"Rendering Broadcast Overlays & Radar ({int(pct)}%)...")
            else:
                self.on_progress("encoding", 78.0, "Rendering Broadcast Overlays & Radar...")
            return

        # Encoding marker
        if "[Encoding]" in text:
            self.on_progress("encoding", 92.0, "Transcoding Web H.264 Video...")
            return

        # Pass 1 tqdm detection
        if "Pass 1/2" in text or self.current_pass == 1:
            pct_match = re.search(r"(\d+)%\|", text)
            if pct_match:
                pct = float(pct_match.group(1))
                # Map 0..100% of pass 1 to 5%..50%
                overall = 5.0 + (pct / 100.0) * 45.0
                self.on_progress("detecting", overall, f"Dual-YOLOv11 Detection & Tracking ({int(pct)}%)...")


def main() -> int:
    parser = argparse.ArgumentParser(description="TacticalVision AI Isolated Worker")
    parser.add_argument("--job-id", required=True, help="Unique job identifier")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", required=True, help="Target match directory in outputs/")
    parser.add_argument("--title", default="User Uploaded Match", help="Match display title")
    parser.add_argument("--max-seconds", type=float, default=5.0, help="Maximum clip duration to analyze")
    parser.add_argument("--mode", default="cpu_quick", help="Analysis mode (cpu_quick | remote_gpu)")
    parser.add_argument(
        "--enable-jersey-ocr",
        dest="enable_jersey_ocr",
        action="store_true",
        default=True,
        help="Run Model 4 Jersey Legibility Gate + PARSeq OCR",
    )
    parser.add_argument(
        "--no-jersey-ocr",
        dest="enable_jersey_ocr",
        action="store_false",
        help="Disable Model 4 Jersey Recognition",
    )

    args = parser.parse_args()

    # Determine paths
    backend_dir = Path(__file__).resolve().parent.parent
    project_root = backend_dir.parent
    jobs_dir = backend_dir / "jobs"
    video_path = Path(args.video).resolve()
    output_dir = Path(args.output_dir).resolve()
    match_id = output_dir.name

    start_time = time.time()

    def progress_callback(status: str, pct: float, label: str):
        update_job_status(
            jobs_dir=jobs_dir,
            job_id=args.job_id,
            status=status,
            progress_pct=pct,
            stage_label=label,
            start_time=start_time,
            match_id=match_id,
            video_path=str(video_path),
        )

    # Initial state
    progress_callback("detecting", 5.0, "Initializing Neural Models & Pipelines...")

    # Redirect stdout and stderr through progress parser (tqdm writes to stderr)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    stdout_parser = ProgressParserStream(orig_stdout, progress_callback)
    stderr_parser = ProgressParserStream(orig_stderr, progress_callback)
    sys.stdout = stdout_parser
    sys.stderr = stderr_parser

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # Guarantee `ffmpeg` is resolvable in PATH for web H.264 transcoding
        ffmpeg_bin = ensure_ffmpeg_in_path(backend_dir)
        if ffmpeg_bin:
            orig_stdout.write(f"[Worker] Using ffmpeg binary: {ffmpeg_bin}\n")
            orig_stdout.flush()

        # In local cpu_quick mode, avoid blocking on a 370MB HuggingFace SigLIP download
        # if not already cached locally (falls back cleanly to fast LAB + PCA in engine).
        if args.mode == "cpu_quick":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from engine.config import EngineConfig
        from engine.pipeline import MatchAnalysisPipeline

        models_dir = project_root / "models"
        players_model = str(models_dir / "players_best.pt")
        pitch_model = str(models_dir / "pitch_best.pt")
        v1_thresh = project_root / "artifacts" / "models_meta" / "optimal_thresholds_v1.json"
        legacy_thresh = project_root / "reports" / "optimal_thresholds.json"
        thresholds_path = str(v1_thresh if v1_thresh.exists() else legacy_thresh)

        if not Path(players_model).exists():
            raise FileNotFoundError(f"Players model not found: {players_model}")
        if not Path(pitch_model).exists():
            raise FileNotFoundError(f"Pitch model not found: {pitch_model}")

        # Optimize inference resolution for local CPU quick-scan (~4x faster on CPU)
        # while keeping full 1280px / 640px precision for remote_gpu or CUDA devices.
        use_cpu_fast = args.mode == "cpu_quick" and not torch.cuda.is_available()
        det_imgsz = 640 if use_cpu_fast else 1280
        pitch_imgsz = 512 if use_cpu_fast else 640
        orig_stdout.write(
            f"[Worker] Mode={args.mode} | CUDA={torch.cuda.is_available()} | "
            f"det_imgsz={det_imgsz}, pitch_imgsz={pitch_imgsz} | jersey_ocr={args.enable_jersey_ocr}\n"
        )
        orig_stdout.flush()

        cfg = EngineConfig(
            players_model_path=players_model,
            pitch_model_path=pitch_model,
            thresholds_path=thresholds_path if Path(thresholds_path).exists() else "",
            det_imgsz=det_imgsz,
            pitch_imgsz=pitch_imgsz,
            enable_jersey_ocr=bool(args.enable_jersey_ocr),
        )

        pipeline = MatchAnalysisPipeline(cfg)
        result = pipeline.run(
            video_path=str(video_path),
            output_dir=str(output_dir),
            max_seconds=args.max_seconds,
        )

        # Write metadata file in output_dir
        meta_payload = {
            "title": args.title,
            "is_user_uploaded": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "duration_analyzed_s": args.max_seconds,
        }
        with open(output_dir / "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, ensure_ascii=False)

        # Clean up temporary uploaded video file if located in tmp_uploads
        try:
            if "tmp_uploads" in str(video_path) and video_path.exists():
                video_path.unlink()
        except Exception as e:
            sys.stderr.write(f"Warning: could not delete temporary video {video_path}: {e}\n")

        # Mark completed
        update_job_status(
            jobs_dir=jobs_dir,
            job_id=args.job_id,
            status="completed",
            progress_pct=100.0,
            stage_label="Analysis Complete!",
            start_time=start_time,
            match_id=match_id,
        )
        return 0

    except Exception as e:
        error_msg = str(e)
        sys.stderr.write(f"\n[Worker Error] {error_msg}\n")
        update_job_status(
            jobs_dir=jobs_dir,
            job_id=args.job_id,
            status="failed",
            progress_pct=0.0,
            stage_label="Analysis Failed",
            start_time=start_time,
            error=error_msg,
            match_id=match_id,
        )
        # Attempt to clean up temp file
        try:
            if "tmp_uploads" in str(video_path) and video_path.exists():
                video_path.unlink()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
