"""Job Runner Service.

Manages background worker subprocesses on Windows without multiprocessing.Process,
writing stage updates atomically to backend/jobs/{job_id}.json.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..config import settings
from .remote_gpu_client import (
    cancel_remote_gpu_job_best_effort,
    run_remote_gpu_bridge,
)

# Module-level reentrant lock protecting all job_*.json reads & atomic replaces on Windows
_JOB_IO_LOCK = threading.RLock()


def _atomic_write_json(target: Path, payload: dict[str, Any], max_retries: int = 10) -> None:
    """Thread-safe and Windows-safe atomic JSON write with unique .tmp and retry backoff."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f"{target.name}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp"
    with _JOB_IO_LOCK:
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()

            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    os.replace(tmp, target)
                    return
                except (PermissionError, OSError) as exc:
                    last_exc = exc
                    if attempt + 1 < max_retries:
                        backoff = min(0.08, 0.02 * (1.35 ** attempt))
                        time.sleep(backoff)
            if last_exc is not None:
                raise last_exc
        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass


def _safe_read_json(path: Path, max_retries: int = 10) -> Optional[dict[str, Any]]:
    """Thread-safe and Windows-safe JSON read with retry backoff against transient locks."""
    with _JOB_IO_LOCK:
        for attempt in range(max_retries):
            if not path.exists():
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                return None
            except (PermissionError, OSError, json.JSONDecodeError):
                if attempt + 1 < max_retries:
                    backoff = min(0.08, 0.02 * (1.35 ** attempt))
                    time.sleep(backoff)
        return None


class JobRunner:
    def __init__(self) -> None:
        self.jobs_dir = settings.jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._active_processes: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, Any] = {}
        self._remote_cancel_events: dict[str, threading.Event] = {}
        self._remote_job_refs: dict[str, dict[str, str]] = {}

    def start_job(
        self,
        video_path: Path,
        title: str,
        mode: str = "cpu_quick",
        max_seconds: float = 0.0,
        start_seconds: float = 0.0,
        enable_jersey_ocr: bool = True,
    ) -> str:
        """Create and spawn an isolated worker subprocess."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        match_id = f"user_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        output_dir = settings.outputs_dir / match_id

        # max_seconds=0 means "Full Video" (no truncation).
        # Otherwise clamp within sane bounds for CPU processing.
        if max_seconds <= 0:
            clamped_seconds = 0.0  # Full video — no truncation
        else:
            clamped_seconds = max(
                settings.cpu_quick_min_seconds,
                min(max_seconds, settings.cpu_quick_max_seconds_limit),
            )
        clamped_start = max(0.0, float(start_seconds or 0.0))

        cmd = [
            sys.executable,
            "-m",
            "backend.services.worker",
            "--job-id",
            job_id,
            "--video",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--title",
            title,
            "--max-seconds",
            str(clamped_seconds),
            "--start-seconds",
            str(clamped_start),
            "--mode",
            mode,
            "--enable-jersey-ocr" if enable_jersey_ocr else "--no-jersey-ocr",
        ]

        # Stream stdout/stderr to a log file (avoids Windows 4KB subprocess.PIPE deadlock)
        log_path = self.jobs_dir / f"{job_id}.log"
        log_fh = open(log_path, "w", encoding="utf-8", errors="replace")

        # Spawn isolated subprocess (Windows-safe: avoids spawn closures)
        proc = subprocess.Popen(
            cmd,
            cwd=str(settings.project_root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._active_processes[job_id] = proc
        self._log_handles[job_id] = log_fh

        # Write initial queued status with worker PID and temp video path
        self._write_status(
            job_id=job_id,
            status="queued",
            progress_pct=0.0,
            stage_label="Queued for Tactical Analysis",
            elapsed_s=0.0,
            match_id=match_id,
            pid=proc.pid,
            video_path=str(video_path),
        )
        return job_id

    def start_remote_gpu_job(
        self,
        video_path: Path,
        title: str,
        remote_gpu_url: str,
        max_seconds: float = 0.0,
        start_seconds: float = 0.0,
        enable_jersey_ocr: bool = True,
        poll_interval_s: float = 2.0,
    ) -> str:
        """Start an async Remote GPU Worker proxy job (Kaggle / Colab)."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        match_id = f"user_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        output_dir = settings.outputs_dir / match_id
        clamped_seconds = max(0.0, float(max_seconds or 0.0))
        clamped_start = max(0.0, float(start_seconds or 0.0))

        cancel_event = threading.Event()
        self._remote_cancel_events[job_id] = cancel_event
        self._remote_job_refs[job_id] = {"url": remote_gpu_url.strip().rstrip("/"), "remote_job_id": ""}

        self._write_status(
            job_id=job_id,
            status="queued",
            progress_pct=2.0,
            stage_label="Queued for Remote GPU Worker",
            elapsed_s=0.0,
            match_id=match_id,
            pid=None,
            video_path=str(video_path),
        )

        def _on_status(
            st: str,
            pct: float,
            label: str,
            elapsed: float,
            err: Optional[str],
        ) -> None:
            if cancel_event.is_set() and st != "cancelled":
                return
            self._write_status(
                job_id=job_id,
                status=st,
                progress_pct=round(pct, 1),
                stage_label=label,
                elapsed_s=elapsed,
                error=err,
                match_id=match_id,
                pid=None,
                video_path=None if st in ("completed", "failed", "cancelled") else str(video_path),
            )
            if st in ("completed", "failed", "cancelled"):
                self._remote_cancel_events.pop(job_id, None)
                self._remote_job_refs.pop(job_id, None)

        def _on_remote_job_id(rid: str) -> None:
            ref = self._remote_job_refs.get(job_id)
            if ref is not None:
                ref["remote_job_id"] = rid

        t = threading.Thread(
            target=run_remote_gpu_bridge,
            kwargs={
                "local_job_id": job_id,
                "match_id": match_id,
                "video_path": video_path,
                "output_dir": output_dir,
                "title": title,
                "target_url": remote_gpu_url,
                "max_seconds": clamped_seconds,
                "start_seconds": clamped_start,
                "enable_jersey_ocr": enable_jersey_ocr,
                "cancel_event": cancel_event,
                "on_status": _on_status,
                "on_remote_job_id": _on_remote_job_id,
                "poll_interval_s": poll_interval_s,
            },
            daemon=True,
        )
        t.start()
        return job_id

    def get_job_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Read current job status from backend/jobs/{job_id}.json."""
        job_file = self.jobs_dir / f"{job_id}.json"
        data = _safe_read_json(job_file)
        if data is None:
            return None

        try:
            # Check if process exited unexpectedly without updating terminal status
            proc = self._active_processes.get(job_id)
            if proc and proc.poll() is not None:
                self._close_log_handle(job_id)
                self._active_processes.pop(job_id, None)

                if proc.returncode != 0 and data.get("status") not in ("failed", "completed", "cancelled"):
                    err_summary = self._read_log_tail(job_id) or "Worker process terminated unexpectedly."
                    data["status"] = "failed"
                    data["error"] = err_summary
                    data["pid"] = None
                    self._write_status(
                        job_id=job_id,
                        status="failed",
                        progress_pct=0.0,
                        stage_label="Analysis Failed",
                        elapsed_s=data.get("elapsed_s", 0.0),
                        error=err_summary,
                        match_id=data.get("match_id"),
                        pid=None,
                        video_path=data.get("video_path"),
                    )

            return data
        except Exception:
            return data

    def cancel_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Safely terminate an active worker subprocess, mark cancelled, and clean up temp files."""
        job_file = self.jobs_dir / f"{job_id}.json"
        data = _safe_read_json(job_file)
        if data is None:
            return None

        if data.get("status") in ("completed", "failed", "cancelled"):
            return data

        pid = data.get("pid")
        match_id = data.get("match_id")
        video_path_str = data.get("video_path")

        # 1. Mark cancelled first so worker won't overwrite status while terminating
        self._write_status(
            job_id=job_id,
            status="cancelled",
            progress_pct=data.get("progress_pct", 0.0),
            stage_label="Analysis Cancelled",
            elapsed_s=data.get("elapsed_s", 0.0),
            error=None,
            match_id=match_id,
            pid=None,
            video_path=None,
        )

        # 2. Signal remote GPU proxy thread (if applicable) & send best-effort remote cancel
        cancel_ev = self._remote_cancel_events.pop(job_id, None)
        if cancel_ev is not None:
            cancel_ev.set()
        remote_ref = self._remote_job_refs.pop(job_id, None)
        if remote_ref and remote_ref.get("url") and remote_ref.get("remote_job_id"):
            threading.Thread(
                target=cancel_remote_gpu_job_best_effort,
                args=(remote_ref["url"], remote_ref["remote_job_id"]),
                daemon=True,
            ).start()

        # 3. Terminate subprocess cleanly on Windows (and POSIX)
        proc = self._active_processes.pop(job_id, None)
        target_pid = proc.pid if proc is not None else pid

        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass

        if target_pid:
            try:
                if os.name == "nt":
                    # Ensure entire process tree (e.g. spawned ffmpeg) is terminated
                    subprocess.run(
                        ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0,
                    )
                else:
                    os.kill(int(target_pid), signal.SIGTERM)
            except Exception:
                pass

        self._close_log_handle(job_id)

        # 3. Clean up temporary uploaded video file
        if video_path_str:
            try:
                vpath = Path(video_path_str).resolve()
                if settings.tmp_uploads_dir.resolve() in vpath.parents and vpath.exists():
                    vpath.unlink(missing_ok=True)
            except Exception:
                pass

        # 4. Clean up incomplete output directory (only if telemetry.json wasn't finished)
        if match_id and str(match_id).startswith("user_"):
            try:
                out_dir = (settings.outputs_dir / match_id).resolve()
                if (
                    out_dir.parent == settings.outputs_dir.resolve()
                    and out_dir.is_dir()
                    and not (out_dir / "telemetry.json").exists()
                ):
                    shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass

        return self.get_job_status(job_id)

    def _close_log_handle(self, job_id: str) -> None:
        fh = self._log_handles.pop(job_id, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    def _read_log_tail(self, job_id: str, max_lines: int = 8) -> str:
        log_path = self.jobs_dir / f"{job_id}.log"
        if not log_path.exists():
            return ""
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "%|" not in ln]
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""

    def _write_status(
        self,
        job_id: str,
        status: str,
        progress_pct: float,
        stage_label: str,
        elapsed_s: float,
        error: str | None = None,
        match_id: str | None = None,
        pid: int | None = None,
        video_path: str | None = None,
    ) -> None:
        target = self.jobs_dir / f"{job_id}.json"
        payload = {
            "job_id": job_id,
            "status": status,
            "progress_pct": progress_pct,
            "stage_label": stage_label,
            "elapsed_s": elapsed_s,
            "error": error,
            "match_id": match_id,
            "pid": pid,
            "video_path": video_path,
        }
        _atomic_write_json(target, payload)


# Singleton
job_runner = JobRunner()
