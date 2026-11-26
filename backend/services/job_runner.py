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
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..config import settings


class JobRunner:
    def __init__(self) -> None:
        self.jobs_dir = settings.jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._active_processes: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, Any] = {}

    def start_job(
        self,
        video_path: Path,
        title: str,
        mode: str = "cpu_quick",
        max_seconds: float = 3.0,
        enable_jersey_ocr: bool = True,
    ) -> str:
        """Create and spawn an isolated worker subprocess."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        match_id = f"user_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        output_dir = settings.outputs_dir / match_id

        # Clamp max_seconds for CPU quick-scan
        clamped_seconds = max(
            settings.cpu_quick_min_seconds,
            min(max_seconds, settings.cpu_quick_max_seconds_limit),
        )

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

    def get_job_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Read current job status from backend/jobs/{job_id}.json."""
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            return None

        try:
            with open(job_file, "r", encoding="utf-8") as f:
                data = json.load(f)

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
            return None

    def cancel_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Safely terminate an active worker subprocess, mark cancelled, and clean up temp files."""
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            return None

        try:
            with open(job_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
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

        # 2. Terminate subprocess cleanly on Windows (and POSIX)
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
        tmp = self.jobs_dir / f"{job_id}.json.tmp"
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
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(target)


# Singleton
job_runner = JobRunner()
