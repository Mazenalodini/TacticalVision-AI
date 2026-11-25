"""Match discovery and data access service.

Scans `outputs/` for pre-analyzed and user-uploaded matches,
caches telemetry summaries in-memory for fast listing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..config import settings


class MatchService:
    def __init__(self) -> None:
        # In-memory cache: match_id → {"mtime": float, "data": dict}
        self._cache: dict[str, dict[str, Any]] = {}

    # ── Discovery ───────────────────────────────────────────────────

    def discover_matches(self) -> list[dict[str, Any]]:
        """Return MatchCard-shaped dicts for every valid match in outputs/."""
        matches: list[dict[str, Any]] = []
        if not settings.outputs_dir.is_dir():
            return matches

        for match_dir in sorted(settings.outputs_dir.iterdir()):
            if not match_dir.is_dir():
                continue
            telemetry_path = match_dir / "telemetry.json"
            if not telemetry_path.exists():
                continue

            data = self._load_telemetry(match_dir.name, telemetry_path)
            if data is None:
                continue

            # Check for custom _meta.json from user upload
            meta_path = match_dir / "_meta.json"
            custom_title = None
            is_user = match_dir.name.startswith("user_")
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                        custom_title = meta.get("title")
                        is_user = bool(meta.get("is_user_uploaded", is_user))
                except Exception:
                    pass

            s = data["summary"]
            mi = data["match_info"]
            ed = data.get("engine_diagnostics", {})
            homo = ed.get("homography", {})
            pitch_ctrl = s.get("pitch_control_pct", {})

            matches.append({
                "id": match_dir.name,
                "title": custom_title or self._format_title(match_dir.name),
                "duration_s": s["duration_s"],
                "fps": s["fps"],
                "resolution": mi["resolution"],
                "possession_pct": s["possession_pct"],
                "pitch_control_pct": {
                    "team_0": float(pitch_ctrl.get("team_0", 50.0)),
                    "team_1": float(pitch_ctrl.get("team_1", 50.0)),
                },
                "passes": s["passes"],
                "turnovers": s["turnovers"],
                "top_speed_kmh": s["top_speed_kmh"],
                "expected_threat_xt": s.get("expected_threat_xt", {"team_0": 0.0, "team_1": 0.0}),
                "events_count": len(data.get("events", [])),
                "players_count": len(data.get("players", [])),
                "ball_coverage_pct": float(ed.get("ball_coverage_pct", 0.0)),
                "homography_stability_pct": float(homo.get("stability_rate_pct", 100.0)),
                "has_thumbnail": (match_dir / "preview_snapshot.png").exists(),
                "is_user_uploaded": is_user,
                "source_video": mi["source_video"],
            })
        return matches

    # ── Data Access ────────────────────────────────────────────────

    def get_telemetry_path(self, match_id: str) -> Optional[Path]:
        p = settings.outputs_dir / match_id / "telemetry.json"
        return p if p.exists() else None

    def get_video_path(self, match_id: str, video_type: str) -> Optional[Path]:
        filename = (
            "broadcast_analyzed.mp4" if video_type == "broadcast"
            else "tactical_radar.mp4"
        )
        p = settings.outputs_dir / match_id / filename
        return p if p.exists() else None

    def get_preview_path(self, match_id: str) -> Optional[Path]:
        p = settings.outputs_dir / match_id / "preview_snapshot.png"
        return p if p.exists() else None

    def get_match_dir(self, match_id: str) -> Optional[Path]:
        p = settings.outputs_dir / match_id
        return p if p.is_dir() and (p / "telemetry.json").exists() else None

    def get_full_telemetry(self, match_id: str) -> Optional[dict[str, Any]]:
        """Return the full parsed telemetry dict (cached)."""
        path = self.get_telemetry_path(match_id)
        if path is None:
            return None
        return self._load_telemetry(match_id, path)

    def invalidate_cache(self, match_id: Optional[str] = None) -> None:
        """Invalidate in-memory cache for a specific match or all matches."""
        if match_id:
            self._cache.pop(match_id, None)
        else:
            self._cache.clear()

    # ── Internal ───────────────────────────────────────────────────

    def _load_telemetry(
        self, match_id: str, path: Path
    ) -> Optional[dict[str, Any]]:
        """Load + cache telemetry JSON (invalidated by mtime change)."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None

        cached = self._cache.get(match_id)
        if cached and cached["mtime"] == mtime:
            return cached["data"]

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        self._cache[match_id] = {"mtime": mtime, "data": data}
        return data

    @staticmethod
    def _format_title(match_id: str) -> str:
        """'match_01_counter' → 'Match 01 Counter'"""
        return match_id.replace("_", " ").title()


# Singleton instance used by all routers
match_service = MatchService()
