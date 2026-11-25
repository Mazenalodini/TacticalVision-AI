"""CSV / JSON data export for players and events."""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from ..services.match_service import match_service

router = APIRouter(prefix="/api/matches", tags=["export"])


@router.get("/{match_id}/export")
async def export_data(
    match_id: str,
    format: str = Query(..., pattern=r"^(csv|json)$"),
    scope: str = Query(..., pattern=r"^(players|events)$"),
):
    """Export players or events as downloadable CSV or JSON."""
    telemetry = match_service.get_full_telemetry(match_id)
    if telemetry is None:
        raise HTTPException(status_code=404, detail="Match not found")

    data: list[dict] = telemetry.get(scope, [])
    filename = f"{match_id}_{scope}.{format}"

    # ── JSON ────────────────────────────────────────────────────────
    if format == "json":
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ── CSV ─────────────────────────────────────────────────────────
    if not data:
        return StreamingResponse(
            iter(["No data\n"]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Collect all possible field names (events have mixed schemas)
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in data:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        flat: dict[str, object] = {}
        for k, v in row.items():
            flat[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        writer.writerow(flat)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
