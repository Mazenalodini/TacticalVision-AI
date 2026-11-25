"""Full telemetry.json endpoint — returns the raw JSON file for maximum speed.

The 2.4 MB file is served via FileResponse (sendfile where possible)
so it never needs to be parsed/re-serialised by Pydantic on every request.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.match_service import match_service

router = APIRouter(prefix="/api/matches", tags=["telemetry"])


@router.get("/{match_id}/telemetry")
async def get_telemetry(match_id: str):
    path = match_service.get_telemetry_path(match_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Telemetry not found")
    return FileResponse(str(path), media_type="application/json")
