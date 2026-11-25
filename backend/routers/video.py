"""HTTP 206 Byte-Range video streaming for broadcast & tactical radar videos."""

from fastapi import APIRouter, HTTPException, Request

from ..services.match_service import match_service
from ..services.video_streamer import create_video_response

router = APIRouter(prefix="/api/matches", tags=["video"])


@router.get("/{match_id}/video/{video_type}")
async def stream_video(match_id: str, video_type: str, request: Request):
    if video_type not in ("broadcast", "radar"):
        raise HTTPException(
            status_code=400,
            detail="video_type must be 'broadcast' or 'radar'",
        )

    path = match_service.get_video_path(match_id, video_type)
    if path is None:
        raise HTTPException(status_code=404, detail="Video not found")

    range_header = request.headers.get("range")
    return create_video_response(path, range_header)
