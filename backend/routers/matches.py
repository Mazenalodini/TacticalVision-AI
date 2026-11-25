"""Match listing & detail endpoints."""

from fastapi import APIRouter, HTTPException

from ..services.match_service import match_service

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.get("")
async def list_matches():
    """Return summary cards for all discovered matches."""
    return match_service.discover_matches()


@router.get("/{match_id}")
async def get_match(match_id: str):
    """Return summary card for a single match."""
    if match_service.get_match_dir(match_id) is None:
        raise HTTPException(status_code=404, detail="Match not found")

    all_matches = match_service.discover_matches()
    match = next((m for m in all_matches if m["id"] == match_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return match
