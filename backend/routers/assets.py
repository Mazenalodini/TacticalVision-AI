"""Thumbnail endpoint — generates and caches a fast 640×360 JPEG.

The original preview_snapshot.png is ultra-wide (~3 MB, broadcast+radar
side-by-side).  We crop the left 16:9 broadcast portion, resize to 640×360,
and cache as _thumb_640.jpg (~40 KB) for instant Match Hub card loading.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..services.match_service import match_service

router = APIRouter(prefix="/api/matches", tags=["assets"])


def _generate_thumbnail(match_id: str) -> Path | None:
    """Generate (or return cached) JPEG thumbnail."""
    preview_path = match_service.get_preview_path(match_id)
    if preview_path is None:
        return None

    match_dir = settings.outputs_dir / match_id
    thumb_path = match_dir / "_thumb_640.jpg"

    # Return cached version if it exists and is newer than the source
    if thumb_path.exists():
        if thumb_path.stat().st_mtime >= preview_path.stat().st_mtime:
            return thumb_path

    try:
        from PIL import Image  # lazy — no need to load PIL at server startup

        img = Image.open(preview_path)
        w, h = img.size

        # Crop the left 16:9 broadcast portion
        crop_w = min(int(h * 16 / 9), w)
        img_cropped = img.crop((0, 0, crop_w, h))

        # Resize to target thumbnail dimensions
        img_thumb = img_cropped.resize(
            (settings.thumb_width, settings.thumb_height),
            Image.LANCZOS,
        )

        # Ensure RGB (PNG may carry alpha channel)
        if img_thumb.mode in ("RGBA", "P", "LA"):
            img_thumb = img_thumb.convert("RGB")

        img_thumb.save(str(thumb_path), "JPEG", quality=settings.thumb_quality)
        return thumb_path
    except Exception:
        return None


@router.get("/{match_id}/thumbnail")
async def get_thumbnail(match_id: str):
    thumb_path = _generate_thumbnail(match_id)
    if thumb_path is None:
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    return FileResponse(
        str(thumb_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
