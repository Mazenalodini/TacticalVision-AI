"""HTTP 206 Partial Content (Byte-Range) video streamer.

Enables instant seeking and event-scrubbing for <video> elements in every
browser by supporting standard Range request negotiation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

from fastapi.responses import FileResponse, StreamingResponse

CHUNK_SIZE = 1_048_576  # 1 MB per chunk


def parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    """Parse ``Range: bytes=START-END`` and return (start, end) inclusive."""
    range_spec = range_header.strip().split("=")[-1]
    parts = range_spec.split("-", 1)
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    # Clamp
    start = max(0, start)
    end = min(end, file_size - 1)
    return start, end


def _file_range_iterator(
    path: Path, start: int, end: int
) -> Generator[bytes, None, None]:
    """Synchronous generator — FastAPI runs it in a threadpool automatically."""
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_size = min(CHUNK_SIZE, remaining)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data


def create_video_response(
    file_path: Path,
    range_header: Optional[str] = None,
) -> FileResponse | StreamingResponse:
    """Return either a full 200 or a 206 Partial Content response."""
    file_size = file_path.stat().st_size

    if range_header is None:
        # No Range header → full file download
        return FileResponse(
            str(file_path),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    start, end = parse_range_header(range_header, file_size)
    content_length = end - start + 1

    return StreamingResponse(
        _file_range_iterator(file_path, start, end),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )
