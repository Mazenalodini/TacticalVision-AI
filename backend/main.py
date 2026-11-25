"""TacticalVision AI — FastAPI Application Entry Point.

NOTE: engine/ is NEVER imported here.  PyTorch / Ultralytics are lazy-loaded
only inside job_runner (Phase 6) via a subprocess, keeping startup instant.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import matches, telemetry, video, export, assets, upload

app = FastAPI(
    title="TacticalVision AI",
    description="Broadcast-grade Football Tactical & Kinematic Analytics API",
    version="1.0.0",
)

# ------------------------------------------------------------------
# CORS — expose byte-range headers so the browser <video> element
# can seek freely and Content-Disposition for file downloads.
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Range",
        "Accept-Ranges",
        "Content-Length",
        "Content-Disposition",
    ],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
app.include_router(matches.router)
app.include_router(telemetry.router)
app.include_router(video.router)
app.include_router(export.router)
app.include_router(assets.router)
app.include_router(upload.router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "TacticalVision AI"}
