"""TacticalVision AI — Backend Configuration (Pydantic v2 Settings)."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All paths are computed relative to the project root (parent of backend/)."""

    project_root: Path = Path(__file__).resolve().parent.parent
    outputs_dir: Path = Path(__file__).resolve().parent.parent / "outputs"
    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    reports_dir: Path = Path(__file__).resolve().parent.parent / "reports"
    jobs_dir: Path = Path(__file__).resolve().parent / "jobs"
    tmp_uploads_dir: Path = Path(__file__).resolve().parent / "tmp_uploads"

    # Upload / Inference settings (CPU Quick-Scan)
    # max_seconds=0 means "Full Video" (no truncation)
    cpu_quick_max_seconds: float = 0.0
    cpu_quick_min_seconds: float = 2.0
    cpu_quick_max_seconds_limit: float = 600.0
    subprocess_timeout_s: int = 3600

    # Remote GPU worker (optional — empty string = disabled)
    remote_gpu_worker_url: str = ""

    # CORS
    frontend_origin: str = "http://localhost:3000"

    # Thumbnail cache settings
    thumb_width: int = 640
    thumb_height: int = 360
    thumb_quality: int = 82

    model_config = {"env_prefix": "TV_"}


settings = Settings()
settings.jobs_dir.mkdir(parents=True, exist_ok=True)
settings.tmp_uploads_dir.mkdir(parents=True, exist_ok=True)
