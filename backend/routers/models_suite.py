from fastapi import APIRouter

from ..services import action_spotting_service

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/suite")
async def get_models_suite():
    """Return the unified 4-model registry, evaluation metrics, ONNX CPU benchmarks, and runtime status."""
    return action_spotting_service.get_models_suite()
