"""
Model management endpoints
"""
from fastapi import APIRouter, HTTPException
from app.schemas import ModelRegisterRequest, ModelListResponse, ModelInfo
from app.services.model_service import model_service
import time


router = APIRouter()


@router.post("")
async def load_model(request: ModelRegisterRequest):
    """
    Load a model into memory

    First load downloads from HuggingFace Hub.
    Subsequent loads use cached version.
    """
    try:
        result = model_service.load_model(request.model, request.device)
        return {
            "id": result["model_id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "huggingface",
            "status": result["status"],
            "device": result["device"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all loaded models (OpenAI-compatible)"""
    models = model_service.list_models()
    return ModelListResponse(
        data=[
            ModelInfo(
                id=model_id,
                created=int(time.time())
            )
            for model_id in models
        ]
    )


@router.delete("/{model_id}")
async def unload_model(model_id: str):
    """
    Unload a model from memory

    Note: Model cache remains on disk for future use
    """
    success = model_service.unload_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return {
        "id": model_id,
        "object": "model",
        "deleted": True
    }
