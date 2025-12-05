"""
API routes - All endpoints (/v1/chat, /v1/models) defined here
"""
from fastapi import APIRouter, HTTPException
from app.schemas.protocol import (
    ModelRegisterRequest,
    ModelListResponse,
    ModelInfo,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatMessage,
)
from app.services.model_manager import model_manager
import time
import uuid


router = APIRouter()


# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.post("/v1/models")
async def load_model(request: ModelRegisterRequest):
    """
    Load a model into memory

    First load downloads from HuggingFace Hub.
    Subsequent loads use cached version.
    """
    try:
        result = model_manager.load_model(request.model, request.device)
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


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List all loaded models (OpenAI-compatible)"""
    models = model_manager.list_models()
    return ModelListResponse(
        data=[
            ModelInfo(
                id=model_id,
                created=int(time.time())
            )
            for model_id in models
        ]
    )


@router.delete("/v1/models/{model_id}")
async def unload_model(model_id: str):
    """
    Unload a model from memory

    Note: Model cache remains on disk for future use
    """
    success = model_manager.unload_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return {
        "id": model_id,
        "object": "model",
        "deleted": True
    }


# ============================================================================
# Chat Completion Endpoints (OpenAI-compatible)
# ============================================================================

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible)

    Converts chat messages to prompt and generates response
    """
    # Check if model is loaded
    if model_manager.get_model(request.model) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not loaded. Available: {model_manager.list_models()}"
        )

    # Convert chat messages to prompt
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")

    prompt = "\n".join(prompt_parts) + "\nAssistant:"

    try:
        # Generate response
        generated_text = model_manager.generate_text(
            model_id=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        # Extract assistant response (remove prompt)
        assistant_response = generated_text[len(prompt):].strip()

        # Estimate token counts
        prompt_tokens = len(prompt.split())
        completion_tokens = len(assistant_response.split())

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=assistant_response
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
