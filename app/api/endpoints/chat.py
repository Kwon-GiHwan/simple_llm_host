"""
Chat completion endpoints (OpenAI-compatible)
"""
from fastapi import APIRouter, HTTPException
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionUsage, ChatMessage
from app.services.model_service import model_service
import time
import uuid


router = APIRouter()


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible)

    Converts chat messages to prompt and generates response
    """
    # Check if model is loaded
    if model_service.get_model(request.model) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not loaded. Available: {model_service.list_models()}"
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
        generated_text = model_service.generate_text(
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
