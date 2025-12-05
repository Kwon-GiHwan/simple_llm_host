"""
OpenAI-compatible API protocol schemas
All request/response models defined here
"""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# ============================================================================
# Model Management Schemas
# ============================================================================

class ModelRegisterRequest(BaseModel):
    """Request to register/load a model"""
    model: str = Field(..., description="HuggingFace model identifier")
    device: Optional[str] = Field(default=None, description="Device: 'cuda' or 'cpu'")


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "huggingface"


class ModelListResponse(BaseModel):
    """List of models response"""
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ============================================================================
# Chat Completion Schemas (OpenAI-compatible)
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)"""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response (OpenAI-compatible)"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
