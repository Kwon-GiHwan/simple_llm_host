"""
Model-related schemas
"""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


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
