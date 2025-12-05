"""
API v1 route aggregation
"""
from fastapi import APIRouter
from app.api.v1.endpoints import models, chat


api_router = APIRouter()

# Include endpoint routers
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
