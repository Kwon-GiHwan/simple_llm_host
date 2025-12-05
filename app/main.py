"""
FastAPI application instance - Entry point for the API
"""
from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import router
from app.services.model_manager import model_manager
import torch


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application

    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
    )

    # Include API routes
    app.include_router(router)

    @app.get("/")
    async def root():
        """API root - basic information"""
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "endpoints": {
                "docs": "/docs",
                "openapi": "/openapi.json",
                "models": "/v1/models",
                "chat": "/v1/chat/completions"
            }
        }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "loaded_models": len(model_manager.list_models()),
            "cuda_available": torch.cuda.is_available(),
            "cache_dir": settings.HF_CACHE_DIR
        }

    return app


# Application instance
app = create_application()
