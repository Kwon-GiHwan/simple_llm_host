"""
Application configuration
"""
import os
import torch


class Settings:
    """Application settings"""

    # API Settings
    API_TITLE: str = "OpenAI-compatible LLM API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Multi-model LLM serving with OpenAI-compatible endpoints"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Settings
    DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    HF_CACHE_DIR: str = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # Generation Settings
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9


settings = Settings()
