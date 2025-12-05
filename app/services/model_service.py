"""
Model management service (business logic layer)
"""
import torch
from typing import Dict, List, Optional
from app.core.model_loader import load_huggingface_model
from app.core.config import settings


class ModelService:
    """
    Service for managing LLM models

    Responsibilities:
    - Model lifecycle management (load, unload)
    - Model registry management
    - Text generation with loaded models
    """

    def __init__(self):
        self._models: Dict[str, dict] = {}

    def load_model(self, model_id: str, device: Optional[str] = None) -> dict:
        """
        Load a model into memory

        Args:
            model_id: HuggingFace model identifier
            device: Target device ('cuda' or 'cpu')

        Returns:
            Model information

        Raises:
            ValueError: If model already loaded
        """
        if model_id in self._models:
            return {
                "model_id": model_id,
                "status": "already_loaded",
                "device": self._models[model_id]["device"]
            }

        device = device or settings.DEFAULT_DEVICE
        model_data = load_huggingface_model(model_id, device)
        self._models[model_id] = model_data

        return {
            "model_id": model_id,
            "status": "loaded",
            "device": device
        }

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get loaded model data"""
        return self._models.get(model_id)

    def list_models(self) -> List[str]:
        """Get list of loaded model IDs"""
        return list(self._models.keys())

    def unload_model(self, model_id: str) -> bool:
        """
        Unload model from memory (cache preserved on disk)

        Args:
            model_id: Model to unload

        Returns:
            True if unloaded, False if not found
        """
        if model_id not in self._models:
            return False

        del self._models[model_id]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> str:
        """
        Generate text using a loaded model

        Args:
            model_id: Model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text

        Raises:
            ValueError: If model not loaded
        """
        model_data = self.get_model(model_id)
        if model_data is None:
            raise ValueError(f"Model '{model_id}' not loaded")

        pipe = model_data["pipeline"]

        # Use defaults if not provided
        max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS
        temperature = temperature or settings.DEFAULT_TEMPERATURE
        top_p = top_p or settings.DEFAULT_TOP_P

        # Generate
        output = pipe(
            prompt,
            max_length=len(pipe.tokenizer.encode(prompt)) + max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        return output[0]["generated_text"]


# Global service instance
model_service = ModelService()
