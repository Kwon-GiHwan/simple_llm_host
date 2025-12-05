"""
Model Manager - Core business logic for model management

Responsibilities:
- Model loading with HuggingFace caching
- Model registry and lifecycle management
- Text generation with loaded models
- Memory management (GPU/CPU)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Optional
from app.core.config import settings


class ModelManager:
    """
    Manages multiple LLM models with automatic caching

    HuggingFace models are automatically cached to disk.
    Subsequent loads use the cache for instant loading.
    """

    def __init__(self):
        self._models: Dict[str, dict] = {}
        print(f"Model cache directory: {settings.HF_CACHE_DIR}")

    def load_model(self, model_id: str, device: Optional[str] = None) -> dict:
        """
        Load a model into memory (uses cached version if available)

        Args:
            model_id: HuggingFace model identifier
            device: Target device ('cuda' or 'cpu')

        Returns:
            Model information dict
        """
        # Check if already loaded
        if model_id in self._models:
            return {
                "model_id": model_id,
                "status": "already_loaded",
                "device": self._models[model_id]["device"]
            }

        # Determine device
        device = device or settings.DEFAULT_DEVICE

        print(f"Loading model: {model_id} on {device}")
        print("Note: First load downloads, subsequent loads use cache")

        # Load tokenizer (cached automatically)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=settings.HF_CACHE_DIR
        )

        # Load model (cached automatically)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            cache_dir=settings.HF_CACHE_DIR
        )

        if device == "cpu":
            model = model.to(device)

        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )

        # Store in registry
        self._models[model_id] = {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe,
            "device": device,
            "model_id": model_id
        }

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
            True if unloaded successfully
        """
        if model_id not in self._models:
            return False

        del self._models[model_id]

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Model '{model_id}' unloaded from memory (cache preserved)")
        return True

    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
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


# Global instance
model_manager = ModelManager()
