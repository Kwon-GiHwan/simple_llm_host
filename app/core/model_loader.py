"""
Model loading functionality
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict
from app.core.config import settings


def load_huggingface_model(model_id: str, device: str) -> Dict:
    """
    Load a HuggingFace model with caching support

    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Dictionary containing model components
    """
    print(f"Loading model: {model_id} on {device}")
    print(f"Cache directory: {settings.HF_CACHE_DIR}")

    # Load tokenizer (uses cache automatically)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=settings.HF_CACHE_DIR
    )

    # Load model (uses cache automatically)
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

    return {
        "tokenizer": tokenizer,
        "model": model,
        "pipeline": pipe,
        "device": device,
        "model_id": model_id
    }
