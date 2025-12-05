"""
Example usage of the OpenAI-compatible LLM API
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def load_model(model_id: str, device: str = "cpu"):
    """Load a model"""
    response = requests.post(
        f"{BASE_URL}/v1/models",
        json={"model": model_id, "device": device}
    )
    print(f"Load model '{model_id}':")
    print(json.dumps(response.json(), indent=2))
    print("-" * 80)


def list_models():
    """List all loaded models"""
    response = requests.get(f"{BASE_URL}/v1/models")
    print("Available models:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 80)


def chat_completion(model_id: str, messages: list):
    """Create chat completion (OpenAI compatible)"""
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    print(f"Chat completion with '{model_id}':")
    print(json.dumps(response.json(), indent=2))
    print("-" * 80)


def unload_model(model_id: str):
    """Unload a model"""
    response = requests.delete(f"{BASE_URL}/v1/models/{model_id}")
    print(f"Unload model '{model_id}':")
    print(json.dumps(response.json(), indent=2))
    print("-" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("OpenAI-compatible LLM API - Example Usage")
    print("=" * 80)

    # 1. Load a model (uses cache if already downloaded)
    print("\n1. Loading model...")
    print("Note: First run downloads model, subsequent runs use cached version")
    load_model("gpt2", device="cpu")  # Use "cuda" if GPU available

    # 2. List loaded models
    print("\n2. Listing models...")
    list_models()

    # 3. Chat completion (OpenAI compatible)
    print("\n3. Creating chat completion...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is artificial intelligence?"}
    ]
    chat_completion("gpt2", messages)

    # 4. Another example
    print("\n4. Another chat completion...")
    messages = [
        {"role": "user", "content": "Write a short poem about coding."}
    ]
    chat_completion("gpt2", messages)

    # 5. Unload model (optional - cache remains on disk)
    # print("\n5. Unloading model...")
    # unload_model("gpt2")
    # list_models()

    print("\n" + "=" * 80)
    print("Example completed!")
    print("Model cache location: ~/.cache/huggingface/")
    print("=" * 80)
