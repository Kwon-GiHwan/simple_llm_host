# OpenAI-Compatible LLM API Server

FastAPI-based multi-model LLM serving system with OpenAI-compatible endpoints and automatic model caching.

## Features

- ✅ OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`)
- ✅ Automatic model caching (models downloaded once, reused forever)
- ✅ Multi-model support (load and switch between multiple models)
- ✅ GPU/CPU auto-detection
- ✅ HuggingFace model integration

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Server

```bash
python main.py
```

Server runs on `http://localhost:8000`

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Model Caching

**Models are automatically cached!**

- First load: Downloads from HuggingFace Hub
- Subsequent loads: Uses cached version from `~/.cache/huggingface/`
- Server restart: Models load from cache instantly
- Unloading: Removes from memory but keeps cache on disk

## API Usage (OpenAI Compatible)

### 1. Load a Model

```bash
curl -X POST "http://localhost:8000/v1/models" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "device": "cpu"
  }'
```

**Available devices:** `cpu`, `cuda`

### 2. List Models

```bash
curl http://localhost:8000/v1/models
```

### 3. Chat Completion (OpenAI Compatible)

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is AI?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Parameters:**
- `model`: Model ID to use
- `messages`: Array of chat messages
- `max_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature 0.0-2.0 (default: 0.7)
- `top_p`: Nucleus sampling (default: 0.9)

### 4. Unload Model

```bash
curl -X DELETE "http://localhost:8000/v1/models/gpt2"
```

**Note:** Unloading removes model from memory but preserves cache on disk.

## Python Client Example

```python
import requests

# Load model
requests.post("http://localhost:8000/v1/models",
    json={"model": "gpt2", "device": "cpu"})

# Chat completion
response = requests.post("http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt2",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 50
    })

print(response.json()["choices"][0]["message"]["content"])
```

Or run the example script:

```bash
python example_usage.py
```

## Recommended Models

| Model | HuggingFace ID | Size | Speed |
|-------|---------------|------|-------|
| GPT-2 | `gpt2` | ~500MB | Fast |
| GPT-2 Medium | `gpt2-medium` | ~1.5GB | Medium |
| GPT-2 Large | `gpt2-large` | ~3GB | Slow |
| DistilGPT-2 | `distilgpt2` | ~350MB | Very Fast |
| GPT-Neo 125M | `EleutherAI/gpt-neo-125m` | ~500MB | Fast |
| GPT-Neo 1.3B | `EleutherAI/gpt-neo-1.3B` | ~5GB | Slow |

## Project Structure

```
simple_llm_host/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── routes.py         # API endpoints
│   ├── schemas.py        # Pydantic models
│   └── model_manager.py  # Model management
├── main.py               # Entry point
├── example_usage.py      # Usage examples
└── requirements.txt
```

## OpenAI SDK Compatibility

This API is compatible with OpenAI's Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required
)

response = client.chat.completions.create(
    model="gpt2",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Environment Variables

- `HF_HOME`: Set custom HuggingFace cache directory (default: `~/.cache/huggingface/`)

```bash
export HF_HOME=/path/to/cache
python main.py
```

## GPU Support

The server automatically detects CUDA availability:

```bash
# Force CPU mode
curl -X POST "http://localhost:8000/v1/models" \
  -d '{"model": "gpt2", "device": "cpu"}'

# Use GPU (if available)
curl -X POST "http://localhost:8000/v1/models" \
  -d '{"model": "gpt2", "device": "cuda"}'
```

## Troubleshooting

### CUDA Out of Memory

Use smaller model or CPU mode:
```bash
curl -X POST "http://localhost:8000/v1/models" \
  -d '{"model": "distilgpt2", "device": "cpu"}'
```

### Clear Cache

```bash
rm -rf ~/.cache/huggingface/
```

### Model Download Slow

First download is slow. Subsequent loads use cache and are instant.

## License

MIT License - Educational and research purposes.
