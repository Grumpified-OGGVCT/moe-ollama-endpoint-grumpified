# MoE Ollama Endpoint

A production-grade OpenAI-compatible endpoint orchestrating Ollama Cloud models as a Mixture of Experts (MoE) system with RAG, vision, and tool support.

## Features

- 🎯 **Mixture of Experts (MoE)**: Intelligent routing to specialized models (general, code, reasoning, vision)
- 🧠 **DSPy Integration**: Smart routing decisions using DSPy framework
- 📚 **RAG Support**: Retrieval-Augmented Generation with PostgreSQL/pgvector
- 👁️ **Vision Models**: Multi-modal support for image understanding
- 🛠️ **Tool Support**: Function calling capabilities
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI API
- 🐳 **Podman/Docker**: Full containerization support
- 🚀 **FastAPI**: High-performance async API

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      FastAPI Endpoint           │
│  (OpenAI Compatible API)        │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│      DSPy Router (MoE)          │
│  ┌─────────────────────────┐   │
│  │ - General: llama3.1:8b  │   │
│  │ - Code: codellama:13b   │   │
│  │ - Reasoning: llama3.1:70b│  │
│  │ - Vision: llava:13b     │   │
│  └─────────────────────────┘   │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│     Ollama Cloud Models         │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│   PostgreSQL + pgvector         │
│   (RAG Document Storage)        │
└─────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Podman or Docker
- PostgreSQL with pgvector extension (handled by containers)
- Ollama Cloud API key

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Grumpified-OGGVCT/moe-ollama-endpoint-grumpified.git
cd moe-ollama-endpoint-grumpified
```

2. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your Ollama API key
```

3. **Run with Podman Compose** (recommended):
```bash
podman-compose up -d
```

Or with Docker Compose:
```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

### Manual Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up PostgreSQL with pgvector**:
```bash
# Install PostgreSQL and pgvector extension
# Or use the provided containers
```

3. **Run the application**:
```bash
python -m app.main
```

## Usage

### OpenAI-Compatible Chat API

```python
import openai

# Configure client to use local endpoint
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key for Ollama is configured server-side
)

# Create chat completion
response = client.chat.completions.create(
    model="auto",  # Let MoE router choose the best model
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

### Vision Model Support

```python
# Multi-modal request with image
response = client.chat.completions.create(
    model="auto",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
```

### RAG (Retrieval-Augmented Generation)

1. **Ingest documents**:
```python
import httpx

# Ingest documents into RAG system
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/v1/rag/ingest",
        json={
            "documents": [
                {
                    "content": "Your document content here...",
                    "metadata": {"source": "manual", "topic": "AI"},
                    "collection": "knowledge_base"
                }
            ]
        }
    )
    print(response.json())
```

2. **Query with RAG**:
```python
# Enable RAG in chat completion
response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "user", "content": "What does the documentation say about AI?"}
    ],
    extra_body={
        "use_rag": True,
        "rag_collections": ["knowledge_base"]
    }
)
```

### Embeddings

```python
# Generate embeddings
response = client.embeddings.create(
    model="nomic-embed-text",
    input="Text to embed"
)

print(response.data[0].embedding)
```

### List Models

```python
# List available models
models = client.models.list()
for model in models.data:
    print(model.id)
```

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama API base URL | `https://api.ollama.cloud` |
| `OLLAMA_API_KEY` | Your Ollama API key | Required |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_USER` | PostgreSQL user | `moe_user` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `moe_password` |
| `POSTGRES_DB` | PostgreSQL database | `moe_rag` |
| `DEFAULT_MODEL` | Default model for general queries | `llama3.1:8b` |
| `VISION_MODEL` | Model for vision tasks | `llava:13b` |
| `CODE_MODEL` | Model for code tasks | `codellama:13b` |
| `REASONING_MODEL` | Model for complex reasoning | `llama3.1:70b` |
| `EMBEDDING_MODEL` | Model for embeddings | `nomic-embed-text` |

## API Endpoints

### Chat Completions
- `POST /v1/chat/completions` - Create chat completion
  - Supports streaming with `stream: true`
  - Automatic MoE routing with `model: "auto"`
  - RAG support with `use_rag: true`

### Models
- `GET /v1/models` - List available models
- `GET /v1/models/{model_id}` - Get model information

### Embeddings
- `POST /v1/embeddings` - Create embeddings

### RAG
- `POST /v1/rag/ingest` - Ingest documents
- `GET /v1/rag/search` - Search documents

### Health
- `GET /health` - Health check
- `GET /v1/health` - Health check (versioned)

## MoE Routing Strategy

The DSPy-based router intelligently selects models based on:

1. **Vision Detection**: Automatically routes to vision model if images are present
2. **Code Keywords**: Routes to code model for programming-related queries
3. **Reasoning Keywords**: Routes to reasoning model for complex analysis
4. **RAG Keywords**: Enables RAG for retrieval-based queries
5. **Default**: Falls back to general-purpose model

Keywords are configurable in `app/services/router.py`.

## Development

### Run Tests
```bash
pytest
```

### Code Formatting
```bash
black app/
```

### Linting
```bash
ruff check app/
```

### Type Checking
```bash
mypy app/
```

## Podman Commands

Build the image:
```bash
podman build -t moe-ollama-endpoint -f Containerfile .
```

Run standalone:
```bash
podman run -d \
  -p 8000:8000 \
  --env-file .env \
  --name moe-endpoint \
  moe-ollama-endpoint
```

With Podman Compose:
```bash
podman-compose up -d
podman-compose logs -f
podman-compose down
```

## Production Deployment

1. **Security**: 
   - Use secrets management for API keys
   - Configure CORS appropriately
   - Enable HTTPS/TLS
   - Use authentication middleware

2. **Scaling**:
   - Run multiple app instances behind a load balancer
   - Scale PostgreSQL with read replicas
   - Use connection pooling

3. **Monitoring**:
   - Add Prometheus metrics
   - Set up logging aggregation
   - Configure health check alerts

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
podman ps | grep postgres

# Check logs
podman-compose logs postgres

# Test connection
PGPASSWORD=moe_password psql -h localhost -U moe_user -d moe_rag
```

### Ollama API Issues
- Verify `OLLAMA_API_KEY` is set correctly
- Check `OLLAMA_BASE_URL` is accessible
- Review application logs: `podman-compose logs app`

### RAG Not Working
- Ensure pgvector extension is enabled
- Check documents are ingested: `GET /v1/rag/search?query=test`
- Verify embedding model is available

## License

See [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.ai/)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [pgvector](https://github.com/pgvector/pgvector)
