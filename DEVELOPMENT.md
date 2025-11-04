# MoE Ollama Endpoint - Developer Guide

## Architecture Overview

### Components

1. **FastAPI Application** (`app/main.py`)
   - Main application entry point
   - Route registration
   - Middleware configuration
   - Lifecycle management

2. **Configuration** (`app/core/config.py`)
   - Environment-based settings
   - Pydantic settings management
   - Database URL construction

3. **Data Models** (`app/models/schemas.py`)
   - OpenAI-compatible request/response models
   - Pydantic validation
   - Type hints

4. **Database Layer** (`app/db/`)
   - PostgreSQL with pgvector
   - SQLAlchemy async ORM
   - Vector similarity search

5. **Services** (`app/services/`)
   - `ollama_client.py`: Ollama API integration
   - `router.py`: DSPy-based MoE routing
   - `rag.py`: RAG implementation

6. **API Routes** (`app/routes/`)
   - `chat.py`: Chat completions
   - `models.py`: Model management
   - `embeddings.py`: Embedding generation
   - `rag.py`: RAG endpoints

## Development Setup

### Local Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

3. Set up environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run PostgreSQL (for RAG):
```bash
podman run -d \
  --name moe-postgres \
  -e POSTGRES_USER=moe_user \
  -e POSTGRES_PASSWORD=moe_password \
  -e POSTGRES_DB=moe_rag \
  -p 5432:5432 \
  ankane/pgvector:latest
```

5. Run the application:
```bash
python -m app.main
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
ruff check app/ tests/

# Type checking
mypy app/

# All quality checks
black app/ tests/ && ruff check app/ tests/ && mypy app/
```

## Adding New Features

### Adding a New Model Type

1. Update `app/core/config.py`:
```python
class Settings(BaseSettings):
    # ... existing settings ...
    translation_model: str = "llama3.1:13b-translate"
```

2. Update `app/services/router.py`:
```python
class MoERouter:
    def __init__(self):
        # ... existing init ...
        self.translation_model = settings.translation_model
    
    async def route_request(self, messages, use_rag=False):
        # Add routing logic
        translation_keywords = ["translate", "translation"]
        if any(kw in last_message.lower() for kw in translation_keywords):
            return self.translation_model, False
```

### Adding a New API Endpoint

1. Create route file `app/routes/new_feature.py`:
```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/new-feature")
async def new_feature():
    return {"message": "New feature"}
```

2. Register in `app/main.py`:
```python
from app.routes import new_feature

app.include_router(new_feature.router, prefix="/v1", tags=["NewFeature"])
```

### Extending RAG Capabilities

1. Add new methods to `app/services/rag.py`:
```python
async def custom_search(self, query: str, filters: Dict[str, Any]):
    # Implement custom search logic
    pass
```

2. Create new endpoint in `app/routes/rag.py`:
```python
@router.get("/rag/custom-search")
async def custom_search(query: str):
    results = await rag_service.custom_search(query, {})
    return {"results": results}
```

## Performance Optimization

### Caching

Add caching for frequently accessed data:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_cached_embedding(text: str):
    return await ollama_service.generate_embeddings(text)
```

### Connection Pooling

Already configured in `app/db/database.py` via SQLAlchemy.

### Async Operations

All I/O operations use async/await for non-blocking execution.

## Monitoring and Logging

### Structured Logging

```python
import logging
import json

logger = logging.getLogger(__name__)

# Use structured logs
logger.info(
    "Request processed",
    extra={
        "model": model,
        "tokens": total_tokens,
        "latency_ms": latency
    }
)
```

### Metrics (Future Enhancement)

Consider adding Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total API requests')
request_latency = Histogram('api_request_latency_seconds', 'Request latency')
```

## Deployment Best Practices

### Environment Variables

- Never commit `.env` files
- Use secrets management in production
- Validate all required variables on startup

### Database Migrations

For schema changes, use Alembic:

```bash
# Install Alembic
pip install alembic

# Initialize
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add new column"

# Apply migration
alembic upgrade head
```

### Security

1. **API Authentication**: Add middleware for API key validation
2. **Rate Limiting**: Implement rate limiting per client
3. **Input Validation**: All inputs validated via Pydantic
4. **SQL Injection**: Prevented by SQLAlchemy ORM

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL is running
   - Verify connection string
   - Ensure pgvector extension is installed

2. **Ollama API Errors**
   - Verify API key is valid
   - Check network connectivity
   - Review Ollama service status

3. **Import Errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m app.main
```

## Contributing Guidelines

1. **Code Style**
   - Follow PEP 8
   - Use type hints
   - Write docstrings

2. **Testing**
   - Write tests for new features
   - Maintain test coverage > 80%
   - Test edge cases

3. **Documentation**
   - Update README for user-facing changes
   - Update this guide for developer changes
   - Add docstrings to all functions

4. **Pull Requests**
   - Create feature branch
   - Write descriptive commit messages
   - Request review before merging

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
