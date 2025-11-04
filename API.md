# API Reference

## Base URL

```
http://localhost:8000/v1
```

## Authentication

The Ollama API key is configured server-side via environment variables. Client applications do not need to provide authentication for the endpoint itself.

## Endpoints

### Health & Status

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

#### GET /

Root endpoint with API information.

**Response:**
```json
{
  "name": "MoE Ollama Endpoint",
  "version": "0.1.0",
  "description": "OpenAI-compatible API with MoE routing, RAG, vision, and tool support"
}
```

---

### Chat Completions

#### POST /v1/chat/completions

Create a chat completion (OpenAI-compatible).

**Request Body:**
```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "use_rag": false,
  "rag_collections": ["default"]
}
```

**Parameters:**
- `model` (string, required): Model to use. Use "auto" for automatic routing.
- `messages` (array, required): Array of message objects with `role` and `content`.
- `temperature` (number, optional): 0.0 to 2.0, default 0.7
- `max_tokens` (number, optional): Maximum tokens to generate
- `stream` (boolean, optional): Enable streaming responses
- `use_rag` (boolean, optional): Enable RAG augmentation
- `rag_collections` (array, optional): Collections to search in RAG

**Response:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3.1:8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**Vision Support:**

For multi-modal requests with images:

```json
{
  "model": "auto",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }
  ]
}
```

**Streaming:**

Set `stream: true` for SSE streaming:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

---

### Models

#### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.1:8b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "ollama-moe"
    },
    {
      "id": "llava:13b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "ollama-moe"
    }
  ]
}
```

#### GET /v1/models/{model_id}

Get information about a specific model.

**Response:**
```json
{
  "id": "llama3.1:8b",
  "object": "model",
  "created": 1234567890,
  "owned_by": "ollama"
}
```

---

### Embeddings

#### POST /v1/embeddings

Generate embeddings for text.

**Request Body:**
```json
{
  "model": "nomic-embed-text",
  "input": "The quick brown fox jumps over the lazy dog"
}
```

**Parameters:**
- `model` (string, required): Embedding model to use
- `input` (string or array, required): Text(s) to embed
- `encoding_format` (string, optional): "float" or "base64"

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "nomic-embed-text",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 0,
    "total_tokens": 10
  }
}
```

---

### RAG (Retrieval-Augmented Generation)

#### POST /v1/rag/ingest

Ingest documents into the RAG system.

**Request Body:**
```json
{
  "documents": [
    {
      "content": "This is the document content...",
      "metadata": {"source": "manual", "author": "John Doe"},
      "collection": "knowledge_base"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "documents_ingested": 1,
  "message": "Successfully ingested 1 documents"
}
```

#### GET /v1/rag/search

Search for similar documents.

**Query Parameters:**
- `query` (string, required): Search query
- `collection` (string, optional): Collection to search in
- `top_k` (number, optional): Number of results (default: 5)

**Example:**
```bash
curl "http://localhost:8000/v1/rag/search?query=machine%20learning&top_k=3"
```

**Response:**
```json
{
  "results": [
    {
      "content": "Machine learning is a subset of AI...",
      "metadata": {"source": "textbook", "chapter": 1},
      "collection": "knowledge_base",
      "similarity_score": 0.95
    }
  ]
}
```

---

## MoE Routing

The endpoint uses intelligent routing to select the best model:

| Query Type | Model Used | Trigger Keywords |
|------------|------------|------------------|
| **Vision** | llava:13b | Images in content |
| **Code** | codellama:13b | code, function, class, programming |
| **Reasoning** | llama3.1:70b | analyze, explain, reasoning, complex |
| **RAG** | llama3.1:8b + RAG | search, find, lookup, retrieve |
| **Default** | llama3.1:8b | All other queries |

You can override automatic routing by specifying a model explicitly.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad request (validation error)
- `404`: Not found (model doesn't exist)
- `500`: Internal server error

---

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider adding rate limiting middleware.

---

## Examples

### Python (OpenAI SDK)

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)

# Embeddings
response = client.embeddings.create(
    model="nomic-embed-text",
    input="Hello, world!"
)

print(response.data[0].embedding)
```

### cURL

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# List models
curl http://localhost:8000/v1/models

# Create embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Test text"
  }'
```

### JavaScript (fetch)

```javascript
// Chat completion
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'auto',
    messages: [
      {role: 'user', content: 'What is JavaScript?'}
    ]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

---

## Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation where you can test all endpoints directly in your browser.
