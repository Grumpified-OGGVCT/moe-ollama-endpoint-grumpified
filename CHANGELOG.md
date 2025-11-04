# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-04

### Added
- Initial release of MoE Ollama Endpoint
- OpenAI-compatible API endpoints (chat completions, embeddings, models)
- FastAPI-based web server with async support
- DSPy-based intelligent routing for Mixture of Experts
- PostgreSQL with pgvector for RAG (Retrieval-Augmented Generation)
- Multi-modal support for vision models
- Tool/function calling schema support
- Containerization with Podman/Docker support
- Docker Compose and Podman Compose configurations
- Comprehensive documentation (README, API docs, development guide)
- Example scripts for common usage patterns
- Test suite with pytest
- GitHub Actions CI/CD workflows
- Health check endpoints
- Automatic model routing based on query type:
  - Vision model for image inputs
  - Code model for programming queries
  - Reasoning model for complex analysis
  - RAG-augmented responses for retrieval queries
- Environment-based configuration
- Makefile for common development tasks
- Quick start script for easy deployment

### Features
- **MoE Routing**: Automatically selects the best model for each query
- **RAG Support**: Ingest and search documents with vector similarity
- **Vision Support**: Handle multi-modal inputs with images
- **OpenAI Compatible**: Drop-in replacement for OpenAI API
- **Streaming**: Support for Server-Sent Events (SSE) streaming
- **Production Ready**: Full containerization and health checks

### Documentation
- Comprehensive README with quick start guide
- API reference with examples in multiple languages
- Development guide for contributors
- Contributing guidelines
- Example scripts demonstrating all features

### Technical Stack
- Python 3.11+
- FastAPI for web framework
- DSPy for intelligent routing
- PostgreSQL + pgvector for vector database
- SQLAlchemy for ORM
- Pydantic for data validation
- Ollama for model inference
- Uvicorn for ASGI server

[0.1.0]: https://github.com/Grumpified-OGGVCT/moe-ollama-endpoint-grumpified/releases/tag/v0.1.0
