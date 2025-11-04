"""Embeddings routes."""
from fastapi import APIRouter, HTTPException
from app.models.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingData, Usage
from app.services.ollama_client import ollama_service
from app.core.config import settings
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings (OpenAI-compatible).
    """
    try:
        # Handle both single string and list of strings
        texts = [request.input] if isinstance(request.input, str) else request.input
        
        # Use embedding model from settings or request
        model = request.model if request.model else settings.embedding_model
        
        # Generate embeddings
        embeddings = await ollama_service.generate_embeddings(model=model, text=texts)
        
        # Format response
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(
                EmbeddingData(
                    embedding=embedding,
                    index=i,
                )
            )
        
        # Calculate approximate token usage
        total_tokens = sum(len(text) // 4 for text in texts)
        
        return EmbeddingResponse(
            data=embedding_data,
            model=model,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )
    
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
