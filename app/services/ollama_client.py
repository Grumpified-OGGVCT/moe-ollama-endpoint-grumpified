"""Ollama client service for model interaction."""
import httpx
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for interacting with Ollama API."""
    
    def __init__(self):
        """Initialize Ollama client."""
        self.base_url = settings.ollama_base_url
        self.api_key = settings.ollama_api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=120.0,
        )
    
    async def generate_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate chat completion using Ollama."""
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            # Add any additional parameters
            payload.update(kwargs)
            
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    async def generate_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate streaming chat completion."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            payload.update(kwargs)
            
            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
        except Exception as e:
            logger.error(f"Error in streaming completion: {e}")
            raise
    
    async def generate_embeddings(self, model: str, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for text."""
        try:
            texts = [text] if isinstance(text, str) else text
            embeddings = []
            
            for txt in texts:
                payload = {"model": model, "prompt": txt}
                response = await self.client.post("/api/embeddings", json=payload)
                response.raise_for_status()
                result = response.json()
                embeddings.append(result.get("embedding", []))
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Global Ollama service instance
ollama_service = OllamaService()
