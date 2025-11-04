"""DSPy-based routing service for Mixture of Experts."""
import dspy
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.ollama_client import ollama_service
import logging

logger = logging.getLogger(__name__)


class OllamaLM(dspy.LM):
    """Custom DSPy Language Model wrapper for Ollama."""
    
    def __init__(self, model: str):
        """Initialize Ollama LM."""
        super().__init__(model)
        self.model = model
        self.history = []
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Synchronous call wrapper for compatibility.
        Note: This is a simple wrapper. For production use with DSPy,
        implement proper async handling or use DSPy's async capabilities.
        """
        import asyncio
        try:
            # Try to get or create event loop for sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use run_until_complete
                logger.warning("Cannot make sync call in async context, use acall instead")
                return ""
            return loop.run_until_complete(self.acall([{"role": "user", "content": prompt}], **kwargs))
        except RuntimeError:
            # No event loop available, create one
            return asyncio.run(self.acall([{"role": "user", "content": prompt}], **kwargs))
    
    async def acall(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Async call to Ollama."""
        response = await ollama_service.generate_completion(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        content = response.get("message", {}).get("content", "")
        return content


class RouterSignature(dspy.Signature):
    """Signature for routing decisions."""
    query = dspy.InputField(desc="User query to route")
    task_type = dspy.OutputField(desc="Task type: general, vision, code, reasoning, or rag")
    reasoning = dspy.OutputField(desc="Brief explanation for the routing decision")


class MoERouter:
    """Mixture of Experts router using DSPy."""
    
    def __init__(self):
        """Initialize the MoE router."""
        self.default_model = settings.default_model
        self.vision_model = settings.vision_model
        self.code_model = settings.code_model
        self.reasoning_model = settings.reasoning_model
    
    async def route_request(
        self,
        messages: List[Dict[str, Any]],
        use_rag: bool = False,
    ) -> tuple[str, bool]:
        """
        Route request to appropriate model.
        
        Returns:
            tuple of (model_name, use_rag_flag)
        """
        # Extract last user message for routing decision
        last_message = ""
        has_images = False
        
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_message = content
                elif isinstance(content, list):
                    # Multi-modal content
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                last_message = item.get("text", "")
                            elif item.get("type") == "image_url":
                                has_images = True
                break
        
        # Rule-based routing with simple heuristics
        if has_images:
            logger.info("Routing to vision model due to image content")
            return self.vision_model, False
        
        # Check for code-related keywords
        code_keywords = ["code", "function", "class", "programming", "debug", "implement", "script"]
        if any(keyword in last_message.lower() for keyword in code_keywords):
            logger.info("Routing to code model")
            return self.code_model, False
        
        # Check for reasoning-related keywords
        reasoning_keywords = ["analyze", "explain", "reasoning", "why", "complex", "detailed"]
        if any(keyword in last_message.lower() for keyword in reasoning_keywords):
            logger.info("Routing to reasoning model")
            return self.reasoning_model, use_rag
        
        # Check for RAG-related keywords
        rag_keywords = ["search", "find", "lookup", "retrieve", "document"]
        if any(keyword in last_message.lower() for keyword in rag_keywords):
            logger.info("Routing to default model with RAG enabled")
            return self.default_model, True
        
        # Default routing
        logger.info("Routing to default model")
        return self.default_model, use_rag
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about available models in the MoE."""
        return {
            "default": self.default_model,
            "vision": self.vision_model,
            "code": self.code_model,
            "reasoning": self.reasoning_model,
        }


# Global router instance
moe_router = MoERouter()
