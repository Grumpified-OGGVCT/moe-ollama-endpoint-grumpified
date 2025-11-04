"""DSPy-based routing service for Mixture of Experts."""
import dspy
from typing import List, Dict, Any
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
    """Mixture of Experts router using DSPy with backup strategy."""
    
    # Routing keyword configurations
    CODE_KEYWORDS = ["code", "class", "programming", "debug", "implement", 
                     "script", "bug", "error", "compile", "syntax", "refactor", "test"]
    SIMPLE_CODE_KEYWORDS = ["simple", "basic", "quick", "small"]
    MATH_TOOL_KEYWORDS = ["math", "calculate", "equation", "solve", "tool", "function call",
                          "agent", "autonomous", "workflow", "integrate", "api", "invoke"]
    REASONING_KEYWORDS = ["analyze", "reasoning", "why", "complex", "detailed", "explain in depth",
                         "comprehensive", "thorough", "trace", "step-by-step", "think"]
    ENTERPRISE_KEYWORDS = ["enterprise", "production", "critical", "important", "detailed analysis"]
    AGGREGATION_KEYWORDS = ["summarize", "combine", "aggregate", "synthesize", "merge", "consolidate"]
    RAG_KEYWORDS = ["search", "find", "lookup", "retrieve", "document", "knowledge base"]
    
    def __init__(self):
        """Initialize the MoE router with Ollama Cloud models."""
        # Primary models
        self.default_model = settings.default_model
        self.reasoning_model = settings.reasoning_model
        self.fallback_model = settings.fallback_model
        self.enterprise_model = settings.enterprise_model
        self.math_tool_model = settings.math_tool_model
        self.code_model = settings.code_model
        self.aggregator_model = settings.aggregator_model
        self.cost_code_model = settings.cost_code_model
        self.vision_model = settings.vision_model
        self.vision_thinking_model = settings.vision_thinking_model
        
        # Backup strategy mapping: primary -> [first_backup, second_backup]
        self.backup_chain = {
            # deepseek-v3.1:671b-cloud (complex reasoning)
            self.reasoning_model: [self.enterprise_model, self.fallback_model],
            # gpt-oss:20b-cloud (fallback)
            self.fallback_model: [self.enterprise_model],
            # gpt-oss:120b-cloud (enterprise)
            self.enterprise_model: [self.reasoning_model, self.fallback_model],
            # kimi-k2:1t-cloud (math/tool)
            self.math_tool_model: [self.aggregator_model, self.enterprise_model],
            # qwen3-coder:480b-cloud (code)
            self.code_model: [self.cost_code_model, self.fallback_model],
            # glm-4.6:cloud (aggregator)
            self.aggregator_model: [self.reasoning_model, self.enterprise_model],
            # minimax-m2:cloud (cost-coding)
            self.cost_code_model: [self.fallback_model],
            # qwen3-vl:235b-cloud (vision)
            self.vision_model: [self.vision_thinking_model, self.fallback_model],
            # qwen3-vl:235b-instruct-cloud (vision+thinking)
            self.vision_thinking_model: [self.vision_model, self.fallback_model],
        }
    
    async def route_request(
        self,
        messages: List[Dict[str, Any]],
        use_rag: bool = False,
    ) -> tuple[str, bool]:
        """
        Route request to appropriate Ollama Cloud model based on task type.
        
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
        
        lower_message = last_message.lower()
        
        # Priority 1: Vision tasks (multimodal)
        if has_images:
            # Check if complex reasoning with images is needed
            if any(keyword in lower_message for keyword in self.REASONING_KEYWORDS + self.MATH_TOOL_KEYWORDS):
                logger.info("Routing to vision+thinking model for multimodal reasoning")
                return self.vision_thinking_model, False
            else:
                logger.info("Routing to vision model for GUI/visual tasks")
                return self.vision_model, False
        
        # Priority 2: Code generation/debugging
        if any(keyword in lower_message for keyword in self.CODE_KEYWORDS):
            # Check if cost-effective coding is suitable
            if any(keyword in lower_message for keyword in self.SIMPLE_CODE_KEYWORDS):
                logger.info("Routing to cost-effective code model")
                return self.cost_code_model, False
            else:
                logger.info("Routing to advanced code model")
                return self.code_model, False
        
        # Priority 3: Math/tool-calling/agentic workflows
        if any(keyword in lower_message for keyword in self.MATH_TOOL_KEYWORDS):
            logger.info("Routing to math/tool-calling model")
            return self.math_tool_model, use_rag
        
        # Priority 4: Complex reasoning (long-form, audit trails)
        if any(keyword in lower_message for keyword in self.REASONING_KEYWORDS):
            logger.info("Routing to complex reasoning model")
            return self.reasoning_model, use_rag
        
        # Priority 5: Enterprise deep reasoning (multi-turn, production-grade)
        if any(keyword in lower_message for keyword in self.ENTERPRISE_KEYWORDS):
            logger.info("Routing to enterprise model")
            return self.enterprise_model, use_rag
        
        # Priority 6: Aggregation tasks (multi-expert synthesis)
        if any(keyword in lower_message for keyword in self.AGGREGATION_KEYWORDS):
            logger.info("Routing to aggregator model")
            return self.aggregator_model, use_rag
        
        # Priority 7: RAG-related queries
        if any(keyword in lower_message for keyword in self.RAG_KEYWORDS):
            logger.info("Routing to fallback model with RAG enabled")
            return self.fallback_model, True
        
        # Default: Low-latency fallback for generic queries
        logger.info("Routing to default fallback model")
        return self.fallback_model, use_rag
    
    def get_backup_models(self, primary_model: str) -> List[str]:
        """
        Get backup models for a given primary model.
        
        Args:
            primary_model: The primary model name
            
        Returns:
            List of backup model names (may be empty)
        """
        return self.backup_chain.get(primary_model, [])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models in the MoE."""
        return {
            "reasoning": self.reasoning_model,
            "fallback": self.fallback_model,
            "enterprise": self.enterprise_model,
            "math_tool": self.math_tool_model,
            "code": self.code_model,
            "aggregator": self.aggregator_model,
            "cost_code": self.cost_code_model,
            "vision": self.vision_model,
            "vision_thinking": self.vision_thinking_model,
            "default": self.default_model,
            "backup_strategy": self.backup_chain.copy()
        }


# Global router instance
moe_router = MoERouter()
