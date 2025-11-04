"""Configuration settings for the MoE Ollama endpoint."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    
    # Ollama Configuration
    ollama_base_url: str = "https://api.ollama.cloud"
    ollama_api_key: str = ""
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "moe_user"
    postgres_password: str = "moe_password"
    postgres_db: str = "moe_rag"
    
    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    
    # MoE Configuration - Ollama Cloud Models (Nov 4, 2025)
    # Text Models
    reasoning_model: str = "deepseek-v3.1:671b-cloud"  # Complex reasoning with thinking mode
    fallback_model: str = "gpt-oss:20b-cloud"  # Low-latency fallback
    enterprise_model: str = "gpt-oss:120b-cloud"  # Deep multi-turn reasoning
    math_tool_model: str = "kimi-k2:1t-cloud"  # Math/tool-calling/agentic
    code_model: str = "qwen3-coder:480b-cloud"  # Code generation/debugging
    aggregator_model: str = "glm-4.6:cloud"  # Aggregation with tool-use
    cost_code_model: str = "minimax-m2:cloud"  # Cost-effective coding
    
    # Vision Models
    vision_model: str = "qwen3-vl:235b-cloud"  # Visual agent for GUI/multimodal
    vision_thinking_model: str = "qwen3-vl:235b-instruct-cloud"  # Multimodal reasoning with thinking
    
    # Legacy/Compatibility
    default_model: str = "gpt-oss:20b-cloud"  # Default fallback for generic queries
    
    # RAG Configuration
    embedding_model: str = "nomic-embed-text"
    vector_dimension: int = 768
    top_k_results: int = 5
    
    # DSPy Configuration
    dspy_cache_dir: str = ".dspy_cache"
    dspy_max_retries: int = 3
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL database URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


settings = Settings()
