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
    
    # MoE Configuration
    default_model: str = "llama3.1:8b"
    vision_model: str = "llava:13b"
    code_model: str = "codellama:13b"
    reasoning_model: str = "llama3.1:70b"
    
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
