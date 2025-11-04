"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.core.config import settings
from app.routes import chat, models, embeddings, rag
from app.db.database import init_db
from app.services.ollama_client import ollama_service

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MoE Ollama Endpoint...")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. RAG features may not work.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MoE Ollama Endpoint...")
    await ollama_service.close()


# Create FastAPI app
app = FastAPI(
    title="MoE Ollama Endpoint",
    description="Production-grade OpenAI-compatible endpoint orchestrating Ollama Cloud models as a MoE system with RAG, vision, and tool support",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/v1", tags=["Chat"])
app.include_router(models.router, prefix="/v1", tags=["Models"])
app.include_router(embeddings.router, prefix="/v1", tags=["Embeddings"])
app.include_router(rag.router, prefix="/v1", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "MoE Ollama Endpoint",
        "version": "0.1.0",
        "description": "OpenAI-compatible API with MoE routing, RAG, vision, and tool support",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/health")
async def health_check_v1():
    """Health check endpoint (v1)."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
