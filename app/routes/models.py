"""Model listing and management routes."""
from fastapi import APIRouter, HTTPException
from app.models.schemas import ModelListResponse, ModelInfo
from app.services.ollama_client import ollama_service
from app.services.router import moe_router
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """
    List available models (OpenAI-compatible).
    """
    try:
        # Get models from Ollama
        ollama_models = await ollama_service.list_models()
        
        # Get MoE configuration
        moe_config = moe_router.get_model_info()
        
        # Format models
        models = []
        seen = set()
        
        # Add MoE-configured models first
        for name, model_id in moe_config.items():
            if model_id not in seen:
                models.append(
                    ModelInfo(
                        id=model_id,
                        created=int(time.time()),
                        owned_by="ollama-moe",
                    )
                )
                seen.add(model_id)
        
        # Add other available models
        for model in ollama_models:
            model_id = model.get("name", "")
            if model_id and model_id not in seen:
                models.append(
                    ModelInfo(
                        id=model_id,
                        created=int(time.time()),
                        owned_by="ollama",
                    )
                )
                seen.add(model_id)
        
        return ModelListResponse(data=models)
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while listing models. Please try again later."
        )


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get information about a specific model.
    """
    try:
        # Check if model exists
        models = await ollama_service.list_models()
        model_names = [m.get("name", "") for m in models]
        
        if model_id not in model_names:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="ollama",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving model information. Please try again later."
        )
