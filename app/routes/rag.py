"""RAG-specific routes for document ingestion and management."""
from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGIngestRequest, RAGIngestResponse
from app.services.rag import rag_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rag/ingest", response_model=RAGIngestResponse)
async def ingest_documents(request: RAGIngestRequest):
    """
    Ingest documents into the RAG system.
    """
    try:
        documents = [doc.model_dump() for doc in request.documents]
        count = await rag_service.ingest_documents(documents)
        
        return RAGIngestResponse(
            success=True,
            documents_ingested=count,
            message=f"Successfully ingested {count} documents",
        )
    
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while ingesting documents. Please try again later."
        )


@router.get("/rag/search")
async def search_documents(query: str, collection: str = None, top_k: int = 5):
    """
    Search for similar documents in the RAG system.
    """
    try:
        results = await rag_service.search_similar(query, collection, top_k)
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching documents. Please try again later."
        )
