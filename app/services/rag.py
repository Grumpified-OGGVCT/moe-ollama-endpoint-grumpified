"""RAG service for retrieval-augmented generation."""
from typing import List, Dict, Any, Optional
from sqlalchemy import select
from app.db.models import Document
from app.db.database import get_db_session
from app.services.ollama_client import ollama_service
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.embedding_model = settings.embedding_model
        self.top_k = settings.top_k_results
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Ingest documents into the vector database.
        
        Args:
            documents: List of documents with 'content', 'metadata', and 'collection'
        
        Returns:
            Number of documents ingested
        """
        ingested_count = 0
        
        async with get_db_session() as session:
            for doc in documents:
                try:
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    collection = doc.get("collection", "default")
                    
                    # Generate embedding
                    embeddings = await ollama_service.generate_embeddings(
                        model=self.embedding_model,
                        text=content
                    )
                    
                    if embeddings and len(embeddings) > 0:
                        embedding = embeddings[0]
                        
                        # Create document record
                        db_doc = Document(
                            content=content,
                            embedding=embedding,
                            doc_metadata=metadata,
                            collection=collection,
                        )
                        
                        session.add(db_doc)
                        ingested_count += 1
                except Exception as e:
                    logger.error(f"Error ingesting document: {e}")
                    continue
            
            await session.commit()
        
        logger.info(f"Ingested {ingested_count} documents")
        return ingested_count
    
    async def search_similar(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query
            collection: Optional collection filter
            top_k: Number of results to return
        
        Returns:
            List of similar documents with content and metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        try:
            # Generate query embedding
            embeddings = await ollama_service.generate_embeddings(
                model=self.embedding_model,
                text=query
            )
            
            if not embeddings or len(embeddings) == 0:
                logger.warning("Failed to generate query embedding")
                return []
            
            query_embedding = embeddings[0]
            
            async with get_db_session() as session:
                # Build query with error handling for vector operations
                try:
                    stmt = select(
                        Document.content,
                        Document.doc_metadata,
                        Document.collection,
                        Document.embedding.cosine_distance(query_embedding).label("distance")
                    )
                except Exception as e:
                    logger.error(f"Vector operation failed, pgvector may not be available: {e}")
                    return []
                
                if collection:
                    stmt = stmt.where(Document.collection == collection)
                
                stmt = stmt.order_by("distance").limit(top_k)
                
                result = await session.execute(stmt)
                rows = result.all()
                
                # Format results
                documents = []
                for row in rows:
                    documents.append({
                        "content": row.content,
                        "metadata": row.doc_metadata,
                        "collection": row.collection,
                        "similarity_score": 1 - row.distance,  # Convert distance to similarity
                    })
                
                return documents
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    async def augment_prompt(
        self,
        query: str,
        collection: Optional[str] = None,
    ) -> str:
        """
        Augment a query with relevant context from RAG.
        
        Args:
            query: Original query
            collection: Optional collection filter
        
        Returns:
            Augmented prompt with context
        """
        documents = await self.search_similar(query, collection)
        
        if not documents:
            return query
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}]\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Construct augmented prompt
        augmented_prompt = f"""Based on the following relevant documents, please answer the question.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the context provided."""
        
        return augmented_prompt


# Global RAG service instance
rag_service = RAGService()
