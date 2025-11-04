"""Database models for vector storage."""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.core.config import settings

Base = declarative_base()


class Document(Base):
    """Document table with vector embeddings."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.vector_dimension))  # Use configurable dimension
    doc_metadata = Column(JSON, default={})  # Renamed from 'metadata' to avoid conflict
    collection = Column(String(255), index=True, default="default")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
