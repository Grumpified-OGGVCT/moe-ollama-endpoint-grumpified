"""Pydantic models for OpenAI-compatible API."""
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class ToolDefinition(BaseModel):
    """Tool definition."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # RAG-specific parameters
    use_rag: Optional[bool] = False
    rag_collections: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = "float"


class EmbeddingData(BaseModel):
    """Embedding data."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ollama"


class ModelListResponse(BaseModel):
    """Model list response."""
    object: str = "list"
    data: List[ModelInfo]


class RAGDocument(BaseModel):
    """Document for RAG ingestion."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    collection: Optional[str] = "default"


class RAGIngestRequest(BaseModel):
    """RAG document ingestion request."""
    documents: List[RAGDocument]


class RAGIngestResponse(BaseModel):
    """RAG document ingestion response."""
    success: bool
    documents_ingested: int
    message: str
