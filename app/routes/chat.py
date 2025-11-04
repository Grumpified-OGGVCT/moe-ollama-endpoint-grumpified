"""Chat completion routes."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, Usage, Message
from app.services.ollama_client import ollama_service
from app.services.router import moe_router
from app.services.rag import rag_service
import time
import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible).
    """
    try:
        # Route to appropriate model
        model, use_rag = await moe_router.route_request(
            messages=[msg.model_dump() for msg in request.messages],
            use_rag=request.use_rag or False,
        )
        
        # Override with explicit model if provided
        if request.model and request.model != "auto":
            model = request.model
        
        # Prepare messages
        messages = [msg.model_dump() for msg in request.messages]
        
        # Apply RAG if enabled
        if use_rag or request.use_rag:
            # Get last user message
            last_user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        last_user_msg = content
                    break
            
            if last_user_msg:
                # Augment with RAG context
                augmented_prompt = await rag_service.augment_prompt(
                    last_user_msg,
                    collection=request.rag_collections[0] if request.rag_collections else None,
                )
                
                # Replace last user message with augmented version
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["content"] = augmented_prompt
                        break
        
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request, model, messages),
                media_type="text/event-stream"
            )
        
        # Generate completion
        response = await ollama_service.generate_completion(
            model=model,
            messages=messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            stream=False,
        )
        
        # Extract response content
        assistant_message = response.get("message", {})
        content = assistant_message.get("content", "")
        
        # Calculate token usage (approximate)
        prompt_tokens = sum(len(str(msg.get("content", ""))) // 4 for msg in messages)
        completion_tokens = len(content) // 4
        
        # Format response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        
        return completion_response
    
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completion(
    request: ChatCompletionRequest,
    model: str,
    messages: list,
) -> AsyncIterator[str]:
    """Stream chat completion responses."""
    try:
        async for chunk in ollama_service.generate_completion_stream(
            model=model,
            messages=messages,
            temperature=request.temperature or 0.7,
        ):
            if chunk:
                try:
                    data = json.loads(chunk)
                    message = data.get("message", {})
                    content = message.get("content", "")
                    
                    # Format as SSE
                    sse_data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    
                    yield f"data: {json.dumps(sse_data)}\n\n"
                    
                    if data.get("done", False):
                        # Send final chunk
                        final_data = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"
                        yield "data: [DONE]\n\n"
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
