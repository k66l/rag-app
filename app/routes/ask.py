from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app import storage
from app.schemas import AskRequest, AskResponse
from app.utils.logger import log_function_call, get_logger
from app.middleware.auth_middleware import get_current_user
from app.config import settings
import json
import asyncio
from typing import AsyncGenerator

# Create router for question answering endpoints
router = APIRouter()

# Get logger for this module
logger = get_logger(__name__)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a Question About Your Documents",
    description="Ask questions about the content of your uploaded PDF documents using AI",
    response_description="Returns an AI-generated answer based on document content",
    responses={
        200: {"model": AskResponse, "description": "Successfully generated answer"},
        400: {"description": "Invalid question format"},
        401: {"description": "Authentication required"},
        404: {"description": "No documents uploaded yet"},
        500: {"description": "Error generating answer"}
    }
)
@log_function_call(log_args=True, log_performance=True, log_result=False)
async def ask_question(request: Request, ask_request: AskRequest) -> AskResponse:
    """
    Ask a question about your uploaded documents using RAG.

    This endpoint implements the complete RAG flow:
    Query → Top-K Retrieval → Context Formatting → Prompt → AI Answer

    Steps:
    1. Validates the input question
    2. Retrieves most relevant document chunks using vector similarity
    3. Formats retrieved context into a structured prompt
    4. Generates AI answer based on the context
    5. Returns formatted response with confidence level

    The system uses semantic search to find the most relevant parts of your
    documents and provides answers grounded in your uploaded content.

    Args:
        request (Request): FastAPI request object (contains authenticated user)
        ask_request (AskRequest): Contains the question to ask

    Returns:
        AskResponse: AI-generated answer with confidence level

    Raises:
        HTTPException: If no documents uploaded or processing fails
    """
    # Get authenticated user info
    user = get_current_user(request)
    user_email = user.get("email", "unknown")

    question = ask_request.question.strip()

    logger.info("Processing question", extra={'extra_fields': {
        "question_length": len(question),
        "user_email": user_email,
        "event_type": "question_received"
    }})

    # Check if we have any documents
    stats = storage.rag_service.get_stats()
    if stats["total_documents"] == 0:
        logger.warning("No documents available for questioning", extra={'extra_fields': {
            "user_email": user_email,
            "event_type": "no_documents_warning"
        }})
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded yet. Please upload a PDF document first using the /upload endpoint."
        )

    try:
        # Use RAG service to generate answer
        logger.info("Starting RAG pipeline", extra={'extra_fields': {
            "total_documents": stats["total_documents"],
            "top_k": settings.TOP_K_RETRIEVAL,
            "user_email": user_email,
            "event_type": "rag_pipeline_start"
        }})

        answer = await storage.rag_service.generate_answer(question)

        logger.info("RAG pipeline completed", extra={'extra_fields': {
            "answer_length": len(answer),
            "user_email": user_email,
            "event_type": "rag_pipeline_completed"
        }})

        # Determine confidence level based on retrieval results
        context = storage.rag_service.retrieve_context(question)
        confidence = "high" if len(context) >= 3 else "medium" if len(
            context) >= 1 else "low"

        logger.info("Question answered successfully", extra={'extra_fields': {
            "confidence": confidence,
            "context_chunks": len(context),
            "user_email": user_email,
            "event_type": "question_answered"
        }})

        return AskResponse(
            answer=answer,
            confidence=confidence,
            user_email=user_email
        )

    except Exception as e:
        logger.error("Error in RAG pipeline", extra={'extra_fields': {
            "error": str(e),
            "error_type": type(e).__name__,
            "user_email": user_email,
            "event_type": "rag_pipeline_error"
        }}, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


@router.post(
    "/ask/stream",
    summary="Ask a Question with Streaming Response",
    description="""Ask questions with real-time streaming response implementing the complete RAG pipeline:
    
    **Complete RAG Flow:** PDF Upload → Chunking → Embedding → Vector Store (FAISS) → Top-K Retrieval → Prompt → GPT Call → Answer → UI Stream
    
    **Streaming Pipeline:**
    1. 🔍 **Document Retrieval**: Searches your uploaded documents using FAISS vector similarity
    2. 📄 **Context Analysis**: Analyzes and ranks the most relevant document sections  
    3. 🤖 **AI Generation**: Generates answers using Google AI Studio (Gemini) or OpenAI with fallback
    4. ✨ **Real-time Streaming**: Streams the answer as it's being generated for better UX
    
    **Stream Format**: Server-Sent Events (SSE) with JSON chunks containing:
    - `status` chunks: Pipeline progress updates with emojis
    - `answer_start` chunks: Signals when answer generation begins
    - `content` chunks: Real-time answer text as it's generated
    - `complete` chunks: Completion signal with metadata about the RAG pipeline
    - `error` chunks: Any errors that occur during processing
    
    **Progress Tracking**: Each chunk includes `progress` (0-100%) and `stage` information.
    """,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Streaming Server-Sent Events with RAG pipeline progress and answer",
            "content": {
                "text/event-stream": {
                    "example": """data: {"type": "status", "data": "🔍 Searching your documents...", "stage": "retrieval", "progress": 10}

data: {"type": "status", "data": "📄 Found 5 relevant sections. Analyzing...", "stage": "analysis", "progress": 30}

data: {"type": "answer_start", "data": "✨ Answer:", "stage": "streaming", "progress": 60}

data: {"type": "content", "data": "The main topic of this document is...", "provider": "google", "stage": "streaming", "progress": 75}

data: {"type": "complete", "data": "✅ Answer complete", "provider": "google", "stage": "complete", "progress": 100, "metadata": {"context_chunks": 5, "llm_provider": "google"}}"""
                }
            }
        },
        400: {"description": "Invalid question format"},
        401: {"description": "Authentication required"},
        404: {"description": "No documents uploaded yet"},
        500: {"description": "Error generating answer"}
    }
)
@log_function_call(log_args=True, log_performance=True, log_result=False)
async def ask_question_stream(request: Request, ask_request: AskRequest):
    """
    Ask a question with streaming response for better user experience.

    This endpoint provides the same RAG functionality as /ask but returns
    the response as a server-sent event stream, allowing for real-time
    updates and better perceived performance.

    The stream format follows Server-Sent Events (SSE) specification:
    - data: JSON chunks with answer content
    - event: progress updates and completion signals

    Args:
        request (Request): FastAPI request object (contains authenticated user)
        ask_request (AskRequest): Contains the question to ask

    Returns:
        StreamingResponse: SSE stream with answer chunks

    Raises:
        HTTPException: If no documents uploaded or processing fails
    """
    # Get authenticated user info
    user = get_current_user(request)
    user_email = user.get("email", "unknown")

    question = ask_request.question.strip()

    logger.info("Processing streaming question", extra={'extra_fields': {
        "question_length": len(question),
        "user_email": user_email,
        "event_type": "streaming_question_received"
    }})

    # Check if we have any documents
    stats = storage.rag_service.get_stats()
    if stats["total_documents"] == 0:
        logger.warning("No documents available for streaming question", extra={'extra_fields': {
            "user_email": user_email,
            "event_type": "no_documents_warning"
        }})
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded yet. Please upload a PDF document first."
        )

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            logger.info("Starting streaming RAG pipeline", extra={'extra_fields': {
                "user_email": user_email,
                "event_type": "streaming_rag_start"
            }})

            # Use the enhanced streaming RAG pipeline
            async for chunk in storage.rag_service.generate_streaming_answer(question):
                # Format as Server-Sent Events (SSE)
                event_data = {
                    "type": chunk["type"],
                    "data": chunk["data"],
                    "stage": chunk.get("stage", "unknown"),
                    "progress": chunk.get("progress", 0)
                }

                # Add additional metadata if available
                if "provider" in chunk:
                    event_data["provider"] = chunk["provider"]
                if "metadata" in chunk:
                    event_data["metadata"] = chunk["metadata"]

                # Format as SSE
                sse_data = f"data: {json.dumps(event_data)}\n\n"

                logger.debug("Sending SSE chunk", extra={'extra_fields': {
                    "chunk_type": chunk["type"],
                    "chunk_size": len(chunk["data"]),
                    "stage": chunk.get("stage"),
                    "user_email": user_email,
                    "event_type": "sse_chunk_sent"
                }})

                yield sse_data

                # Small delay for better perceived streaming
                await asyncio.sleep(0.05)

                # Break after completion
                if chunk["type"] == "complete" or chunk["type"] == "error":
                    break

            logger.info("Streaming RAG pipeline completed successfully", extra={'extra_fields': {
                "user_email": user_email,
                "event_type": "streaming_rag_completed"
            }})

        except Exception as e:
            logger.error("Error in streaming RAG pipeline", extra={'extra_fields': {
                "error": str(e),
                "error_type": type(e).__name__,
                "user_email": user_email,
                "event_type": "streaming_rag_error"
            }}, exc_info=True)

            # Send error as SSE
            error_data = {
                "type": "error",
                "data": f"Error generating streaming answer: {str(e)}",
                "stage": "stream_error",
                "progress": 100
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )
