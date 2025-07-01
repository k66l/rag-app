from fastapi import UploadFile, File
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List


class AskRequest(BaseModel):
    """
    Schema for incoming question requests.

    This model defines the structure for questions that users can ask
    about their uploaded documents.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question you want to ask about your uploaded documents",
        json_schema_extra={
            "example": "What are the main topics covered in this document?"}
    )


class LoginRequest(BaseModel):
    """
    Schema for user authentication requests.

    This model defines the structure for login credentials that users
    provide to authenticate with the API.
    """
    email: EmailStr = Field(
        ...,
        description="User's email address for authentication",
        json_schema_extra={"example": "admin@example.com"}
    )
    password: str = Field(
        ...,
        min_length=6,
        description="User's password for authentication",
        json_schema_extra={"example": "admin123"}
    )


class LoginResponse(BaseModel):
    """
    Schema for authentication responses.

    This model defines the structure of responses returned after
    successful user authentication.
    """
    access_token: str = Field(
        ...,
        description="JWT access token for API authentication",
        json_schema_extra={
            "example": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}
    )
    token_type: str = Field(
        default="bearer",
        description="Type of the access token",
        json_schema_extra={"example": "bearer"}
    )
    expires_in: int = Field(
        ...,
        description="Token expiration time in seconds",
        json_schema_extra={"example": 86400}
    )
    user_email: str = Field(
        ...,
        description="Email of the authenticated user",
        json_schema_extra={"example": "admin@example.com"}
    )


class RAGMetadata(BaseModel):
    """
    Schema for RAG pipeline metadata information.

    Contains details about the RAG processing pipeline.
    """
    context_chunks: int = Field(
        ...,
        description="Number of relevant document chunks used",
        json_schema_extra={"example": 5}
    )
    query_length: int = Field(
        ...,
        description="Length of the user's query",
        json_schema_extra={"example": 42}
    )
    retrieval_model: str = Field(
        ...,
        description="Model used for document retrieval",
        json_schema_extra={"example": "FAISS + Local Embeddings"}
    )
    llm_provider: str = Field(
        ...,
        description="LLM provider used for answer generation",
        json_schema_extra={"example": "google"}
    )


class AskResponse(BaseModel):
    """
    Schema for question answering responses.

    This model defines the structure of responses returned when
    users ask questions about their documents.
    """
    answer: str = Field(
        ...,
        description="AI-generated answer based on document content",
        json_schema_extra={
            "example": "The main topics covered include artificial intelligence, machine learning, and data science."}
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level of the answer (low, medium, high)",
        json_schema_extra={"example": "high"}
    )
    user_email: str = Field(
        ...,
        description="Email of the user who asked the question",
        json_schema_extra={"example": "admin@example.com"}
    )


class StreamingChunk(BaseModel):
    """
    Schema for individual streaming response chunks.

    This model defines the structure of each chunk in the streaming response
    from the RAG pipeline.
    """
    type: str = Field(
        ...,
        description="Type of chunk (status, content, answer_start, complete, error)",
        json_schema_extra={"example": "content"}
    )
    data: str = Field(
        ...,
        description="The actual content or message of this chunk",
        json_schema_extra={"example": "The main concept is..."}
    )
    stage: str = Field(
        ...,
        description="Current stage of the RAG pipeline",
        json_schema_extra={"example": "streaming"}
    )
    progress: int = Field(
        ...,
        description="Progress percentage (0-100)",
        json_schema_extra={"example": 75}
    )
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider used (google, openai, local)",
        json_schema_extra={"example": "google"}
    )
    metadata: Optional[RAGMetadata] = Field(
        default=None,
        description="Additional metadata (included in completion chunk)"
    )


class StreamingResponse(BaseModel):
    """
    Schema documenting the complete streaming response format.

    This is used for API documentation to show the expected SSE format.
    """
    message: str = Field(
        ...,
        description="Information about the streaming response format",
        json_schema_extra={
            "example": "Streaming response follows Server-Sent Events (SSE) format with JSON data chunks"}
    )
    format: str = Field(
        default="text/event-stream",
        description="Response format",
        json_schema_extra={"example": "text/event-stream"}
    )
    chunk_types: List[str] = Field(
        default=["status", "answer_start", "content", "complete", "error"],
        description="Possible chunk types in the stream"
    )
    stages: List[str] = Field(
        default=["retrieval", "analysis",
                 "generation", "streaming", "complete"],
        description="Pipeline stages that will be reported"
    )


class UploadResponse(BaseModel):
    """
    Schema for document upload responses.

    This model defines the structure of responses returned after
    successfully uploading and processing a PDF document.
    """
    status: str = Field(
        ...,
        description="Status message indicating processing result",
        json_schema_extra={"example": "Successfully uploaded and indexed"}
    )
    filename: str = Field(
        ...,
        description="Name of the uploaded file",
        json_schema_extra={"example": "document.pdf"}
    )
    pages_processed: int = Field(
        ...,
        description="Number of pages processed from the PDF",
        json_schema_extra={"example": 25}
    )
    chunks_created: int = Field(
        ...,
        description="Number of text chunks created for indexing",
        json_schema_extra={"example": 87}
    )
    embedding_model: str = Field(
        ...,
        description="Embedding model used for processing",
        json_schema_extra={"example": "local"}
    )
    user_email: str = Field(
        ...,
        description="Email of the user who uploaded the document",
        json_schema_extra={"example": "admin@example.com"}
    )


class HealthResponse(BaseModel):
    """
    Schema for health check responses.

    This model defines the structure of responses returned by
    the health check endpoint.
    """
    message: str = Field(
        ...,
        description="Welcome or status message",
        json_schema_extra={"example": "Welcome to the RAG Application API!"}
    )
    status: str = Field(
        ...,
        description="Current health status of the service",
        json_schema_extra={"example": "healthy"}
    )


class ErrorResponse(BaseModel):
    """
    Schema for error responses across all endpoints.

    This model provides a consistent structure for error messages
    returned by the API when something goes wrong.
    """
    error: str = Field(
        ...,
        description="Error type or category",
        json_schema_extra={"example": "ValidationError"}
    )
    detail: str = Field(
        ...,
        description="Detailed error message explaining what went wrong",
        json_schema_extra={
            "example": "The uploaded file is not a valid PDF format"}
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="When the error occurred (ISO format)",
        json_schema_extra={"example": "2024-01-15T10:30:00Z"}
    )
