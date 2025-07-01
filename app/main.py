# FastAPI application setup
from fastapi import Request as FastAPIRequest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import health_check, ask, upload, login
from app.config import settings
from app.middleware.logging_middleware import RequestResponseLoggingMiddleware
from app.middleware.auth_middleware import JWTAuthMiddleware
from app.logging_config import setup_logging

# Set up logging configuration
setup_logging(
    log_level="DEBUG" if settings.DEBUG else "INFO",
    log_file="logs/api.log",
    # Use JSON format in production, human-readable in debug
    json_format=not settings.DEBUG,
    enable_console=True
)

# Create FastAPI application instance with comprehensive metadata
app = FastAPI(
    title="🤖 RAG Application API",
    description="""
## Advanced Document Question-Answering System

This is a **Retrieval-Augmented Generation (RAG)** application that allows you to:

📄 **Upload PDF documents** and have them automatically processed and indexed  
🤔 **Ask natural language questions** about your document content  
🚀 **Get AI-powered answers** based on your uploaded documents  
🔒 **Secure access** with JWT authentication  
📊 **Real-time streaming** responses for better user experience

### How It Works

1. **Document Processing**: Upload PDFs → Text extraction → Smart chunking → Vector embeddings → FAISS storage
2. **Question Answering**: Your question → Semantic search → Context retrieval → AI answer generation → Streaming response

### Authentication

All document operations require authentication. Use the `/api/v1/login` endpoint to get your access token.

### Embedding Models

The system supports both:
- 🌐 **OpenAI Embeddings** (requires API key)
- 🏠 **Local Embeddings** (sentence-transformers, no API required)

### Architecture

- **Frontend**: Interactive Swagger UI for testing
- **Backend**: FastAPI with async processing
- **Vector Database**: FAISS for fast similarity search  
- **Authentication**: JWT-based security
- **Logging**: Comprehensive request/response logging
- **Streaming**: Real-time response chunks via Server-Sent Events

Perfect for document analysis, research assistance, and knowledge extraction from your PDF library! 📚✨
    """,
    version="1.0.0",
    contact={
        "name": "RAG Application",
        "email": "support@rag-app.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": f"http://{settings.HOST}:{settings.PORT}",
            "description": "Development server"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add JWT Authentication Middleware
# This will automatically protect the specified endpoints
app.add_middleware(
    JWTAuthMiddleware,
    protected_paths=[
        "/api/v1/upload",
        "/api/v1/ask",
        "/api/v1/ask/stream"
    ]
)

# Add logging middleware
app.add_middleware(
    RequestResponseLoggingMiddleware,
    logger_name="api_requests",
    log_request_body=True,  # Safe to enable since we now handle multipart content properly
    log_response_body=True,
    max_body_size=1024 * 10,  # 10KB limit for body logging
    exclude_paths=["/docs", "/redoc", "/openapi.json", "/favicon.ico"],
    sensitive_headers=["authorization", "cookie", "x-api-key"]
)

# Include all API route modules
app.include_router(health_check.router, prefix="")
app.include_router(upload.router, prefix="/api/v1",
                   tags=["📄 Document Upload"])
app.include_router(ask.router, prefix="/api/v1", tags=["🤔 Question Answering"])
app.include_router(login.router, prefix="/api/v1", tags=["🔐 Authentication"])

# ---------------------------------------------------------------------------
# Custom exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def custom_request_validation_exception_handler(
    request: FastAPIRequest, exc: RequestValidationError
):
    """Handle validation errors and safely encode bytes payloads.

    FastAPI's default handler tries to decode `bytes` objects as UTF-8, which
    fails for binary data (e.g., PDF uploads).  We override this behaviour by
    providing a custom encoder that converts bytes to a safe placeholder.
    """

    safe_detail = jsonable_encoder(
        exc.errors(),
        custom_encoder={bytes: lambda o: f"[{len(o)} bytes]"},
    )

    return JSONResponse(status_code=422, content={"detail": safe_detail})
