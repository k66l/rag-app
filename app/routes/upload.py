from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from app import storage
from app.config import settings
from app.utils.logger import log_function_call, get_logger
from app.middleware.auth_middleware import get_current_user
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from datetime import datetime, timezone

# Create router for document upload endpoints
router = APIRouter()

# Get logger for this module
logger = get_logger(__name__)


class UploadResponse(BaseModel):
    """Response model for successful upload"""
    status: str
    filename: str
    pages_processed: int
    chunks_created: int
    embedding_model: str
    user_email: str


class ErrorResponse(BaseModel):
    """Response model for upload errors"""
    error: str
    details: str = ""


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload and Index a PDF Document",
    description="Upload a PDF file that will be processed and indexed for question answering",
    response_description="Returns status and processing information",
    responses={
        201: {"model": UploadResponse, "description": "Successfully uploaded and indexed"},
        400: {"model": ErrorResponse, "description": "Invalid file or upload error"},
        401: {"description": "Authentication required"},
        500: {"model": ErrorResponse, "description": "Internal processing error"}
    }
)
@log_function_call(log_args=True, log_performance=True, log_result=False)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload and index")
) -> UploadResponse:
    """
    Upload and process a PDF document for question answering.

    This endpoint implements the RAG flow:
    PDF Upload → Text Extraction → Chunking → Embedding → Vector Store (FAISS)

    Steps:
    1. Validates and saves the uploaded PDF file
    2. Extracts text content from each page
    3. Splits content into optimized chunks for retrieval
    4. Creates vector embeddings (using local or OpenAI models)
    5. Stores embeddings in FAISS vector database for fast similarity search

    The processed document becomes immediately available for question answering
    through the /ask endpoint.

    Args:
        request (Request): FastAPI request object (contains authenticated user)
        file (UploadFile): PDF file to process. Must be valid PDF format.

    Returns:
        UploadResponse: Processing statistics and status information

    Raises:
        HTTPException: If file is invalid, not PDF, or processing fails
    """
    # Get authenticated user info
    user = get_current_user(request)
    user_email = user.get("email", "unknown")

    logger.info("Starting PDF upload and processing", extra={'extra_fields': {
        "filename": file.filename,
        "content_type": file.content_type,
        "user_email": user_email,
        "event_type": "upload_start"
    }})

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning("Invalid file type uploaded", extra={'extra_fields': {
            "filename": file.filename,
            "content_type": file.content_type,
            "user_email": user_email,
            "event_type": "invalid_file_type"
        }})
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    temp_file_path = None
    try:
        # Create temporary file for PDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info("PDF file saved to temporary location", extra={'extra_fields': {
            "filename": file.filename,
            "temp_path": temp_file_path,
            "file_size": len(content),
            "user_email": user_email,
            "event_type": "file_saved"
        }})

        # Extract text from PDF using LangChain
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        logger.info("PDF text extracted", extra={'extra_fields': {
            "filename": file.filename,
            "pages_extracted": len(documents),
            "user_email": user_email,
            "event_type": "text_extracted"
        }})

        # Split documents into chunks for optimal retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_documents = text_splitter.split_documents(documents)

        logger.info("Documents split into chunks", extra={'extra_fields': {
            "filename": file.filename,
            "chunks_created": len(split_documents),
            "user_email": user_email,
            "event_type": "documents_split"
        }})

        # Prepare texts and metadata for vector store
        texts = [doc.page_content for doc in split_documents]
        metadatas = []
        for i, doc in enumerate(split_documents):
            metadata = doc.metadata.copy()
            metadata.update({
                "filename": file.filename,
                "chunk_index": i,
                "total_chunks": len(split_documents),
                "uploaded_by": user_email,
                "upload_timestamp": str(datetime.now(timezone.utc))
            })
            metadatas.append(metadata)

        # Process documents using our RAG service
        logger.info("Creating vector embeddings and storing in FAISS", extra={'extra_fields': {
            "filename": file.filename,
            "chunks_to_embed": len(texts),
            "embedding_model": settings.EMBEDDING_MODEL,
            "user_email": user_email,
            "event_type": "embedding_start"
        }})

        storage.rag_service.process_documents(texts, metadatas)

        logger.info("PDF processing completed successfully", extra={'extra_fields': {
            "filename": file.filename,
            "pages_processed": len(documents),
            "chunks_created": len(split_documents),
            "embedding_model": settings.EMBEDDING_MODEL,
            "user_email": user_email,
            "event_type": "upload_completed"
        }})

        return UploadResponse(
            status="Successfully uploaded and indexed",
            filename=file.filename,
            pages_processed=len(documents),
            chunks_created=len(split_documents),
            embedding_model=settings.EMBEDDING_MODEL,
            user_email=user_email
        )

    except Exception as e:
        # Clean up temporary file in case of error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

        logger.error("PDF processing failed", extra={'extra_fields': {
            "filename": file.filename if file else "unknown",
            "user_email": user_email,
            "error": str(e),
            "error_type": type(e).__name__,
            "event_type": "upload_failed"
        }}, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF file: {str(e)}"
        )

    finally:
        # Always clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
