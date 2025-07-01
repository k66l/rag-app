import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from app.main import app
from app.middleware.auth_middleware import create_jwt_token
from app.config import settings
from app import storage
import io


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def auth_token():
    """Create a valid JWT token for authentication tests."""
    return create_jwt_token(
        email=settings.ADMIN_EMAIL,
        user_id=settings.ADMIN_EMAIL
    )


@pytest.fixture
def auth_headers(auth_token):
    """Create authorization headers with valid JWT token."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def mock_rag_service():
    """Mock the RAG service to avoid external dependencies."""
    with patch.object(storage, 'rag_service') as mock_service:
        # Configure mock methods
        mock_service.get_stats.return_value = {
            "total_documents": 5,
            "total_chunks": 50,
            "embedding_model": "local"
        }
        mock_service.generate_answer.return_value = "This is a test answer from the RAG service."
        mock_service.retrieve_context.return_value = [
            {"content": "Context 1", "metadata": {"source": "doc1.pdf"}},
            {"content": "Context 2", "metadata": {"source": "doc2.pdf"}},
            {"content": "Context 3", "metadata": {"source": "doc3.pdf"}}
        ]
        mock_service.process_documents.return_value = None

        yield mock_service


@pytest.fixture
def sample_pdf_content():
    """Create a sample PDF content for upload tests."""
    # Create a simple PDF using reportlab
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "This is a test PDF document.")
        c.drawString(100, 730, "It contains sample text for testing purposes.")
        c.drawString(
            100, 710, "The RAG system should be able to process this content.")
        c.save()

        buffer.seek(0)
        return buffer.getvalue()
    except ImportError:
        # Fallback: create a minimal PDF-like content
        # This is not a real PDF but will work for basic testing
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n192\n%%EOF"


@pytest.fixture
def invalid_credentials():
    """Provide invalid login credentials for testing."""
    return {
        "email": "invalid@example.com",
        "password": "wrongpassword"
    }


@pytest.fixture
def valid_credentials():
    """Provide valid login credentials for testing."""
    return {
        "email": settings.ADMIN_EMAIL,
        "password": settings.ADMIN_PASSWORD
    }


@pytest.fixture
def mock_empty_rag_service():
    """Mock RAG service with no documents for testing edge cases."""
    with patch.object(storage, 'rag_service') as mock_service:
        mock_service.get_stats.return_value = {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_model": "local"
        }
        yield mock_service


@pytest.fixture
def mock_pdf_loader():
    """Mock PyPDFLoader to avoid file system dependencies."""
    with patch('app.routes.upload.PyPDFLoader') as mock_loader:
        # Create mock documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "This is page 1 content from the test PDF."
        mock_doc1.metadata = {"page": 0, "source": "test.pdf"}

        mock_doc2 = Mock()
        mock_doc2.page_content = "This is page 2 content from the test PDF."
        mock_doc2.metadata = {"page": 1, "source": "test.pdf"}

        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]
        mock_loader.return_value = mock_loader_instance

        yield mock_loader


@pytest.fixture
def mock_text_splitter():
    """Mock text splitter to control document chunking."""
    with patch('app.routes.upload.RecursiveCharacterTextSplitter') as mock_splitter:
        # Create mock split documents
        mock_chunk1 = Mock()
        mock_chunk1.page_content = "This is chunk 1 from the document."
        mock_chunk1.metadata = {"page": 0, "source": "test.pdf"}

        mock_chunk2 = Mock()
        mock_chunk2.page_content = "This is chunk 2 from the document."
        mock_chunk2.metadata = {"page": 0, "source": "test.pdf"}

        mock_chunk3 = Mock()
        mock_chunk3.page_content = "This is chunk 3 from the document."
        mock_chunk3.metadata = {"page": 1, "source": "test.pdf"}

        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = [
            mock_chunk1, mock_chunk2, mock_chunk3]
        mock_splitter.return_value = mock_splitter_instance

        yield mock_splitter


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically clean up any temporary files after each test."""
    yield
    # Cleanup logic can be added here if needed
    pass
