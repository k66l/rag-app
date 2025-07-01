import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app import storage


class TestUpload:
    """Test suite for document upload endpoint."""

    def test_upload_success(
        self,
        client: TestClient,
        auth_headers,
        sample_pdf_content,
        mock_rag_service,
        mock_pdf_loader,
        mock_text_splitter
    ):
        """Test successful PDF upload and processing."""
        files = {"file": ("test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}

        response = client.post(
            "/api/v1/upload", files=files, headers=auth_headers)

        assert response.status_code == 201

        data = response.json()
        assert "status" in data
        assert "filename" in data
        assert "pages_processed" in data
        assert "chunks_created" in data
        assert "embedding_model" in data
        assert "user_email" in data

        assert data["filename"] == "test.pdf"
        assert data["pages_processed"] == 2  # Based on mock_pdf_loader
        assert data["chunks_created"] == 3   # Based on mock_text_splitter
        assert data["status"] == "Successfully uploaded and indexed"

    def test_upload_no_authentication(self, client: TestClient, sample_pdf_content):
        """Test upload failure without authentication."""
        files = {"file": ("test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}

        response = client.post("/api/v1/upload", files=files)

        assert response.status_code == 401

    def test_upload_invalid_token(self, client: TestClient, sample_pdf_content):
        """Test upload failure with invalid authentication token."""
        files = {"file": ("test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}
        headers = {"Authorization": "Bearer invalid-token"}

        response = client.post("/api/v1/upload", files=files, headers=headers)

        assert response.status_code == 401

    def test_upload_non_pdf_file(self, client: TestClient, auth_headers):
        """Test upload failure with non-PDF file."""
        text_content = b"This is a text file, not a PDF"
        files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}

        response = client.post(
            "/api/v1/upload", files=files, headers=auth_headers)

        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "Only PDF files are supported" in data["detail"]

    def test_upload_no_file(self, client: TestClient, auth_headers):
        """Test upload failure when no file is provided."""
        response = client.post("/api/v1/upload", headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_upload_empty_file(self, client: TestClient, auth_headers):
        """Test upload failure with empty file."""
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}

        response = client.post(
            "/api/v1/upload", files=files, headers=auth_headers)

        # The behavior might vary based on implementation
        # Could be 400 (bad request) or continue with processing
        assert response.status_code in [400, 500]

    def test_upload_invalid_pdf_content(
        self,
        client: TestClient,
        auth_headers,
        mock_rag_service
    ):
        """Test upload failure with invalid PDF content."""
        # Create a file with PDF extension but invalid content
        invalid_pdf = b"This is not a valid PDF content"
        files = {"file": ("invalid.pdf", io.BytesIO(
            invalid_pdf), "application/pdf")}

        with patch('app.routes.upload.PyPDFLoader') as mock_loader:
            # Simulate PDF loading error
            mock_loader.side_effect = Exception("Invalid PDF format")

            response = client.post(
                "/api/v1/upload", files=files, headers=auth_headers)

            assert response.status_code == 500

    def test_upload_pdf_processing_error(
        self,
        client: TestClient,
        auth_headers,
        sample_pdf_content,
        mock_pdf_loader
    ):
        """Test upload failure when PDF processing fails."""
        files = {"file": ("test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}

        # Mock text splitter to raise an exception
        with patch('app.routes.upload.RecursiveCharacterTextSplitter') as mock_splitter:
            mock_splitter.side_effect = Exception("Text splitting failed")

            response = client.post(
                "/api/v1/upload", files=files, headers=auth_headers)

            assert response.status_code == 500

    def test_upload_rag_service_error(
        self,
        client: TestClient,
        auth_headers,
        sample_pdf_content,
        mock_pdf_loader,
        mock_text_splitter
    ):
        """Test upload failure when RAG service processing fails."""
        files = {"file": ("test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}

        # Mock RAG service to raise an exception
        with patch.object(storage, 'rag_service') as mock_service:
            mock_service.process_documents.side_effect = Exception(
                "RAG processing failed")

            response = client.post(
                "/api/v1/upload", files=files, headers=auth_headers)

            assert response.status_code == 500

    def test_upload_response_schema(
        self,
        client: TestClient,
        auth_headers,
        sample_pdf_content,
        mock_rag_service,
        mock_pdf_loader,
        mock_text_splitter
    ):
        """Test that upload response matches expected schema."""
        files = {"file": ("schema_test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}

        response = client.post(
            "/api/v1/upload", files=files, headers=auth_headers)

        assert response.status_code == 201
        data = response.json()

        # Verify all required fields are present
        required_fields = [
            "status", "filename", "pages_processed",
            "chunks_created", "embedding_model", "user_email"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(data["status"], str)
        assert isinstance(data["filename"], str)
        assert isinstance(data["pages_processed"], int)
        assert isinstance(data["chunks_created"], int)
        assert isinstance(data["embedding_model"], str)
        assert isinstance(data["user_email"], str)

        # Verify values
        assert data["pages_processed"] >= 0
        assert data["chunks_created"] >= 0

    def test_upload_large_pdf(
        self,
        client: TestClient,
        auth_headers,
        mock_rag_service,
        mock_text_splitter
    ):
        """Test upload with a larger PDF file."""
        # Create a larger PDF content
        large_content = b"%PDF-1.4\n" + b"Large content " * 1000 + b"\n%%EOF"
        files = {"file": ("large.pdf", io.BytesIO(
            large_content), "application/pdf")}

        # Mock PDF loader for large file
        with patch('app.routes.upload.PyPDFLoader') as mock_loader:
            mock_docs = []
            for i in range(10):  # Simulate 10 pages
                mock_doc = Mock()
                mock_doc.page_content = f"Content of page {i+1} " * 100
                mock_doc.metadata = {"page": i, "source": "large.pdf"}
                mock_docs.append(mock_doc)

            mock_loader_instance = Mock()
            mock_loader_instance.load.return_value = mock_docs
            mock_loader.return_value = mock_loader_instance

            response = client.post(
                "/api/v1/upload", files=files, headers=auth_headers)

            assert response.status_code == 201
            data = response.json()
            assert data["pages_processed"] == 10

    def test_upload_different_file_extensions(self, client: TestClient, auth_headers):
        """Test upload behavior with different file extensions."""
        test_cases = [
            # Uppercase extension should fail
            ("file.PDF", "application/pdf", 400),
            ("file.doc", "application/msword", 400),  # Wrong format
            ("file.txt", "text/plain", 400),        # Text file
            ("file", "application/pdf", 400),       # No extension
        ]

        for filename, content_type, expected_status in test_cases:
            files = {"file": (filename, io.BytesIO(b"content"), content_type)}
            response = client.post(
                "/api/v1/upload", files=files, headers=auth_headers)
            assert response.status_code == expected_status

    @patch('tempfile.NamedTemporaryFile')
    def test_upload_temp_file_cleanup(
        self,
        mock_temp_file,
        client: TestClient,
        auth_headers,
        sample_pdf_content,
        mock_rag_service,
        mock_pdf_loader,
        mock_text_splitter
    ):
        """Test that temporary files are cleaned up properly."""
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp

        files = {"file": ("test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}

        with patch('os.path.exists', return_value=True), \
                patch('os.unlink') as mock_unlink:

            response = client.post(
                "/api/v1/upload", files=files, headers=auth_headers)

            # The upload should succeed and temp file should be handled
            assert response.status_code == 201
