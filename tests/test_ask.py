import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


class TestAsk:
    """Test suite for question answering endpoint."""

    def test_ask_success(self, client: TestClient, auth_headers, mock_rag_service):
        """Test successful question answering."""
        question_data = {"question": "What is the main topic of the document?"}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "user_email" in data

        assert data["answer"] == "This is a test answer from the RAG service."
        assert data["confidence"] in ["low", "medium", "high"]
        assert data["user_email"] == "admin@example.com"

    def test_ask_no_authentication(self, client: TestClient, mock_rag_service):
        """Test ask failure without authentication."""
        question_data = {"question": "What is this about?"}

        response = client.post("/api/v1/ask", json=question_data)

        assert response.status_code == 401

    def test_ask_invalid_token(self, client: TestClient, mock_rag_service):
        """Test ask failure with invalid authentication token."""
        question_data = {"question": "What is this about?"}
        headers = {"Authorization": "Bearer invalid-token"}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=headers)

        assert response.status_code == 401

    def test_ask_no_documents(self, client: TestClient, auth_headers, mock_empty_rag_service):
        """Test ask failure when no documents are uploaded."""
        question_data = {"question": "What is this about?"}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "No documents have been uploaded" in data["detail"]

    def test_ask_empty_question(self, client: TestClient, auth_headers, mock_rag_service):
        """Test ask failure with empty question."""
        question_data = {"question": ""}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_question_too_short(self, client: TestClient, auth_headers, mock_rag_service):
        """Test ask failure with question too short."""
        question_data = {"question": "Hi"}  # Less than 3 characters

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_question_too_long(self, client: TestClient, auth_headers, mock_rag_service):
        """Test ask failure with question too long."""
        long_question = "What is this about? " * 50  # Exceeds 500 characters
        question_data = {"question": long_question}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_no_question_field(self, client: TestClient, auth_headers, mock_rag_service):
        """Test ask failure when question field is missing."""
        question_data = {"query": "What is this about?"}  # Wrong field name

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_no_body(self, client: TestClient, auth_headers, mock_rag_service):
        """Test ask failure with no request body."""
        response = client.post("/api/v1/ask", headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_ask_rag_service_error(self, client: TestClient, auth_headers):
        """Test ask failure when RAG service throws an error."""
        question_data = {"question": "What is this about?"}

        with patch('app.storage.rag_service') as mock_service:
            mock_service.get_stats.return_value = {"total_documents": 5}
            mock_service.generate_answer.side_effect = Exception(
                "RAG service error")

            response = client.post(
                "/api/v1/ask", json=question_data, headers=auth_headers)

            assert response.status_code == 500

            data = response.json()
            assert "detail" in data
            assert "Error generating answer" in data["detail"]

    def test_ask_confidence_levels(self, client: TestClient, auth_headers):
        """Test different confidence levels based on context."""
        question_data = {"question": "What is the main topic?"}

        # Test high confidence (3+ context chunks)
        with patch('app.storage.rag_service') as mock_service:
            mock_service.get_stats.return_value = {"total_documents": 5}
            mock_service.generate_answer.return_value = "Test answer"
            mock_service.retrieve_context.return_value = [
                {"content": "Context 1"}, {"content": "Context 2"},
                {"content": "Context 3"}, {"content": "Context 4"}
            ]

            response = client.post(
                "/api/v1/ask", json=question_data, headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["confidence"] == "high"

        # Test medium confidence (1-2 context chunks)
        with patch('app.storage.rag_service') as mock_service:
            mock_service.get_stats.return_value = {"total_documents": 5}
            mock_service.generate_answer.return_value = "Test answer"
            mock_service.retrieve_context.return_value = [
                {"content": "Context 1"}]

            response = client.post(
                "/api/v1/ask", json=question_data, headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["confidence"] == "medium"

        # Test low confidence (0 context chunks)
        with patch('app.storage.rag_service') as mock_service:
            mock_service.get_stats.return_value = {"total_documents": 5}
            mock_service.generate_answer.return_value = "Test answer"
            mock_service.retrieve_context.return_value = []

            response = client.post(
                "/api/v1/ask", json=question_data, headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["confidence"] == "low"

    def test_ask_response_schema(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that ask response matches expected schema."""
        question_data = {"question": "What is the main topic?"}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        required_fields = ["answer", "confidence", "user_email"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["confidence"], str)
        assert isinstance(data["user_email"], str)

        # Verify values
        assert data["confidence"] in ["low", "medium", "high"]
        assert len(data["answer"]) > 0

    def test_ask_question_whitespace_handling(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that questions with leading/trailing whitespace are handled correctly."""
        question_data = {"question": "  What is this about?  "}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 200
        # The question should be stripped of whitespace before processing

    def test_ask_special_characters(self, client: TestClient, auth_headers, mock_rag_service):
        """Test asking questions with special characters."""
        special_questions = [
            "What is this about? 🤔",
            "How does it work (specifically)?",
            "What's the main point [summary]?",
            "Can you explain this & that?",
            "What about section #3?"
        ]

        for question in special_questions:
            question_data = {"question": question}
            response = client.post(
                "/api/v1/ask", json=question_data, headers=auth_headers)
            assert response.status_code == 200

    def test_ask_content_type(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that ask endpoint accepts and returns JSON content type."""
        question_data = {"question": "What is this about?"}

        response = client.post(
            "/api/v1/ask",
            json=question_data,
            headers={**auth_headers, "Content-Type": "application/json"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_ask_multiple_questions(self, client: TestClient, auth_headers, mock_rag_service):
        """Test asking multiple different questions."""
        questions = [
            "What is the main topic?",
            "Who are the key people mentioned?",
            "What are the important dates?",
            "Can you summarize the document?",
            "What conclusions can be drawn?"
        ]

        for question in questions:
            question_data = {"question": question}
            response = client.post(
                "/api/v1/ask", json=question_data, headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert "answer" in data
            assert len(data["answer"]) > 0

    def test_ask_async_behavior(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that the ask endpoint properly handles async operations."""
        question_data = {"question": "What is this about?"}

        response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
