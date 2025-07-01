import pytest
from fastapi.testclient import TestClient
import io


class TestIntegration:
    """Integration tests that test multiple endpoints working together."""

    def test_full_workflow(
        self,
        client: TestClient,
        sample_pdf_content,
        mock_rag_service,
        mock_pdf_loader,
        mock_text_splitter
    ):
        """Test the complete workflow: login -> upload -> ask."""

        # Step 1: Login
        login_data = {
            "email": "admin@example.com",
            "password": "admin123"
        }
        login_response = client.post("/api/v1/login", json=login_data)
        assert login_response.status_code == 201

        token = login_response.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}

        # Step 2: Upload a document
        files = {"file": ("test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}
        upload_response = client.post(
            "/api/v1/upload", files=files, headers=auth_headers)
        assert upload_response.status_code == 201

        # Step 3: Ask a question
        question_data = {"question": "What is this document about?"}
        ask_response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)
        assert ask_response.status_code == 200

        ask_data = ask_response.json()
        assert "answer" in ask_data
        assert "confidence" in ask_data
        assert ask_data["user_email"] == "admin@example.com"

    def test_authentication_flow(self, client: TestClient):
        """Test authentication is properly enforced across endpoints."""

        # Test that protected endpoints require authentication
        protected_endpoints = [
            ("/api/v1/upload", "POST"),
            ("/api/v1/ask", "POST"),
            ("/api/v1/ask/stream", "POST")
        ]

        for endpoint, method in protected_endpoints:
            if method == "POST":
                response = client.post(endpoint)
                assert response.status_code == 401, f"Endpoint {endpoint} should require auth"

    def test_error_handling_consistency(self, client: TestClient, auth_headers):
        """Test that error responses are consistent across endpoints."""

        # Test validation errors
        validation_test_cases = [
            ("/api/v1/login", {"email": "invalid", "password": "123"}),
            ("/api/v1/ask", {"question": ""}),
            ("/api/v1/ask/stream", {"question": ""})
        ]

        for endpoint, invalid_data in validation_test_cases:
            if endpoint == "/api/v1/login":
                response = client.post(endpoint, json=invalid_data)
            else:
                response = client.post(
                    endpoint, json=invalid_data, headers=auth_headers)

            assert response.status_code == 422
            assert "detail" in response.json()

    def test_content_type_consistency(self, client: TestClient, auth_headers, mock_rag_service):
        """Test that all endpoints return proper content types."""

        # Health check
        health_response = client.get("/")
        assert health_response.headers["content-type"] == "application/json"

        # Login
        login_data = {"email": "admin@example.com", "password": "admin123"}
        login_response = client.post("/api/v1/login", json=login_data)
        assert login_response.headers["content-type"] == "application/json"

        # Ask
        question_data = {"question": "What is this about?"}
        ask_response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)
        assert ask_response.headers["content-type"] == "application/json"

    def test_rate_limiting_simulation(self, client: TestClient, auth_headers, mock_rag_service):
        """Test multiple rapid requests to simulate rate limiting scenarios."""

        question_data = {"question": "What is this about?"}

        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.post(
                "/api/v1/ask", json=question_data, headers=auth_headers)
            responses.append(response)

        # All should succeed (no rate limiting implemented)
        for response in responses:
            assert response.status_code == 200

    def test_concurrent_uploads(
        self,
        client: TestClient,
        auth_headers,
        sample_pdf_content,
        mock_rag_service,
        mock_pdf_loader,
        mock_text_splitter
    ):
        """Test handling of concurrent upload requests."""

        # Simulate multiple uploads (in sequence due to TestClient limitations)
        for i in range(3):
            files = {"file": (f"test_{i}.pdf", io.BytesIO(
                sample_pdf_content), "application/pdf")}
            response = client.post(
                "/api/v1/upload", files=files, headers=auth_headers)
            assert response.status_code == 201

    def test_cross_endpoint_data_consistency(
        self,
        client: TestClient,
        auth_headers,
        sample_pdf_content,
        mock_rag_service,
        mock_pdf_loader,
        mock_text_splitter
    ):
        """Test that data remains consistent across different endpoints."""

        # Upload a document
        files = {"file": ("consistency_test.pdf", io.BytesIO(
            sample_pdf_content), "application/pdf")}
        upload_response = client.post(
            "/api/v1/upload", files=files, headers=auth_headers)
        assert upload_response.status_code == 201

        upload_data = upload_response.json()
        upload_user_email = upload_data["user_email"]

        # Ask a question and verify user consistency
        question_data = {"question": "What is this document about?"}
        ask_response = client.post(
            "/api/v1/ask", json=question_data, headers=auth_headers)
        assert ask_response.status_code == 200

        ask_data = ask_response.json()
        ask_user_email = ask_data["user_email"]

        # User email should be consistent
        assert upload_user_email == ask_user_email
