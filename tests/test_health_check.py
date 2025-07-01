import pytest
from fastapi.testclient import TestClient


class TestHealthCheck:
    """Test suite for health check endpoint."""

    def test_health_check_success(self, client: TestClient):
        """Test successful health check response."""
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["message"] == "Welcome to the RAG Application API!"
        assert data["status"] == "healthy"

    def test_health_check_content_type(self, client: TestClient):
        """Test that health check returns JSON content type."""
        response = client.get("/")

        assert response.headers["content-type"] == "application/json"

    def test_health_check_no_authentication_required(self, client: TestClient):
        """Test that health check doesn't require authentication."""
        # Test without any headers
        response = client.get("/")
        assert response.status_code == 200

        # Test with invalid auth header (should still work)
        response = client.get(
            "/", headers={"Authorization": "Bearer invalid-token"})
        assert response.status_code == 200

    def test_health_check_response_schema(self, client: TestClient):
        """Test that health check response matches expected schema."""
        response = client.get("/")
        data = response.json()

        # Verify all required fields are present
        required_fields = ["message", "status"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(data["message"], str)
        assert isinstance(data["status"], str)

        # Verify status value
        assert data["status"] in ["healthy", "unhealthy"]

    def test_health_check_multiple_requests(self, client: TestClient):
        """Test that health check is consistent across multiple requests."""
        responses = []
        for _ in range(5):
            response = client.get("/")
            responses.append(response)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

        # All responses should be identical
        first_response_data = responses[0].json()
        for response in responses[1:]:
            assert response.json() == first_response_data
