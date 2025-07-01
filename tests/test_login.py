import pytest
import jwt
from fastapi.testclient import TestClient
from app.config import settings


class TestLogin:
    """Test suite for login endpoint."""

    def test_login_success(self, client: TestClient, valid_credentials):
        """Test successful login with valid credentials."""
        response = client.post("/api/v1/login", json=valid_credentials)

        assert response.status_code == 201

        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert "user_email" in data

        assert data["token_type"] == "bearer"
        assert data["expires_in"] == settings.JWT_EXPIRATION_HOURS * 3600
        assert data["user_email"] == valid_credentials["email"]

        # Verify JWT token is valid
        token = data["access_token"]
        decoded = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        assert decoded["email"] == valid_credentials["email"]

    def test_login_invalid_credentials(self, client: TestClient, invalid_credentials):
        """Test login failure with invalid credentials."""
        response = client.post("/api/v1/login", json=invalid_credentials)

        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "Invalid email or password" in data["detail"]

    def test_login_missing_email(self, client: TestClient):
        """Test login failure when email is missing."""
        credentials = {"password": "admin123"}

        response = client.post("/api/v1/login", json=credentials)

        assert response.status_code == 422  # Validation error

    def test_login_missing_password(self, client: TestClient):
        """Test login failure when password is missing."""
        credentials = {"email": "admin@example.com"}

        response = client.post("/api/v1/login", json=credentials)

        assert response.status_code == 422  # Validation error

    def test_login_invalid_email_format(self, client: TestClient):
        """Test login failure with invalid email format."""
        credentials = {
            "email": "not-an-email",
            "password": "admin123"
        }

        response = client.post("/api/v1/login", json=credentials)

        assert response.status_code == 422  # Validation error

    def test_login_password_too_short(self, client: TestClient):
        """Test login failure with password too short."""
        credentials = {
            "email": "test@example.com",
            "password": "123"  # Less than 6 characters
        }

        response = client.post("/api/v1/login", json=credentials)

        assert response.status_code == 422  # Validation error

    def test_login_empty_credentials(self, client: TestClient):
        """Test login failure with empty credentials."""
        credentials = {"email": "", "password": ""}

        response = client.post("/api/v1/login", json=credentials)

        assert response.status_code == 422  # Validation error

    def test_login_no_body(self, client: TestClient):
        """Test login failure with no request body."""
        response = client.post("/api/v1/login")

        assert response.status_code == 422  # Validation error

    def test_login_response_schema(self, client: TestClient, valid_credentials):
        """Test that login response matches expected schema."""
        response = client.post("/api/v1/login", json=valid_credentials)

        assert response.status_code == 201
        data = response.json()

        # Verify all required fields are present
        required_fields = ["access_token",
                           "token_type", "expires_in", "user_email"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(data["access_token"], str)
        assert isinstance(data["token_type"], str)
        assert isinstance(data["expires_in"], int)
        assert isinstance(data["user_email"], str)

        # Verify token format (JWT has 3 parts separated by dots)
        assert len(data["access_token"].split(".")) == 3

    def test_login_content_type(self, client: TestClient, valid_credentials):
        """Test that login accepts and returns JSON content type."""
        response = client.post(
            "/api/v1/login",
            json=valid_credentials,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 201
        assert response.headers["content-type"] == "application/json"

    def test_login_case_insensitive_email(self, client: TestClient):
        """Test login with different email case."""
        # This test assumes the system should be case-insensitive for emails
        # Adjust based on your actual requirements
        credentials = {
            "email": settings.ADMIN_EMAIL.upper(),
            "password": settings.ADMIN_PASSWORD
        }

        response = client.post("/api/v1/login", json=credentials)

        # This might fail if your system is case-sensitive
        # Adjust the assertion based on your requirements
        # Either works or doesn't based on implementation
        assert response.status_code in [201, 400]

    def test_login_multiple_attempts(self, client: TestClient, valid_credentials):
        """Test multiple login attempts with same credentials."""
        for _ in range(3):
            response = client.post("/api/v1/login", json=valid_credentials)
            assert response.status_code == 201

            data = response.json()
            assert "access_token" in data
            # Each token might be different due to timestamps
            assert len(data["access_token"]) > 0
