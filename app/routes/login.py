from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.schemas import LoginRequest
from app.config import settings
from app.utils.logger import log_function_call, get_logger
from app.middleware.auth_middleware import create_jwt_token
from datetime import datetime, timezone

# Create router for authentication endpoints
router = APIRouter()

# Get logger for this module
logger = get_logger(__name__)


class LoginResponse(BaseModel):
    """Response model for successful login"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until expiration
    user_email: str


class ErrorResponse(BaseModel):
    """Response model for login errors"""
    error: str
    details: str = ""


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=201,
    summary="User Authentication",
    description="Authenticate user credentials and receive a JWT access token",
    response_description="Returns JWT token for API access",
    responses={
        201: {"model": LoginResponse, "description": "Successfully authenticated"},
        400: {"model": ErrorResponse, "description": "Invalid credentials"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    }
)
@log_function_call(log_args=True, log_performance=True, log_result=False)
async def login(request: LoginRequest) -> LoginResponse:
    """
    Authenticate user and return JWT access token.

    This endpoint validates user credentials and returns a JWT token
    that must be included in the Authorization header for protected endpoints.

    Authentication Flow:
    1. Validates email format and password requirements
    2. Checks credentials against configured admin account
    3. Generates JWT token with user information and expiration
    4. Returns token with metadata for client use

    The returned JWT token should be used in subsequent API calls:
    - Header: `Authorization: Bearer <token>`
    - Token expires in 24 hours by default (configurable)

    Args:
        request (LoginRequest): User credentials (email and password)

    Returns:
        LoginResponse: JWT access token and metadata

    Raises:
        HTTPException: If credentials are invalid or authentication fails
    """

    logger.info("Login attempt", extra={'extra_fields': {
        "email": request.email,
        "event_type": "login_attempt"
    }})

    # Validate credentials against admin account
    # In a real application, this would check against a user database
    if (request.email != settings.ADMIN_EMAIL or
            request.password != settings.ADMIN_PASSWORD):

        logger.warning("Invalid login credentials", extra={'extra_fields': {
            "email": request.email,
            "event_type": "login_failed"
        }})

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or password"
        )

    try:
        # Create JWT token
        access_token = create_jwt_token(
            email=request.email,
            user_id=request.email  # Using email as user ID for simplicity
        )

        # Calculate expiration time in seconds
        expires_in = settings.JWT_EXPIRATION_HOURS * 3600

        logger.info("Login successful", extra={'extra_fields': {
            "email": request.email,
            "token_expires_in_hours": settings.JWT_EXPIRATION_HOURS,
            "event_type": "login_successful"
        }})

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            user_email=request.email
        )

    except Exception as e:
        logger.error("Error creating JWT token", extra={'extra_fields': {
            "email": request.email,
            "error": str(e),
            "error_type": type(e).__name__,
            "event_type": "jwt_creation_error"
        }}, exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )
