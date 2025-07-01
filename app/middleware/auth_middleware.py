from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import jwt
from datetime import datetime, timezone
from typing import Optional, List
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Security scheme for Swagger UI
security = HTTPBearer()


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT Authentication Middleware

    This middleware automatically validates JWT tokens for protected routes.
    It checks the Authorization header and validates the token signature and expiration.
    """

    def __init__(self, app, protected_paths: List[str] = None):
        super().__init__(app)
        # Default protected paths - can be configured
        self.protected_paths = protected_paths or [
            "/api/v1/upload",
            "/api/v1/ask",
            "/api/v1/ask/stream"
        ]

    async def dispatch(self, request: Request, call_next):
        """
        Process each request and validate JWT if needed
        """
        # Check if this path requires authentication
        path = request.url.path
        needs_auth = any(path.startswith(protected_path)
                         for protected_path in self.protected_paths)

        if not needs_auth:
            # Skip authentication for non-protected routes
            return await call_next(request)

        # Extract and validate JWT token
        try:
            token = self._extract_token(request)
            if not token:
                return self._unauthorized_response("Missing or invalid authorization header")

            # Validate the token
            payload = self._validate_token(token)

            # Add user info to request state for use in route handlers
            request.state.user = {
                "email": payload.get("email"),
                "sub": payload.get("sub"),
                "exp": payload.get("exp"),
                "iat": payload.get("iat")
            }

            logger.debug("JWT authentication successful", extra={'extra_fields': {
                "user_email": payload.get("email"),
                "path": path,
                "event_type": "jwt_auth_success"
            }})

        except HTTPException as e:
            logger.warning("JWT authentication failed", extra={'extra_fields': {
                "path": path,
                "error": str(e.detail),
                "status_code": e.status_code,
                "event_type": "jwt_auth_failed"
            }})
            return self._unauthorized_response(str(e.detail))

        except Exception as e:
            logger.error("Unexpected error in JWT middleware", extra={'extra_fields': {
                "path": path,
                "error": str(e),
                "error_type": type(e).__name__,
                "event_type": "jwt_middleware_error"
            }}, exc_info=True)
            return self._unauthorized_response("Authentication error")

        # Continue to the route handler
        response = await call_next(request)
        return response

    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from Authorization header
        Supports both 'Bearer <token>' and '<token>' formats
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        # Handle 'Bearer <token>' format
        if auth_header.startswith("Bearer "):
            return auth_header.split(" ", 1)[1]

        # Handle direct token format
        return auth_header

    def _validate_token(self, token: str) -> dict:
        """
        Validate JWT token and return payload
        """
        try:
            # Decode and validate the token
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM]
            )

            # Check if token is expired
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                exp_datetime = datetime.fromtimestamp(
                    exp_timestamp, tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_datetime:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has expired"
                    )

            # Validate required fields
            required_fields = ["email", "sub"]
            for field in required_fields:
                if field not in payload:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Token missing required field: {field}"
                    )

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
        except Exception as e:
            logger.error("Token validation error", extra={'extra_fields': {
                "error": str(e),
                "error_type": type(e).__name__,
                "event_type": "token_validation_error"
            }})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed"
            )

    def _unauthorized_response(self, detail: str) -> Response:
        """
        Return a standardized 401 Unauthorized response
        """
        return Response(
            content=f'{{"detail": "{detail}"}}',
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"Content-Type": "application/json"}
        )


def get_current_user(request: Request) -> dict:
    """
    Utility function to get current user from request state
    Can be used in route handlers to access authenticated user info
    """
    if not hasattr(request.state, 'user'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated"
        )
    return request.state.user


async def verify_token(credentials: HTTPAuthorizationCredentials = security) -> dict:
    """
    Dependency function for manual JWT verification in route handlers
    Can be used as a FastAPI dependency for routes that need user info
    """
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # Check expiration
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            exp_datetime = datetime.fromtimestamp(
                exp_timestamp, tz=timezone.utc)
            if datetime.now(timezone.utc) > exp_datetime:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def create_jwt_token(email: str, user_id: str = None) -> str:
    """
    Create a new JWT token for a user
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    exp = now + timedelta(hours=settings.JWT_EXPIRATION_HOURS)

    payload = {
        "email": email,
        "sub": user_id or email,  # Subject (user identifier)
        "iat": int(now.timestamp()),  # Issued at
        "exp": int(exp.timestamp())   # Expiration
    }

    token = jwt.encode(payload, settings.JWT_SECRET,
                       algorithm=settings.JWT_ALGORITHM)

    logger.info("JWT token created", extra={'extra_fields': {
        "user_email": email,
        "expires_at": exp.isoformat(),
        "event_type": "jwt_token_created"
    }})

    return token
