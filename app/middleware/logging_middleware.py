"""
Logging middleware for FastAPI applications.

This middleware automatically logs all incoming requests and outgoing responses
with detailed information including timing, status codes, and request/response data.
"""

import time
import json
import uuid
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from app.logging_config import get_logger


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log HTTP requests and responses.

    This middleware captures:
    - Request details (method, URL, headers, body)
    - Response details (status code, headers, body)
    - Processing time
    - Unique request ID for tracking
    - Client IP address
    - User agent information

    Features:
    - Configurable body logging (can be disabled for large payloads)
    - Sensitive data filtering (passwords, tokens, etc.)
    - Request correlation IDs
    - Performance monitoring
    """

    def __init__(
        self,
        app: ASGIApp,
        logger_name: str = "api_requests",
        log_request_body: bool = True,
        log_response_body: bool = True,
        max_body_size: int = 1024 * 1024,  # 1MB default
        exclude_paths: Optional[list] = None,
        sensitive_headers: Optional[list] = None
    ):
        """
        Initialize the logging middleware.

        Args:
            app: The ASGI application
            logger_name: Name of the logger to use
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            max_body_size: Maximum body size to log (in bytes)
            exclude_paths: List of paths to exclude from logging
            sensitive_headers: List of header names to filter out
        """
        super().__init__(app)
        self.logger = get_logger(logger_name)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.exclude_paths = exclude_paths or [
            "/docs", "/redoc", "/openapi.json", "/favicon.ico"]
        self.sensitive_headers = sensitive_headers or [
            "authorization", "cookie", "x-api-key", "x-auth-token"
        ]

    def _should_log_request(self, request: Request) -> bool:
        """
        Determine if a request should be logged.

        Args:
            request: The incoming request

        Returns:
            True if the request should be logged, False otherwise
        """
        path = request.url.path
        return not any(excluded in path for excluded in self.exclude_paths)

    def _is_binary_content(self, content_type: str) -> bool:
        """
        Check if content type indicates binary data.

        Args:
            content_type: The content type header value

        Returns:
            True if the content is likely binary
        """
        content_type = content_type.lower()
        binary_types = [
            "application/octet-stream",
            "application/pdf",
            "application/zip",
            "application/x-rar",
            "image/",
            "video/",
            "audio/",
            "multipart/form-data"
        ]
        return any(binary_type in content_type for binary_type in binary_types)

    def _filter_sensitive_data(self, headers: dict) -> dict:
        """
        Filter out sensitive information from headers.

        Args:
            headers: Dictionary of headers

        Returns:
            Filtered headers dictionary
        """
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                filtered[key] = "[FILTERED]"
            else:
                filtered[key] = value
        return filtered

    def _safe_json_parse(self, data: str) -> Any:
        """
        Safely parse JSON data.

        Args:
            data: String data to parse

        Returns:
            Parsed JSON or original string if parsing fails
        """
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    async def _get_request_body(self, request: Request) -> Optional[str]:
        """
        Extract request body if logging is enabled.

        Args:
            request: The incoming request

        Returns:
            Request body as string or None
        """
        if not self.log_request_body:
            return None

        try:
            # Check content type first to avoid consuming the body stream for multipart data
            content_type = request.headers.get("content-type", "").lower()

            # Skip reading body for multipart/form-data entirely to avoid interfering with FastAPI's parsing
            if "multipart/form-data" in content_type:
                return f"[MULTIPART FORM DATA: {content_type}]"

            # Skip other binary content types as well
            if self._is_binary_content(content_type):
                return f"[BINARY CONTENT: {content_type}]"

            body = await request.body()
            if len(body) > self.max_body_size:
                return f"[BODY TOO LARGE: {len(body)} bytes]"

            if body:
                # Try to decode as UTF-8
                try:
                    body_str = body.decode('utf-8')
                    # Try to parse as JSON for better formatting
                    return self._safe_json_parse(body_str)
                except UnicodeDecodeError:
                    # If UTF-8 decoding fails, it's likely binary content
                    return f"[BINARY CONTENT: {len(body)} bytes]"
            return None
        except Exception as e:
            self.logger.warning(f"Failed to read request body: {e}")
            return "[BODY READ ERROR]"

    async def _get_response_body(self, response_body: bytes) -> Optional[Any]:
        """
        Extract response body if logging is enabled.

        Args:
            response_body: The response body bytes

        Returns:
            Response body as parsed data or None
        """
        if not self.log_response_body:
            return None

        try:
            if len(response_body) > self.max_body_size:
                return f"[BODY TOO LARGE: {len(response_body)} bytes]"

            if response_body:
                # Try to decode as UTF-8
                try:
                    body_str = response_body.decode('utf-8')
                    return self._safe_json_parse(body_str)
                except UnicodeDecodeError:
                    # If UTF-8 decoding fails, it's likely binary content
                    return f"[BINARY CONTENT: {len(response_body)} bytes]"
            return None
        except Exception as e:
            self.logger.warning(f"Failed to read response body: {e}")
            return "[BODY READ ERROR]"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log details.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            The response from the application
        """
        # Skip logging for excluded paths
        if not self._should_log_request(request):
            return await call_next(request)

        # Generate unique request ID for correlation
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Extract client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Get request body
        request_body = await self._get_request_body(request)

        # Prepare request log data
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": self._filter_sensitive_data(dict(request.headers)),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "body": request_body
        }

        # Log the incoming request
        self.logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={'extra_fields': {
                "event_type": "request",
                "request_data": request_data
            }}
        )

        # Process the request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log any exceptions that occur during processing
            processing_time = time.time() - start_time

            self.logger.error(
                f"Request processing failed: {request.method} {request.url.path}",
                extra={'extra_fields': {
                    "event_type": "request_error",
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time_ms": round(processing_time * 1000, 2)
                }},
                exc_info=True
            )
            raise

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get response body for logging
        response_body = None
        if isinstance(response, StreamingResponse):
            # For streaming responses, we can't easily capture the body
            response_body = "[STREAMING RESPONSE]"
        else:
            # For regular responses, read the body
            if hasattr(response, 'body'):
                response_body = await self._get_response_body(response.body)

        # Prepare response log data
        response_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response_body,
            "processing_time_ms": round(processing_time * 1000, 2)
        }

        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
            message = f"Server error: {request.method} {request.url.path} -> {response.status_code}"
        elif response.status_code >= 400:
            log_level = logging.WARNING
            message = f"Client error: {request.method} {request.url.path} -> {response.status_code}"
        else:
            log_level = logging.INFO
            message = f"Request completed: {request.method} {request.url.path} -> {response.status_code}"

        # Log the response
        self.logger.log(
            log_level,
            message,
            extra={'extra_fields': {
                "event_type": "response",
                "response_data": response_data
            }}
        )

        # Add request ID to response headers for client tracking
        response.headers["X-Request-ID"] = request_id

        return response


def create_logging_middleware(
    logger_name: str = "api_requests",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    **middleware_kwargs
) -> RequestResponseLoggingMiddleware:
    """
    Factory function to create a configured logging middleware.

    This function sets up logging configuration and returns a middleware instance
    that can be easily added to a FastAPI application.

    Args:
        logger_name: Name of the logger to use
        log_level: Minimum log level for the logger
        log_file: Optional file to write logs to
        json_format: Whether to use JSON formatting
        **middleware_kwargs: Additional arguments for the middleware

    Returns:
        Configured RequestResponseLoggingMiddleware instance

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> 
        >>> # Add the logging middleware
        >>> middleware = create_logging_middleware(
        ...     log_level="DEBUG",
        ...     log_file="logs/api.log",
        ...     json_format=True
        ... )
        >>> app.add_middleware(RequestResponseLoggingMiddleware, **middleware.get_config())
    """
    # Set up logging configuration
    from app.logging_config import setup_logging
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        json_format=json_format
    )

    # Return a partial function with the configuration
    def middleware_factory(app: ASGIApp):
        return RequestResponseLoggingMiddleware(
            app=app,
            logger_name=logger_name,
            **middleware_kwargs
        )

    return middleware_factory
