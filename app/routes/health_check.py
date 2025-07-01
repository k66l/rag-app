from fastapi import APIRouter
from pydantic import BaseModel

# Create router for health check endpoints
router = APIRouter()


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    message: str
    status: str


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service is running and healthy",
    response_description="Returns a welcome message and service status",
    tags=["Health Check"]
)
def read_root() -> HealthResponse:
    """
    Simple health check endpoint to verify the service is running.

    This endpoint is useful for:
    - Load balancer health checks
    - Monitoring systems
    - Quick verification that the API is responsive

    Returns:
        HealthResponse: A simple message confirming the service is running
    """
    return HealthResponse(
        message="Welcome to the RAG Application API!",
        status="healthy"
    )
