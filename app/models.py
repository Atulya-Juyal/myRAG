"""
API requests and Respone models
Pydantic models for input validation and response structure
"""

from pydantic import BaseModel, Field
from datetime import timezone, datetime


class ChatRequest(BaseModel):
    """Incoming chat requests"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's message to the agent"
    )
    thread_id: str = Field(
        default="default",
        description="Unique identifier for the conversation thread"
    )


class ChatResponse(BaseModel):
    """Chat response from the agent"""
    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    environment: str
    version: str = "1.0.0"
    checks: dict = {}


class MetricsResponse(BaseModel):
    """Metrics response for monitoring"""
    total_requests: int
    total_errors: int
    avg_latency_ms: float
    cache_hit_rate: float
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str | None = None
    request_id: str | None = None


