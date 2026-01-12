"""
Rate limiting middleware
Tracks requests per client and enforces rate limits
"""
import time
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.database.connection import redis_client

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiter using Redis - 100 requests per minute per IP
    """
    
    # Configuration
    REQUESTS_PER_MINUTE = 100
    WINDOW_SIZE_SECONDS = 60
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/"]:
            return await call_next(request)
        
        try:
            # Redis key for this client
            rate_limit_key = f"rate_limit:{client_ip}"
            
            # Increment request count
            current_requests = redis_client.incr(rate_limit_key)
            
            # Set expiration on first request
            if current_requests == 1:
                redis_client.expire(rate_limit_key, self.WINDOW_SIZE_SECONDS)
            
            # Check if rate limit exceeded
            if current_requests > self.REQUESTS_PER_MINUTE:
                logger.warning(
                    f"Rate limit exceeded for {client_ip}: "
                    f"{current_requests} requests in {self.WINDOW_SIZE_SECONDS}s"
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "status": "error",
                        "message": "Rate limit exceeded. Max 100 requests per minute.",
                        "retry_after": self.WINDOW_SIZE_SECONDS
                    }
                )
            
            # Add rate limit info to response headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.REQUESTS_PER_MINUTE)
            response.headers["X-RateLimit-Remaining"] = str(
                self.REQUESTS_PER_MINUTE - current_requests
            )
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time()) + self.WINDOW_SIZE_SECONDS
            )
            
            return response
            
        except Exception as e:
            # If Redis fails, allow request but log error
            logger.error(f"Rate limit check failed: {str(e)}")
            return await call_next(request)
