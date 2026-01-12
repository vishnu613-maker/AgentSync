import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs incoming requests and outgoing responses
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Intercept request, log it, execute handler, log response
        """
        # Start timer
        start_time = time.time()
        
        # Log request method, path, and client IP
        logger.info(
            f"REQUEST: {request.method} {request.url.path} "
            f"| Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Log Authorization header for debugging purposes
        auth_header = request.headers.get("Authorization")
        logger.info(
            f"AUTHORIZATION HEADER: {auth_header} | Path: {request.url.path}"
        )

        try:
            # Call the actual route handler
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"RESPONSE: {request.method} {request.url.path} "
                f"| Status: {response.status_code} | Time: {process_time:.3f}s"
            )
            
            # Add response time to headers
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Log errors
            process_time = time.time() - start_time
            logger.error(
                f"ERROR: {request.method} {request.url.path} "
                f"| Exception: {str(e)} | Time: {process_time:.3f}s"
            )
            raise
