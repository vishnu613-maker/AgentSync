"""
Global error handling middleware
Converts exceptions to standardized JSON responses
"""
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handler - converts exceptions to JSON responses
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except ValidationError as e:
            # Pydantic validation errors (400 Bad Request)
            logger.warning(f"Validation error on {request.url.path}: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "message": "Validation error",
                    "errors": e.errors()
                }
            )
            
        except ValueError as e:
            # Value errors (400 Bad Request)
            logger.warning(f"Value error on {request.url.path}: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "message": str(e)
                }
            )
            
        except ConnectionError as e:
            # Database connection errors (503 Service Unavailable)
            logger.error(f"Connection error on {request.url.path}: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "error",
                    "message": "Service temporarily unavailable"
                }
            )
            
        except Exception as e:
            # Generic errors (500 Internal Server Error)
            logger.error(f"Unexpected error on {request.url.path}: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "error",
                    "message": "Internal server error"
                }
            )
