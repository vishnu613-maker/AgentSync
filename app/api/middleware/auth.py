import logging
from fastapi import Request, status, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)
auth_service = AuthService()

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware - validates JWT tokens or API keys
    Skips auth for public endpoints (health, root, docs, token/verify)
    """

    PUBLIC_PATHS = [
        "/health",
        "/health/",
        "/",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
        "/favicon.ico/",
        "/auth/token",
        "/auth/token/",
        "/auth/verify",     
        "/auth/verify/",
        # "/agents/list",
        
    ]

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public endpoints
        if request.url.path in self.PUBLIC_PATHS:
            logger.info(f"Skipping authentication for {request.url.path}")
            return await call_next(request)

        # Extract auth header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            logger.warning(f"Missing auth header for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            # Check if it's a Bearer token (JWT) or API key
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = auth_service.verify_token(token)
                request.state.user = payload
                logger.info(f"User set from JWT for {request.url.path}")
            elif auth_header.startswith("ApiKey "):
                api_key = auth_header.split(" ")[1]
                user = auth_service.verify_api_key(api_key)
                request.state.user = user
                logger.info(f"User set from API key for {request.url.path}")
            else:
                logger.error("Invalid authorization format")
                raise ValueError("Invalid authorization format")

            response = await call_next(request)
            return response

        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
