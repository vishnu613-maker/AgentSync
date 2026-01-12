"""
Authentication endpoints
Generate and validate tokens
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
auth_service = AuthService()

class TokenRequest(BaseModel):
    """Token generation request"""
    username: str
    password: str

class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    token_type: str

@router.post("/token", response_model=TokenResponse)
def get_token(request: TokenRequest):
    """
    Generate JWT token
    
    For development: username=test, password=test
    """
    # TODO: Implement proper user validation against database
    if request.username == "test" and request.password == "test":
        token = auth_service.create_token(
            data={
                "user_id": 1,
                "username": request.username
            }
        )
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

@router.get("/verify")
def verify_token():
    """
    Verify token is valid
    This endpoint requires valid auth (tests middleware)
    """
    return {
        "status": "authenticated",
        "message": "Token is valid"
    }
