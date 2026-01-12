"""
Authentication service
Handles JWT token generation, validation, and API key management
"""
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class AuthService:
    """
    Authentication service for JWT and API key management
    """
    
    def __init__(self):
        self.algorithm = settings.jwt_algorithm
        self.secret_key = settings.secret_key
        self.expiration_minutes = 1440  # 24 hours
    
    def create_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT token
        
        Args:
            data: Payload to encode
            expires_delta: Custom expiration time
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.expiration_minutes)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded payload
            
        Raises:
            Exception: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            logger.error("Invalid token")
            raise Exception("Invalid token")
    
    def verify_api_key(self, api_key: str) -> Dict:
        """
        Verify API key
        
        For now, this is a simple check.
        In production, verify against database.
        
        Args:
            api_key: API key to verify
            
        Returns:
            User data dict
            
        Raises:
            Exception: If API key is invalid
        """
        # TODO: Implement proper API key storage and validation in database
        # For development, we'll use a simple hardcoded key
        
        valid_keys = {
            "test-api-key-123": {"user_id": 1, "username": "test_user"}
        }
        
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}***")
            raise Exception("Invalid API key")
        
        return valid_keys[api_key]
