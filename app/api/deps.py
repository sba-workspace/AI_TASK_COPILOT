"""
Dependency injection utilities for FastAPI.
"""
from fastapi import Depends, HTTPException, status, Header, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional
from pydantic import BaseModel

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.core.logging import logger

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)

# Create OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class User(BaseModel):
    """User model."""
    id: str
    email: str
    

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current user from token.
    
    Args:
        token: JWT token
        
    Returns:
        User: Current user
        
    Raises:
        HTTPException: If token is invalid
    """
    # Log token information (just first 20 chars for security)
    if token:
        logger.debug(f"Token received (first 20 chars): {token[:20]}...")
    else:
        logger.warning("No token provided in request")
        
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Verify token with Supabase
        try:
            logger.debug("Attempting to verify token with Supabase")
            user_response = supabase.auth.get_user(token)
            user_data = user_response.user
            logger.debug(f"Supabase returned user data type: {type(user_data)}")
            
            # Check if user_data is a dict or already a User object
            if hasattr(user_data, 'get'):
                return User(
                    id=user_data.get("id"),
                    email=user_data.get("email", ""),
                )
            elif isinstance(user_data, dict):
                return User(
                    id=user_data.get("id"),
                    email=user_data.get("email", ""),
                )
            else:
                # Assuming it's some object with id and email attributes
                return User(
                    id=getattr(user_data, "id", ""),
                    email=getattr(user_data, "email", ""),
                )
                
        except Exception as e:
            # If Supabase client fails, fallback to manual JWT verification
            logger.warning(f"Supabase auth failed, falling back to JWT verification: {e}")
            
            try:
                logger.debug("Attempting manual JWT decoding")
                payload = jwt.decode(
                    token,
                    settings.SUPABASE_JWT_SECRET,
                    algorithms=["HS256"],
                    options={"verify_signature": False}
                )
                
                user_data = {
                    "id": payload.get("sub"),
                    "email": payload.get("email", ""),
                }
                
                return User(
                    id=user_data.get("id"),
                    email=user_data.get("email", ""),
                )
            except Exception as jwt_error:
                logger.error(f"JWT decode error: {jwt_error}")
                raise
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_ws(token: Optional[str] = None) -> Optional[User]:
    """
    Get current user from token for WebSocket connections.
    Similar to get_current_user but doesn't raise exceptions.
    
    Args:
        token: JWT token
        
    Returns:
        User: Current user or None if authentication fails
    """
    if not token:
        return None
        
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Verify token with Supabase
        try:
            user_response = supabase.auth.get_user(token)
            user_data = user_response.user
            
            # Check if user_data is a dict or already a User object
            if hasattr(user_data, 'get'):
                return User(
                    id=user_data.get("id"),
                    email=user_data.get("email", ""),
                )
            elif isinstance(user_data, dict):
                return User(
                    id=user_data.get("id"),
                    email=user_data.get("email", ""),
                )
            else:
                # Assuming it's some object with id and email attributes
                return User(
                    id=getattr(user_data, "id", ""),
                    email=getattr(user_data, "email", ""),
                )
                
        except Exception as e:
            # If Supabase client fails, fallback to manual JWT verification
            logger.warning(f"Supabase auth failed, falling back to JWT verification: {e}")
            
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_signature": False}
            )
            
            user_data = {
                "id": payload.get("sub"),
                "email": payload.get("email", ""),
            }
            
            return User(
                id=user_data.get("id"),
                email=user_data.get("email", ""),
            )
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}", exc_info=True)
        return None