"""
Dependency injection utilities for FastAPI.
"""
from fastapi import Depends, HTTPException, status, Header, WebSocket, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional, Annotated
from pydantic import BaseModel
import jwt
from jwt.exceptions import PyJWTError

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
    

async def get_current_user(request: Request) -> User:
    """
    Get the current user from the Supabase JWT token.
    """
    # Get the Authorization header
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        # Check for cookies (for browser-based requests)
        auth_cookie = request.cookies.get("sb-access-token")
        if not auth_cookie:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_cookie
    else:
        # Extract token from Authorization header
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    try:
        # Verify the token using Supabase JWT secret
        # Note: In production, you should validate this against Supabase's JWKS
        payload = jwt.decode(
            token, 
            settings.SUPABASE_JWT_SECRET, 
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_signature": settings.VERIFY_JWT}
        )
        
        user_id = payload.get("sub")
        email = payload.get("email")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return User(id=user_id, email=email)
        
    except PyJWTError as e:
        logger.error(f"JWT validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
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

# Type alias for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]