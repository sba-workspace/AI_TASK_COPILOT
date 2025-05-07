"""
Dependency injection utilities for FastAPI.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import settings
from app.db.supabase import get_supabase_client

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    tokenUrl=f"{settings.API_PREFIX}/auth/token",
    authorizationUrl=f"{settings.SUPABASE_URL}/auth/v1/authorize"
)


class User(BaseModel):
    """User model."""
    id: str
    email: str
    

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Validate token and return current user.
    This dependency can be used to protect routes that require authentication.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Verify token with Supabase
        try:
            user_response = supabase.auth.get_user(token)
            user = user_response.user
            if not user:
                raise credentials_exception
        except Exception:
            # If Supabase client fails, fallback to manual JWT verification
            try:
                payload = jwt.decode(
                    token, 
                    settings.SUPABASE_JWT_SECRET, 
                    algorithms=["HS256"],
                    options={"verify_sub": True}
                )
                user_id = payload.get("sub")
                email = payload.get("email")
                if user_id is None or email is None:
                    raise credentials_exception
                return User(id=user_id, email=email)
            except JWTError:
                raise credentials_exception
        
        # Return user from Supabase response
        return User(id=user.id, email=user.email)
    
    except Exception:
        raise credentials_exception