"""
Authentication endpoints for the AI Task Copilot API.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Cookie
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
import secrets
import base64
import hashlib
import json
import uuid

from app.core.logging import logger
from app.db.supabase import get_supabase_client
from app.api.deps import get_current_user, User, security
from app.core.config import settings

router = APIRouter()

# Store PKCE state temporarily (in memory - would need a proper storage in production)
pkce_states = {}


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None


class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str


class OAuthRequest(BaseModel):
    """OAuth request model."""
    provider: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "github"
            }
        }


SUPPORTED_PROVIDERS = [
    "github",
    "google",
    "slack",

]


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """
    Register a new user using email and password.
    """
    try:
        supabase = get_supabase_client()
        
        # Register user with Supabase
        try:
            auth_response = supabase.auth.sign_up({
                "email": user_data.email,
                "password": user_data.password
            })
            
            if auth_response.user is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Registration failed"
                )
            
            return UserResponse(
                id=auth_response.user.id,
                email=auth_response.user.email
            )
        except Exception as e:
            logger.error(f"Supabase registration error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registration failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """
    Login using email and password.
    """
    try:
        supabase = get_supabase_client()
        
        # Authenticate with Supabase
        try:
            auth_response = supabase.auth.sign_in_with_password({
                "email": user_data.email,
                "password": user_data.password
            })
            
            if auth_response.session is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return TokenResponse(
                access_token=auth_response.session.access_token,
                token_type="bearer",
                refresh_token=auth_response.session.refresh_token,
                expires_in=auth_response.session.expires_in
            )
        except Exception as e:
            logger.error(f"Supabase login error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


@router.post("/token", response_model=TokenResponse)
async def login_oauth2_form(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login using OAuth2 password flow (for compatibility with OAuth2 clients).
    This endpoint is used by the Swagger UI.
    """
    try:
        supabase = get_supabase_client()
        
        # Authenticate with Supabase
        try:
            auth_response = supabase.auth.sign_in_with_password({
                "email": form_data.username,  # OAuth2 form uses username field for email
                "password": form_data.password
            })
            
            if auth_response.session is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return TokenResponse(
                access_token=auth_response.session.access_token,
                token_type="bearer",
                refresh_token=auth_response.session.refresh_token,
                expires_in=auth_response.session.expires_in
            )
        except Exception as e:
            logger.error(f"Supabase OAuth2 login error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth2 login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


@router.get("/me", response_model=UserResponse)
async def get_user_me(current_user: User = Depends(get_current_user)):
    """
    Get the current authenticated user.
    """
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout the current user.
    """
    try:
        supabase = get_supabase_client()
        
        # Sign out from Supabase
        try:
            supabase.auth.sign_out()
            return {"message": "Successfully logged out"}
        except Exception as e:
            logger.error(f"Supabase logout error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Logout failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


def generate_code_verifier(length: int = 43) -> str:
    """
    Generate a code verifier for PKCE.
    Using 43 characters as per Supabase's implementation.
    """
    code_verifier = secrets.token_urlsafe(32)  # This generates ~43 chars
    return code_verifier

def generate_code_challenge(verifier: str) -> str:
    """
    Generate a code challenge for PKCE using SHA256 and base64url encoding.
    Following RFC 7636 specification.
    """
    m = hashlib.sha256()
    m.update(verifier.encode('ascii'))
    code_challenge = base64.urlsafe_b64encode(m.digest()).decode('ascii').replace('=', '')
    return code_challenge

@router.get("/oauth/login/{provider}")
async def oauth_login(provider: str, request: Request):
    """
    Initiate OAuth login flow with the specified provider.
    Returns the OAuth URL to redirect the user to.
    """
    try:
        supabase = get_supabase_client()
        
        # Validate provider
        if not provider or provider.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider not specified. Supported providers are: {', '.join(SUPPORTED_PROVIDERS)}",
            )
            
        provider = provider.lower().strip()
        if provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider: '{provider}'. Supported providers are: {', '.join(SUPPORTED_PROVIDERS)}",
            )
        
        # Get the redirect URL - use our callback endpoint
        base_url = str(request.base_url).rstrip('/')
        redirect_url = f"{base_url}/api/auth/oauth/callback"
        
        logger.info(f"Initiating OAuth flow for provider: {provider} with redirect URL: {redirect_url}")
        
        # Get OAuth URL from Supabase - without PKCE
        try:
            oauth_response = supabase.auth.sign_in_with_oauth({
                "provider": provider,
                "options": {
                    "redirect_to": redirect_url,
                    "scopes": settings.OAUTH_SCOPES.get(provider, ""),
                }
            })
            
            if not oauth_response or not oauth_response.url:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get OAuth URL for provider: {provider}",
                )
            
            logger.info("Successfully initiated OAuth flow")
            return {"url": oauth_response.url}
            
        except Exception as e:
            logger.error(f"Supabase OAuth error for provider '{provider}': {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initiate OAuth flow for provider '{provider}': {str(e)}",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


class TokenRequest(BaseModel):
    """Token exchange request model."""
    code: str


@router.post("/exchange_token", response_model=TokenResponse)
async def exchange_token(request: TokenRequest):
    """
    Exchange an OAuth code for an access token.
    In a real implementation, this would verify the code and return a token.
    """
    try:
        # This is a placeholder - in a real implementation,
        # you'd exchange the code for a token via Supabase
        
        # Mock response for development
        return TokenResponse(
            access_token="mock_token_for_development",
            token_type="bearer",
        )
        
    except Exception as e:
        logger.error(f"Token exchange error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to exchange token",
        )


@router.get("/oauth/callback")
async def oauth_callback(request: Request, code: str = None, response: Response = None):
    """
    Handle OAuth callback.
    Process the callback parameters from Supabase Auth and get the session.
    Returns tokens for the frontend to handle or redirects to the frontend callback URL.
    """
    try:
        # Get the full URL and code for debugging
        full_url = str(request.url)
        logger.info(f"Received OAuth callback with URL: {full_url}")
        
        if not code:
            logger.error("No code parameter in callback URL")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No authorization code provided in callback"
            )
            
        logger.info(f"Attempting to exchange code: {code}")
        
        supabase = get_supabase_client()
        try:
            # Exchange code for session
            auth_response = supabase.auth.exchange_code_for_session({
                "auth_code": code
            })
            
            if not auth_response or not hasattr(auth_response, 'session') or not auth_response.session:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get session from code"
                )
                
            session = auth_response.session
            access_token = session.access_token
            refresh_token = session.refresh_token
            expires_in = session.expires_in
            
            # Get frontend callback URL from config or query params
            frontend_url = settings.FRONTEND_URL if hasattr(settings, 'FRONTEND_URL') else None
            redirect_to = request.query_params.get('redirect_to')
            
            # Option 1: Return JSON response with tokens (for SPA frontend to handle)
            if not frontend_url and not redirect_to:
                return {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": expires_in
                }
            
            # Option 2: Redirect to frontend with token in URL fragment
            redirect_url = redirect_to or f"{frontend_url}/auth/callback"
            # Use a hash fragment to pass the token securely to the frontend
            full_redirect_url = f"{redirect_url}#access_token={access_token}&token_type=bearer"
            
            return RedirectResponse(url=full_redirect_url)
            
        except Exception as e:
            logger.error(f"Failed to process OAuth callback: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process OAuth callback: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication service unavailable: {str(e)}"
        )