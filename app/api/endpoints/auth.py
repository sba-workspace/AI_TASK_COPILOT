"""
Authentication endpoints for the AI Task Copilot API.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.logging import logger
from app.db.supabase import get_supabase_client

router = APIRouter()


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class OAuthRequest(BaseModel):
    """OAuth request model."""
    provider: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "github"
            }
        }


@router.post("/oauth", response_model=dict)
async def oauth_login(request: OAuthRequest):
    """
    Initiate OAuth login flow.
    Returns the OAuth URL to redirect the user to.
    """
    try:
        supabase = get_supabase_client()
        
        if request.provider not in ["github", "slack"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider: {request.provider}",
            )
        
        # Get OAuth URL from Supabase
        # In a real implementation, you'd define scopes, redirect URL, etc.
        try:
            oauth_response = supabase.auth.sign_in_with_oauth({
                "provider": request.provider,
                "options": {
                    "redirect_to": "your-frontend-url/auth/callback",
                }
            })
            
            return {"url": oauth_response.url}
        except Exception as e:
            logger.error(f"Supabase OAuth error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initiate OAuth flow",
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


@router.post("/token", response_model=TokenResponse)
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