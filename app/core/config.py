"""
Configuration settings for the application.
"""
import os
from typing import Any, Dict, List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # FastAPI settings
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    APP_ENV: str = "development"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Frontend URL for redirects
    FRONTEND_URL: str = "http://localhost:3000"
    
    # OAuth settings
    OAUTH_REDIRECT_URL: Optional[str] = None
    OAUTH_SCOPES: Dict[str, str] = {
        "github": "read:user user:email",
        "google": "openid email profile",
        
        "slack": "identity.basic identity.email",
        
    }
    
    # LLM settings
    GOOGLE_API_KEY: str
    
    # Weaviate settings
    WEAVIATE_URL: str
    WEAVIATE_API_KEY: Optional[str] = None
    
    # Supabase settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_JWT_SECRET: str
    VERIFY_JWT: bool = True
    
    # Notion API
    NOTION_API_KEY: str
    
    # GitHub API
    GITHUB_API_TOKEN: str
    
    # Slack API
    SLACK_BOT_TOKEN: str
    SLACK_APP_TOKEN: Optional[str] = None
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Cohere settings
    COHERE_API_KEY: str = ""
    
    # GPU settings
    USE_GPU: bool = False

    # Embedding retry settings
    EMBEDDING_RETRY_ATTEMPTS: int = 3
    EMBEDDING_RETRY_MULTIPLIER: int = 1
    EMBEDDING_RETRY_MIN_WAIT: int = 2  # seconds
    EMBEDDING_RETRY_MAX_WAIT: int = 10 # seconds

    # Primary and Fallback Embedding Model Names
    PRIMARY_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    FALLBACK_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    @field_validator("USE_GPU", "VERIFY_JWT", mode="before")
    def parse_boolean(cls, v: Any) -> bool:
        """Parse boolean values, handling comments in env file."""
        if isinstance(v, str):
            # Remove comments and whitespace
            clean_value = v.split('#')[0].strip().lower()
            if clean_value in ('true', '1', 'yes', 'y'):
                return True
            elif clean_value in ('false', '0', 'no', 'n', ''):
                return False
        return bool(v)
    
    @field_validator("CORS_ORIGINS")
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        raise ValueError(v)

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Create settings instance
settings = Settings()