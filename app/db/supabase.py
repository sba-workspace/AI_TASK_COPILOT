"""
Supabase client for authentication and database operations.
"""
from functools import lru_cache

from supabase import create_client, Client

from app.core.config import settings
from app.core.logging import logger


@lru_cache()
def get_supabase_client() -> Client:
    """
    Create and return a Supabase client instance.
    Uses lru_cache to ensure only one client is created.
    """
    try:
        logger.debug("Creating Supabase client")
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        return supabase
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}", exc_info=True)
        # Re-raise to allow proper error handling at the API level
        raise 