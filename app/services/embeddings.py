"""
Embeddings service for vector representations.
"""
import asyncio
import json
import base64
from functools import lru_cache
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import timedelta
import torch
import warnings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)
from langchain_cohere import CohereEmbeddings
from langchain_core.embeddings import Embeddings
import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Initialize Redis client
redis_client = redis.Redis.from_url(settings.REDIS_URL)


class CachedEmbeddings:
    """Custom wrapper for embeddings with Redis caching."""
    
    def __init__(self, embeddings: Embeddings, ttl_days: int = 7):
        self.embeddings = embeddings
        self.ttl_days = ttl_days
        self.namespace = "embeddings_cache:"
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the text."""
        # Use hash to avoid key length issues
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.namespace}{text_hash}"
    
    async def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        try:
            cache_key = self._get_cache_key(text)
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Error getting from cache: {e}")
            return None
    
    async def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        try:
            cache_key = self._get_cache_key(text)
            await redis_client.set(
                cache_key,
                json.dumps(embedding),
                ex=int(timedelta(days=self.ttl_days).total_seconds())
            )
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single text with caching."""
        # Try to get from cache first
        cached = await self._get_from_cache(text)
        if cached is not None:
            logger.debug(f"Cache hit for query")
            return cached
        
        # If not in cache, generate embedding
        try:
            if hasattr(self.embeddings, 'embed_query'):
                embedding = self.embeddings.embed_query(text)
            else:
                # Some embedding classes might not have embed_query
                embeddings = self.embeddings.embed_documents([text])
                embedding = embeddings[0] if embeddings else []
            
            # Save to cache
            await self._save_to_cache(text, embedding)
            logger.debug(f"Generated and cached new embedding")
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            raise
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with caching."""
        if not texts:
            return []
        
        # Try to get from cache first
        try:
            cached_results = await asyncio.gather(
                *[self._get_from_cache(text) for text in texts],
                return_exceptions=True
            )
        except Exception as e:
            logger.warning(f"Error checking cache: {e}")
            cached_results = [None] * len(texts)
        
        # Find texts that need embedding
        to_embed = []
        to_embed_indices = []
        results = [None] * len(texts)
        
        for i, (text, cached) in enumerate(zip(texts, cached_results)):
            if cached is not None and not isinstance(cached, Exception):
                results[i] = cached
            else:
                to_embed.append(text)
                to_embed_indices.append(i)
        
        logger.info(f"Cache hits: {len(texts) - len(to_embed)}/{len(texts)}")
        
        # Generate embeddings for uncached texts
        if to_embed:
            try:
                new_embeddings = self.embeddings.embed_documents(to_embed)
                
                # Save new embeddings to cache (without waiting)
                cache_tasks = [
                    self._save_to_cache(text, embedding)
                    for text, embedding in zip(to_embed, new_embeddings)
                ]
                asyncio.create_task(asyncio.gather(*cache_tasks, return_exceptions=True))
                
                # Update results
                for idx, embedding in zip(to_embed_indices, new_embeddings):
                    results[idx] = embedding
                    
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise
        
        return results


@lru_cache(maxsize=1)
def get_device() -> str:
    """Get the appropriate device (CPU/CUDA) for the embedding model."""
    if torch.cuda.is_available() and getattr(settings, 'USE_GPU', False):
        logger.info("Using CUDA device for embeddings")
        return "cuda"
    logger.info("Using CPU device for embeddings")
    return "cpu"


# Global variable to store the embeddings instance
_embeddings_instance = None


def _create_embeddings_instance(fallback_level: int = 0) -> Embeddings:
    """
    Create embeddings model instance.
    
    Args:
        fallback_level: Level of fallback (0=primary, 1=cohere, 2=mini)
        
    Returns:
        Embedding model instance
    """
    try:
        if fallback_level == 0:
            # Try primary model - use a valid model name
            logger.info("Loading primary embedding model: sentence-transformers/all-mpnet-base-v2")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                base_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={"device": get_device()},
                    encode_kwargs={"normalize_embeddings": True}
                )
        
        elif fallback_level == 1:
            # Try Cohere if available
            if hasattr(settings, 'COHERE_API_KEY') and settings.COHERE_API_KEY:
                logger.info("Loading Cohere embedding model")
                base_embeddings = CohereEmbeddings(
                    cohere_api_key=settings.COHERE_API_KEY,
                    model="embed-english-v3.0"
                )
            else:
                logger.warning("Cohere API key not available, skipping to next fallback")
                return _create_embeddings_instance(fallback_level + 1)
        
        else:
            # Final fallback to a smaller local model
            logger.info("Loading fallback embedding model: sentence-transformers/all-MiniLM-L6-v2")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                base_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": get_device()},
                    encode_kwargs={"normalize_embeddings": True}
                )
        
        # Test the embedding model
        try:
            test_embedding = base_embeddings.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding model returned empty result")
            logger.info(f"Embedding model test successful, dimension: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Embedding model test failed: {e}")
            if fallback_level < 2:
                return _create_embeddings_instance(fallback_level + 1)
            raise
        
        return base_embeddings
        
    except Exception as e:
        logger.error(f"Failed to create embedding model at fallback level {fallback_level}: {e}")
        if fallback_level < 2:
            return _create_embeddings_instance(fallback_level + 1)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((OSError, RuntimeError, ConnectionError))
)
def get_embeddings() -> CachedEmbeddings:
    """
    Get or create embeddings model instance with caching.
    Uses global variable to ensure only one model is loaded.
    
    Returns:
        Cached embedding model instance
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        logger.info("Initializing embeddings service")
        try:
            base_embeddings = _create_embeddings_instance()
            _embeddings_instance = CachedEmbeddings(base_embeddings)
            logger.info("Embeddings service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings service: {e}", exc_info=True)
            raise
    
    return _embeddings_instance


async def embed_texts(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False
) -> List[List[float]]:
    """
    Embed a list of texts using the embedding model.
    Includes batching and progress tracking.
    
    Args:
        texts: List of text strings to embed
        batch_size: Size of batches for processing
        show_progress: Whether to show progress bar
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    # Filter out empty texts
    non_empty_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
    if not non_empty_texts:
        logger.warning("All texts are empty, returning empty embeddings")
        return [[0.0] * 384] * len(texts)  # Return dummy embeddings
    
    try:
        embeddings = get_embeddings()
        all_embeddings = [None] * len(texts)
        
        # Process non-empty texts
        texts_to_embed = [text for _, text in non_empty_texts]
        
        # Process in batches
        batch_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            try:
                batch_result = await embeddings.embed_documents(batch)
                batch_embeddings.extend(batch_result)
                
                if show_progress:
                    logger.info(f"Embedded {i + len(batch)}/{len(texts_to_embed)} texts")
                    
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                # Add dummy embeddings for failed batch
                dummy_embedding = [0.0] * 384
                batch_embeddings.extend([dummy_embedding] * len(batch))
        
        # Map embeddings back to original positions
        for (original_idx, _), embedding in zip(non_empty_texts, batch_embeddings):
            all_embeddings[original_idx] = embedding
        
        # Fill in dummy embeddings for empty texts
        dummy_embedding = [0.0] * 384
        for i, embedding in enumerate(all_embeddings):
            if embedding is None:
                all_embeddings[i] = dummy_embedding
        
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error embedding texts: {e}", exc_info=True)
        # Return dummy embeddings instead of failing
        dummy_embedding = [0.0] * 384
        return [dummy_embedding] * len(texts)


async def embed_query(
    query: str,
    retry_count: int = 3
) -> List[float]:
    """
    Embed a single query text using the embedding model.
    Includes retry logic for reliability.
    
    Args:
        query: The query text to embed
        retry_count: Number of retries on failure
        
    Returns:
        Embedding vector
    """
    if not query or not query.strip():
        logger.warning("Empty query provided, returning dummy embedding")
        return [0.0] * 384
    
    for attempt in range(retry_count):
        try:
            embeddings = get_embeddings()
            return await embeddings.embed_query(query.strip())
        except Exception as e:
            if attempt == retry_count - 1:
                logger.error(f"Error embedding query after {retry_count} attempts: {e}")
                # Return dummy embedding instead of raising
                return [0.0] * 384
            logger.warning(f"Retry {attempt + 1}/{retry_count} after error: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


async def get_embedding_stats() -> dict:
    """
    Get statistics about the embedding cache and model.
    
    Returns:
        Dictionary with statistics
    """
    try:
        # Get cache stats
        try:
            cache_keys = await redis_client.keys("embeddings_cache:*")
            cache_size = len(cache_keys)
        except Exception as e:
            logger.warning(f"Could not get cache stats: {e}")
            cache_size = -1
        
        # Get model info
        try:
            embeddings = get_embeddings()
            base_embeddings = embeddings.embeddings
            
            model_name = "unknown"
            if hasattr(base_embeddings, 'model_name'):
                model_name = base_embeddings.model_name
            elif hasattr(base_embeddings, 'model'):
                model_name = str(base_embeddings.model)
                
            is_cohere = isinstance(base_embeddings, CohereEmbeddings)
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            model_name = "error"
            is_cohere = False
        
        # Get device info
        device = get_device()
        
        return {
            "cache_size": cache_size,
            "model_name": model_name,
            "device": device,
            "using_cohere": is_cohere,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error getting embedding stats: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "error"
        }


async def clear_embedding_cache() -> bool:
    """
    Clear the embedding cache.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_keys = await redis_client.keys("embeddings_cache:*")
        if cache_keys:
            await redis_client.delete(*cache_keys)
            logger.info(f"Cleared {len(cache_keys)} items from embedding cache")
        return True
    except Exception as e:
        logger.error(f"Error clearing embedding cache: {e}")
        return False


def reset_embeddings_instance():
    """Reset the global embeddings instance (useful for testing)."""
    global _embeddings_instance
    _embeddings_instance = None
    logger.info("Embeddings instance reset")