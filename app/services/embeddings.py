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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    CohereEmbeddings
)
from langchain_core.embeddings import Embeddings
import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger


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
        return f"{self.namespace}{text}"
    
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
                ex=timedelta(days=self.ttl_days)
            )
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single text with caching."""
        # Try to get from cache first
        cached = await self._get_from_cache(text)
        if cached is not None:
            return cached
        
        # If not in cache, generate embedding
        embedding = self.embeddings.embed_query(text)
        
        # Save to cache
        await self._save_to_cache(text, embedding)
        
        return embedding
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with caching."""
        if not texts:
            return []
        
        # Try to get from cache first
        cached_results = await asyncio.gather(
            *[self._get_from_cache(text) for text in texts]
        )
        
        # Find texts that need embedding
        to_embed = []
        to_embed_indices = []
        results = [None] * len(texts)
        
        for i, (text, cached) in enumerate(zip(texts, cached_results)):
            if cached is not None:
                results[i] = cached
            else:
                to_embed.append(text)
                to_embed_indices.append(i)
        
        # Generate embeddings for uncached texts
        if to_embed:
            new_embeddings = self.embeddings.embed_documents(to_embed)
            
            # Save new embeddings to cache
            await asyncio.gather(*[
                self._save_to_cache(text, embedding)
                for text, embedding in zip(to_embed, new_embeddings)
            ])
            
            # Update results
            for idx, embedding in zip(to_embed_indices, new_embeddings):
                results[idx] = embedding
        
        return results


@lru_cache()
def get_device() -> str:
    """Get the appropriate device (CPU/CUDA) for the embedding model."""
    if torch.cuda.is_available() and settings.USE_GPU:
        return "cuda"
    return "cpu"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((OSError, RuntimeError))
)
@lru_cache()
def get_embeddings(fallback: bool = False) -> Embeddings:
    """
    Create and return an embeddings model instance with caching.
    Uses lru_cache to ensure only one model is loaded.
    
    Args:
        fallback: Whether to use fallback model
        
    Returns:
        Cached embedding model instance
    """
    try:
        logger.info("Loading embedding model")
        
        # Try primary model (nomic-embed-text)
        if not fallback:
            base_embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text",
                model_kwargs={"device": get_device()},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            # Fallback to Cohere if available
            if settings.COHERE_API_KEY:
                base_embeddings = CohereEmbeddings(
                    cohere_api_key=settings.COHERE_API_KEY
                )
            else:
                # Final fallback to a smaller local model
                base_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": get_device()},
                    encode_kwargs={"normalize_embeddings": True}
                )
        
        # Wrap with our custom caching
        cached_embeddings = CachedEmbeddings(base_embeddings)
        
        return cached_embeddings
    except Exception as e:
        if not fallback:
            logger.warning(f"Primary embedding model failed, trying fallback: {e}")
            return get_embeddings(fallback=True)
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        raise


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
    
    try:
        embeddings = get_embeddings()
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                logger.info(f"Embedded {i + len(batch)}/{len(texts)} texts")
        
        return all_embeddings
    except Exception as e:
        logger.error(f"Error embedding texts: {e}", exc_info=True)
        raise


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
    if not query:
        raise ValueError("Query text cannot be empty")
    
    for attempt in range(retry_count):
        try:
            embeddings = get_embeddings()
            return await embeddings.embed_query(query)
        except Exception as e:
            if attempt == retry_count - 1:
                logger.error(f"Error embedding query after {retry_count} attempts: {e}")
                raise
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
        cache_keys = await redis_client.keys("embeddings_cache:*")
        cache_size = len(cache_keys)
        
        # Get model info
        embeddings = get_embeddings()
        model_name = embeddings.embeddings.model_name if hasattr(embeddings.embeddings, 'model_name') else 'unknown'
        
        # Get device info
        device = get_device()
        
        return {
            "cache_size": cache_size,
            "model_name": model_name,
            "device": device,
            "using_fallback": isinstance(embeddings.embeddings, CohereEmbeddings)
        }
    except Exception as e:
        logger.error(f"Error getting embedding stats: {e}", exc_info=True)
        return {"error": str(e)} 