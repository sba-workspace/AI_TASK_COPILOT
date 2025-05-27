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
                # Run the synchronous embedding generation in a thread pool executor
                loop = asyncio.get_running_loop()
                new_embeddings = await loop.run_in_executor(
                    None,  # Uses the default ThreadPoolExecutor
                    self.embeddings.embed_documents,
                    to_embed
                )
                
                # Save new embeddings to cache
                save_tasks = [
                    self._save_to_cache(text, embedding)
                    for text, embedding in zip(to_embed, new_embeddings)
                ]
                await asyncio.gather(*save_tasks)
                
                # Update results
                for i, embedding_idx in enumerate(to_embed_indices):
                    results[embedding_idx] = new_embeddings[i]

            except Exception as e:
                logger.error(f"Error generating or caching embeddings for a batch: {e}", exc_info=True)
                # Decide how to handle partial failures. For now, we'll let the Nones propagate
                # and they will be filtered out at the end.
                # Alternatively, re-raise the exception if any failure should stop the whole process:
                # raise
        
        # Filter out any None results that might occur if embedding failed for some texts
        final_results = [res for res in results if res is not None]
        if len(final_results) != len(texts):
            logger.warning(f"Could not generate embeddings for {len(texts) - len(final_results)} texts.")
        return final_results

# End of CachedEmbeddings class


@lru_cache()
def get_device() -> str:
    """Get the appropriate device (CPU/CUDA) for the embedding model."""
    if torch.cuda.is_available() and settings.USE_GPU:
        logger.info("Using CUDA device for embeddings.")
        return "cuda"
    logger.info("Using CPU device for embeddings.")
    return "cpu"


@retry(
    stop=stop_after_attempt(settings.EMBEDDING_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=settings.EMBEDDING_RETRY_MULTIPLIER,
        min=settings.EMBEDDING_RETRY_MIN_WAIT,
        max=settings.EMBEDDING_RETRY_MAX_WAIT
    ),
    retry=retry_if_exception_type((OSError, RuntimeError))
)
@lru_cache()
def get_embeddings(use_fallback: bool = False) -> Embeddings:
    """
    Create and return an embeddings model instance with caching.
    Uses lru_cache to ensure only one model is loaded.
    Prioritizes Nomic, falls back to Cohere, then to a local SentenceTransformer.
    """
    logger.info(f"Attempting to load embeddings model. Fallback: {use_fallback}")
    model_kwargs = {"device": get_device()}
    
    try:
        if not use_fallback and settings.PRIMARY_EMBEDDING_MODEL_NAME:
            logger.info(f"Loading primary embedding model: {settings.PRIMARY_EMBEDDING_MODEL_NAME}")
            if settings.PRIMARY_EMBEDDING_MODEL_NAME.startswith("nomic-ai"):
                 base_embeddings = HuggingFaceEmbeddings(
                    model_name=settings.PRIMARY_EMBEDDING_MODEL_NAME,
                    model_kwargs=model_kwargs,
                    encode_kwargs={'normalize_embeddings': True} 
                )
            else: # Assuming other HuggingFace models
                base_embeddings = HuggingFaceEmbeddings(
                    model_name=settings.PRIMARY_EMBEDDING_MODEL_NAME,
                    model_kwargs=model_kwargs
                )
            logger.info(f"Primary model {settings.PRIMARY_EMBEDDING_MODEL_NAME} loaded successfully.")
        elif settings.COHERE_API_KEY and (use_fallback or not settings.PRIMARY_EMBEDDING_MODEL_NAME):
            logger.info("Loading Cohere embeddings model as fallback or primary.")
            base_embeddings = CohereEmbeddings(cohere_api_key=settings.COHERE_API_KEY)
            logger.info("Cohere model loaded successfully.")
        elif settings.FALLBACK_EMBEDDING_MODEL_NAME:
            logger.info(f"Loading local fallback HuggingFace model: {settings.FALLBACK_EMBEDDING_MODEL_NAME}")
            base_embeddings = HuggingFaceEmbeddings(
                model_name=settings.FALLBACK_EMBEDDING_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True} if "all-MiniLM" in settings.FALLBACK_EMBEDDING_MODEL_NAME else {}
            )
            logger.info(f"Local fallback model {settings.FALLBACK_EMBEDDING_MODEL_NAME} loaded successfully.")
        else:
            logger.error("No embedding models configured or API keys provided.")
            raise ValueError("Embedding model configuration error.")

        return CachedEmbeddings(base_embeddings)

    except Exception as e:
        logger.error(f"Failed to load embedding model (fallback={use_fallback}): {e}", exc_info=True)
        if not use_fallback: # If primary failed, try with full fallback logic
            logger.warning("Attempting to load embeddings with full fallback sequence.")
            return get_embeddings(use_fallback=True)
        raise # If already in fallback mode and failed, raise the exception


async def embed_texts(
    texts: List[str],
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Embed a list of texts using the embedding model.
    """
    if not texts:
        return []
    
    embeddings_service = get_embeddings()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = await embeddings_service.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Error embedding batch (start index {i}): {e}", exc_info=True)
            # Add None for failed embeddings in this batch to maintain order if desired,
            # or handle more gracefully. For now, extend with empty lists or skip.
            all_embeddings.extend([[] for _ in batch]) # Placeholder for failed batch items

    # Filter out empty lists if they were used as placeholders for errors
    return [emb for emb in all_embeddings if emb]


async def embed_query(
    query: str,
) -> List[float]:
    """
    Embed a single query text using the embedding model.
    """
    if not query:
        raise ValueError("Query text cannot be empty")
    
    embeddings_service = get_embeddings()
    try:
        return await embeddings_service.embed_query(query)
    except Exception as e:
        logger.error(f"Error embedding query: {e}", exc_info=True)
        raise


async def get_embedding_stats() -> dict:
    """
    Get statistics about the embedding cache and model.
    """
    try:
        cache_keys = await redis_client.keys(f"{CachedEmbeddings(None).namespace}*") # type: ignore
        cache_size = len(cache_keys)
        
        # Determine current model without re-initializing if possible (tricky with lru_cache)
        # This is a simplified representation; real model name might need deeper inspection
        # or exposing model_name from CachedEmbeddings.
        current_model_name = "Primary/Fallback (details depend on settings)"
        device_info = get_device()

        return {
            "cache_size": cache_size,
            "approximated_model_name": current_model_name,
            "device": device_info,
        }
    except Exception as e:
        logger.error(f"Error getting embedding stats: {e}", exc_info=True)
        return {"error": str(e)}