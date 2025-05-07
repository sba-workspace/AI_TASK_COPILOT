"""
LLM service for AI Task Copilot.
"""
import asyncio
from functools import lru_cache
from typing import AsyncIterator, Optional, List, Dict, Any, Callable
import json
from datetime import datetime, timedelta


import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.cache import RedisCache
import aiohttp
import redis
import redis.asyncio as redis_async
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from app.core.config import settings
from app.core.logging import logger


# Custom implementation of AsyncIteratorCallbackHandler since it's not available in the current package
class AsyncIteratorCallbackHandler:
    """Callback handler that yields tokens as they become available.
    
    Async iterator yielding tokens as they become available.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = asyncio.Event()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new token. Only available when streaming is enabled."""
        await self.queue.put(token)

    async def on_llm_end(self, *args, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.done.set()

    async def on_llm_error(self, *args, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        """Async iterator that yields tokens as they become available."""
        while not self.done.is_set() or not self.queue.empty():
            try:
                token = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                yield token
            except asyncio.TimeoutError:
                # No tokens in the last 0.1 seconds
                if self.done.is_set():
                    break
                continue


# Initialize Redis clients for caching and async operations
redis_client = redis.Redis.from_url(settings.REDIS_URL)
redis_async_client = redis_async.Redis.from_url(settings.REDIS_URL)

# Configure LangChain caching
langchain.llm_cache = RedisCache(redis_client)


class RateLimiter:
    """Simple rate limiter using Redis."""
    
    def __init__(self, key_prefix: str, max_requests: int, time_window: int):
        self.key_prefix = key_prefix
        self.max_requests = max_requests
        self.time_window = time_window
    
    async def acquire(self) -> bool:
        """
        Try to acquire a rate limit token.
        Returns True if acquired, False if rate limited.
        """
        now = datetime.utcnow().timestamp()
        key = f"{self.key_prefix}:{int(now / self.time_window)}"
        
        async with redis_async_client.pipeline() as pipe:
            try:
                # Increment counter and set expiry
                await pipe.incr(key)
                await pipe.expire(key, self.time_window)
                result = await pipe.execute()
                
                current_count = result[0]
                return current_count <= self.max_requests
            except:
                # If Redis fails, allow the request
                return True


# Create rate limiter for Gemini API
gemini_limiter = RateLimiter(
    key_prefix="gemini_rate_limit",
    max_requests=60,  # Adjust based on your quota
    time_window=60  # per minute
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError))
)
@lru_cache()
def get_llm(streaming: bool = False, fallback: bool = False) -> BaseChatModel:
    """
    Create and return a Google Gemini LLM instance with fallback options.
    Uses lru_cache to ensure only one LLM client is created per configuration.
    
    Args:
        streaming: Whether to enable streaming mode
        fallback: Whether to use fallback model (if primary fails)
        
    Returns:
        LangChain chat model instance
    """
    try:
        logger.info("Creating Google Gemini LLM client")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            streaming=streaming,
            retry_on_failure=True,
        )
        return llm
    except Exception as e:
        if not fallback:
            logger.warning(f"Primary LLM failed, trying fallback: {e}")
            return get_llm(streaming=streaming, fallback=True)
        logger.error(f"Failed to create LLM client: {e}", exc_info=True)
        raise


async def generate_stream(
    messages: List[BaseMessage],
    temperature: Optional[float] = None
) -> AsyncIterator[str]:
    """
    Generate a streaming response from the LLM.
    
    Args:
        messages: List of chat messages
        temperature: Optional temperature override
        
    Yields:
        Chunks of the generated response
    """
    callback_handler = AsyncIteratorCallbackHandler()
    
    try:
        # Check rate limit
        if not await gemini_limiter.acquire():
            raise Exception("Rate limit exceeded")
        
        # Get streaming LLM
        llm = get_llm(streaming=True)
        
        # Override temperature if provided
        if temperature is not None:
            llm = llm.bind(temperature=temperature)
        
        # Start generation
        task = asyncio.create_task(
            llm.agenerate(
                messages=[messages],
                callbacks=[callback_handler]
            )
        )
        
        # Stream the response
        async for token in callback_handler.aiter():
            yield token
            
        # Ensure generation completed
        await task
        
    except Exception as e:
        logger.error(f"Error in streaming generation: {e}", exc_info=True)
        raise
    finally:
        await callback_handler.done.set()


async def batch_generate(
    message_lists: List[List[BaseMessage]],
    max_concurrent: int = 5
) -> List[str]:
    """
    Generate responses for multiple message lists concurrently.
    
    Args:
        message_lists: List of message lists to process
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        List of generated responses
    """
    async def process_messages(messages: List[BaseMessage]) -> str:
        try:
            if not await gemini_limiter.acquire():
                raise Exception("Rate limit exceeded")
            
            llm = get_llm()
            response = await llm.agenerate(messages=[messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            return str(e)
    
    # Process in batches
    results = []
    for i in range(0, len(message_lists), max_concurrent):
        batch = message_lists[i:i + max_concurrent]
        batch_results = await asyncio.gather(
            *[process_messages(messages) for messages in batch]
        )
        results.extend(batch_results)
    
    return results


async def generate_structured_output(
    messages: List[BaseMessage],
    output_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a structured JSON response from the LLM.
    
    Args:
        messages: List of chat messages
        output_schema: JSON schema for the expected output
        
    Returns:
        Structured response matching the schema
    """
    try:
        # Create JSON output parser
        parser = JsonOutputParser(pydantic_schema=output_schema)
        
        # Add format instructions to the last message
        last_message = messages[-1]
        format_instructions = parser.get_format_instructions()
        
        if isinstance(last_message.content, str):
            new_content = f"{last_message.content}\n\n{format_instructions}"
            messages[-1] = last_message.__class__(content=new_content)
        
        # Generate response
        llm = get_llm()
        response = await llm.agenerate(messages=[messages])
        text = response.generations[0][0].text
        
        # Parse response
        return parser.parse(text)
    except Exception as e:
        logger.error(f"Error generating structured output: {e}", exc_info=True)
        raise 