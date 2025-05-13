"""
Main FastAPI application entry point.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI app.
    Initialize resources at startup and clean up at shutdown.
    """
    logger.info("Starting AI Task Copilot application")
    # Initialize agent, clients, and connections here
    # e.g., initialize Weaviate client, Supabase client, etc.
    yield
    # Clean up resources here
    logger.info("Shutting down AI Task Copilot application")


# Create FastAPI app
app = FastAPI(
    title="AI Task Copilot",
    description="Agentic assistant for Notion, Slack, and GitHub using LLMs",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to AI Task Copilot API", "status": "healthy"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Include API routers
app.include_router(api_router, prefix=settings.API_PREFIX)


