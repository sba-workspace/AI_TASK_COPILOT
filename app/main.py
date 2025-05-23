"""
Main FastAPI application entry point.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.core.logging import logger
from app.api.deps import get_current_user, User


# Security scheme
security_scheme = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI app.
    Initialize resources at startup and clean up at shutdown.
    """
    logger.info("Starting AI Task Copilot application")
    # Initialize agent, clients, and connections here
    # e.g., initialize Weaviate client, Supabase client, etc.
    
    # Initialize Supabase realtime subscription (if needed)
    # This would be the place to initialize any global realtime subscriptions
    
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

# Custom OpenAPI to use HTTP Bearer security scheme
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add HTTPBearer security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security to all routes except auth login/register and WebSockets
    for path in openapi_schema["paths"]:
        # Skip login, register, and WebSocket routes
        if (not path.endswith("/login") and 
            not path.endswith("/register") and 
            not path.endswith("/login-page") and 
            not "websocket" in path.lower() and
            path != "/health" and 
            path != "/"):
            
            for method in openapi_schema["paths"][path]:
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Determine CORS origins
frontend_origin = settings.FRONTEND_URL
cors_origins = [frontend_origin]
if settings.APP_ENV == "development":
    # Add additional development origins
    cors_origins.extend([
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ])
elif settings.CORS_ORIGINS and settings.CORS_ORIGINS != ["*"]:
    cors_origins.extend(settings.CORS_ORIGINS)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Client-Info"],
    expose_headers=["Content-Length"],
    max_age=600,  # 10 minutes
)

# Include API router
app.include_router(api_router, prefix=settings.API_PREFIX)


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

