"""
API endpoint routers.
"""
from fastapi import APIRouter

from app.api.endpoints.auth import router as auth_router
from app.api.endpoints.tasks import router as tasks_router
from app.api.endpoints.tools import router as tools_router
from app.api.endpoints.chat import router as chat_router

# Create main API router
router = APIRouter()

# Include sub-routers
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
router.include_router(tasks_router, prefix="/tasks", tags=["Tasks"])
router.include_router(tools_router, prefix="/tools", tags=["Tools"])
router.include_router(chat_router, prefix="/chat", tags=["Chat"])
