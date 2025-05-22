"""
API routes module.
"""
from fastapi import APIRouter

from app.api.endpoints import tasks, auth, tools, chat, realtime

# Create main API router
router = APIRouter()

# Include specific endpoint routers
router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(tools.router, prefix="/tools", tags=["Tools"])
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(realtime.router, prefix="/realtime", tags=["Realtime"])