"""
Database models for the application.
"""
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """Task database model."""
    id: str
    user_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    agent_type: Optional[str] = None  # "simple" or "langraph"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None 