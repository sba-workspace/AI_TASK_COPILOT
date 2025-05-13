"""
Task endpoints for the AI Task Copilot API.
"""
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.logging import logger
from app.agent.agent import get_agent
from app.api.deps import get_current_user

router = APIRouter()


class TaskRequest(BaseModel):
    """Task request model."""
    description: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Create a Notion doc for this GitHub issue #123"
            }
        }


class TaskResponse(BaseModel):
    """Task response model."""
    task_id: str
    status: str
    result: Optional[str] = None


@router.post("/run-task", response_model=TaskResponse)
async def run_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Run a task using the AI agent."""
    try:
        logger.info(f"Running task for user {user.id}: {request.description}")
        
        # Get the agent
        agent = get_agent()
        
        # Run the agent in the background for longer tasks
        # For quick tasks, you can await the result directly
        task_id = f"task_{user.id}_{hash(request.description)}"
        
        # Execute the agent synchronously for quick response
        try:
            result = await agent.arun(request.description)
            return TaskResponse(
                task_id=task_id,
                status="completed",
                result=result,
            )
        except Exception as e:
            logger.error(f"Error running task: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task execution failed: {str(e)}",
            )
            
    except Exception as e:
        logger.error(f"Error processing task request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process task: {str(e)}",
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    user=Depends(get_current_user)
):
    """Get the status of a task."""
    # In a real implementation, you'd fetch the task status from a database
    # For now, we'll just return a placeholder response
    try:
        logger.info(f"Fetching task status for user {user.id}: {task_id}")
        
        # Check if task belongs to user (in a real implementation)
        if not task_id.startswith(f"task_{user.id}_"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this task",
            )
            
        # Placeholder for fetching actual task status
        return TaskResponse(
            task_id=task_id,
            status="in_progress",
            result=None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching task status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch task status: {str(e)}",
        )