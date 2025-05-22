"""
Task endpoints for the AI Task Copilot API.
"""
from typing import Optional, List
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.logging import logger
from app.agent.router import get_router_agent
from app.api.deps import get_current_user
from app.db.models import Task, TaskStatus
from app.db import tasks as task_db


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


class TaskListResponse(BaseModel):
    """Task list response model."""
    tasks: List[TaskResponse]


@router.post("/run-task", response_model=TaskResponse)
async def run_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Run a task using the AI agent."""
    try:
        logger.info(f"Running task for user {user.id}: {request.description}")
        
        # Get the router agent
        router = get_router_agent()
        
        # Generate a task ID
        task_id = f"task_{user.id}_{hash(request.description)}"
        
        # Determine which agent to use (without executing yet)
        agent_type = await router.analyze_query(request.description)
        logger.info(f"Selected agent type: {agent_type}")
        
        # Create task in database
        await task_db.create_task(
            task_id=task_id,
            user_id=user.id,
            description=request.description,
            agent_type=agent_type
        )
        
        # Update task status to in_progress
        await task_db.update_task_status(task_id, TaskStatus.IN_PROGRESS)
        
        # Execute task in background
        background_tasks.add_task(
            process_task_in_background,
            task_id=task_id,
            input_text=request.description,
            router=router
        )
        
        # Return task ID immediately
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS.value,
            result=None,
        )
        
    except Exception as e:
        logger.error(f"Error processing task request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process task: {str(e)}",
        )


async def process_task_in_background(task_id: str, input_text: str, router):
    """
    Process a task in the background.
    
    Args:
        task_id: ID of the task
        input_text: User input text
        router: Agent router
    """
    try:
        logger.info(f"Processing task {task_id} in background")
        
        # Route and execute the task
        result = await router.route_and_execute({"input": input_text})
        
        # Extract the output
        output = result.get("output", "I couldn't complete this task.")
        
        # Update task in database
        await task_db.update_task_status(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=output
        )
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing task in background: {e}", exc_info=True)
        
        # Update task status to failed
        try:
            await task_db.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                result=f"Task failed: {str(e)}"
            )
        except Exception as update_error:
            logger.error(f"Error updating task status: {update_error}", exc_info=True)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    user=Depends(get_current_user)
):
    """Get the status of a task."""
    try:
        logger.info(f"Fetching task status for user {user.id}: {task_id}")
        
        # Check if task belongs to user
        if not task_id.startswith(f"task_{user.id}_"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this task",
            )
            
        # Get task from database
        task = await task_db.get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found",
            )
            
        # Convert to response model
        return TaskResponse(
            task_id=task.id,
            status=task.status,
            result=task.result,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching task status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch task status: {str(e)}",
        )


@router.get("/", response_model=TaskListResponse)
async def list_user_tasks(
    limit: int = 10,
    user=Depends(get_current_user)
):
    """List recent tasks for a user."""
    try:
        logger.info(f"Listing tasks for user {user.id}")
        
        # Get tasks from database
        tasks_list = await task_db.get_user_tasks(user.id, limit=limit)
        
        # Convert to response model
        return TaskListResponse(
            tasks=[
                TaskResponse(
                    task_id=task.id,
                    status=task.status,
                    result=task.result,
                )
                for task in tasks_list
            ]
        )
        
    except Exception as e:
        logger.error(f"Error listing user tasks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}",
        )