"""
Task database operations.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.db.models import Task, TaskStatus
from app.db.supabase import get_supabase_client
from app.core.logging import logger


# Supabase table name
TASKS_TABLE = "tasks"


async def create_task(
    task_id: str, 
    user_id: str, 
    description: str,
    agent_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Task:
    """
    Create a new task in the database.
    
    Args:
        task_id: Unique ID for the task
        user_id: ID of the user who created the task
        description: Task description
        agent_type: Type of agent used for the task
        metadata: Additional task metadata
        
    Returns:
        The created task
    """
    try:
        supabase = get_supabase_client()
        
        now = datetime.utcnow()
        task_data = {
            "id": task_id,
            "user_id": user_id,
            "description": description,
            "status": TaskStatus.PENDING.value,
            "agent_type": agent_type,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "metadata": metadata or {},
        }
        
        # Insert task into Supabase
        response = supabase.table(TASKS_TABLE).insert(task_data).execute()
        
        if response.data:
            logger.info(f"Created task {task_id} for user {user_id}")
            return Task(**response.data[0])
        else:
            logger.error(f"Failed to create task: {response.error}")
            raise Exception(f"Failed to create task: {response.error}")
    
    except Exception as e:
        logger.error(f"Error creating task: {e}", exc_info=True)
        raise


async def update_task_status(
    task_id: str,
    status: TaskStatus,
    result: Optional[str] = None
) -> Task:
    """
    Update a task's status and result.
    
    Args:
        task_id: ID of the task to update
        status: New task status
        result: Task result (if completed)
        
    Returns:
        The updated task
    """
    try:
        supabase = get_supabase_client()
        
        # Prepare update data
        update_data = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Add result if provided
        if result is not None:
            update_data["result"] = result
            
        # Update task in Supabase
        response = supabase.table(TASKS_TABLE).update(update_data).eq("id", task_id).execute()
        
        if response.data:
            logger.info(f"Updated task {task_id} status to {status.value}")
            return Task(**response.data[0])
        else:
            logger.error(f"Failed to update task: {response.error}")
            raise Exception(f"Failed to update task: {response.error}")
    
    except Exception as e:
        logger.error(f"Error updating task status: {e}", exc_info=True)
        raise


async def update_task(
    task_id: str,
    updates: Dict[str, Any]
) -> Task:
    """
    Update a task with arbitrary fields.
    
    Args:
        task_id: ID of the task to update
        updates: Dictionary of fields to update
        
    Returns:
        The updated task
    """
    try:
        supabase = get_supabase_client()
        
        # Add updated_at timestamp
        update_data = {
            **updates,
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Convert any enum values to their string representation
        for key, value in update_data.items():
            if hasattr(value, "value"):
                update_data[key] = value.value
        
        # Update task in Supabase
        response = supabase.table(TASKS_TABLE).update(update_data).eq("id", task_id).execute()
        
        if response.data:
            logger.info(f"Updated task {task_id}")
            return Task(**response.data[0])
        else:
            logger.error(f"Failed to update task: {response.error}")
            raise Exception(f"Failed to update task: {response.error}")
    
    except Exception as e:
        logger.error(f"Error updating task: {e}", exc_info=True)
        raise


async def delete_task(task_id: str) -> None:
    """
    Delete a task from the database.
    
    Args:
        task_id: ID of the task to delete
    """
    try:
        supabase = get_supabase_client()
        
        # Delete task from Supabase
        response = supabase.table(TASKS_TABLE).delete().eq("id", task_id).execute()
        
        if response.error:
            logger.error(f"Failed to delete task: {response.error}")
            raise Exception(f"Failed to delete task: {response.error}")
            
        logger.info(f"Deleted task {task_id}")
    
    except Exception as e:
        logger.error(f"Error deleting task: {e}", exc_info=True)
        raise


async def get_task(task_id: str) -> Optional[Task]:
    """
    Get a task by ID.
    
    Args:
        task_id: ID of the task to retrieve
        
    Returns:
        The task if found, None otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Query task from Supabase
        response = supabase.table(TASKS_TABLE).select("*").eq("id", task_id).execute()
        
        if response.data:
            return Task(**response.data[0])
        else:
            logger.info(f"Task {task_id} not found")
            return None
    
    except Exception as e:
        logger.error(f"Error getting task: {e}", exc_info=True)
        raise


async def get_user_tasks(user_id: str, limit: int = 10) -> List[Task]:
    """
    Get a user's recent tasks.
    
    Args:
        user_id: ID of the user
        limit: Maximum number of tasks to return
        
    Returns:
        List of user tasks
    """
    try:
        supabase = get_supabase_client()
        
        # Query tasks from Supabase
        response = supabase.table(TASKS_TABLE)\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        if response.data:
            return [Task(**task) for task in response.data]
        else:
            return []
    
    except Exception as e:
        logger.error(f"Error getting user tasks: {e}", exc_info=True)
        raise


async def get_tasks_by_status(user_id: str, status: TaskStatus) -> List[Task]:
    """
    Get a user's tasks by status.
    
    Args:
        user_id: ID of the user
        status: Task status to filter by
        
    Returns:
        List of tasks with the specified status
    """
    try:
        supabase = get_supabase_client()
        
        # Query tasks from Supabase
        response = supabase.table(TASKS_TABLE)\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("status", status.value)\
            .order("created_at", desc=True)\
            .execute()
        
        if response.data:
            return [Task(**task) for task in response.data]
        else:
            return []
    
    except Exception as e:
        logger.error(f"Error getting tasks by status: {e}", exc_info=True)
        raise 