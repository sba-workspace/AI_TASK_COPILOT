"""
Realtime endpoints for WebSockets communication.
"""
from typing import List, Dict, Any
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
import asyncio

from app.api.deps import get_current_user_ws, User
from app.db.supabase import get_supabase_client
from app.db.models import Task
from app.db import tasks as task_db
from app.core.logging import logger

router = APIRouter()

# Store active connections (websocket -> user_id)
active_connections: Dict[WebSocket, str] = {}


@router.websocket("/tasks-feed")
async def websocket_task_feed(
    websocket: WebSocket,
    token: str = Query(None),
):
    """WebSocket endpoint for real-time task status updates."""
    # Authenticate user
    try:
        # Get current user using the token
        user = await get_current_user_ws(token)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return
            
        # Accept connection
        await websocket.accept()
        
        # Store user connection
        active_connections[websocket] = user.id
        
        # Send initial task list
        tasks = await task_db.get_user_tasks(user.id, limit=20)
        await send_tasks_list(websocket, tasks)
        
        # Set up Supabase realtime subscription
        supabase = get_supabase_client()
        
        # Subscribe to changes in the tasks table for this user
        channel = supabase.channel("task-updates")
        
        channel.on(
            "postgres_changes",
            event="*",
            schema="public",
            table="tasks",
            filter=f"user_id=eq.{user.id}",
            callback=lambda payload: asyncio.create_task(handle_task_update(websocket, payload))
        )
        
        # Start channel
        channel.subscribe()
        
        # Keep connection alive until client disconnects
        try:
            while True:
                # Check for client messages
                data = await websocket.receive_text()
                
                # Process client messages if needed (e.g., requesting refresh)
                if data == "refresh":
                    tasks = await task_db.get_user_tasks(user.id, limit=20)
                    await send_tasks_list(websocket, tasks)
                    
        except WebSocketDisconnect:
            # Remove from active connections
            if websocket in active_connections:
                del active_connections[websocket]
            
            # Unsubscribe from Supabase realtime
            channel.unsubscribe()
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        # Close connection on error
        if websocket.client_state.CONNECTED:
            await websocket.close(code=1011, reason="Server error")


async def handle_task_update(websocket: WebSocket, payload: Dict[str, Any]):
    """
    Handle task update notification from Supabase.
    
    Args:
        websocket: WebSocket connection
        payload: Update payload from Supabase
    """
    try:
        # Get updated task
        if payload.get("new") and payload["new"].get("id"):
            task_id = payload["new"]["id"]
            task = await task_db.get_task(task_id)
            
            if task:
                # Send task update to client
                await send_task_update(websocket, task)
    except Exception as e:
        logger.error(f"Error handling task update: {e}", exc_info=True)


async def send_tasks_list(websocket: WebSocket, tasks: List[Task]):
    """
    Send a list of tasks to the WebSocket client.
    
    Args:
        websocket: WebSocket connection
        tasks: List of tasks to send
    """
    try:
        tasks_json = [
            {
                "task_id": task.id,
                "description": task.description,
                "status": task.status,
                "result": task.result,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            }
            for task in tasks
        ]
        
        await websocket.send_json({
            "type": "tasks_list",
            "data": tasks_json
        })
    except Exception as e:
        logger.error(f"Error sending tasks list: {e}", exc_info=True)


async def send_task_update(websocket: WebSocket, task: Task):
    """
    Send a task update to the WebSocket client.
    
    Args:
        websocket: WebSocket connection
        task: The task to send
    """
    try:
        task_json = {
            "task_id": task.id,
            "description": task.description,
            "status": task.status,
            "result": task.result,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        }
        
        await websocket.send_json({
            "type": "task_update",
            "data": task_json
        })
    except Exception as e:
        logger.error(f"Error sending task update: {e}", exc_info=True) 