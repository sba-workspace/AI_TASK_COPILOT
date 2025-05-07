"""
Tool-related endpoints for the AI Task Copilot API.
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.core.logging import logger
from app.api.deps import get_current_user
from app.services.notion import sync_notion_data
from app.services.github import sync_github_data
from app.services.slack import sync_slack_data

router = APIRouter()


@router.post("/notion-sync")
async def trigger_notion_sync(
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Trigger synchronization of Notion data to the vector store."""
    try:
        logger.info(f"Triggering Notion sync for user {user.id}")
        background_tasks.add_task(sync_notion_data, user_id=user.id)
        return {"status": "success", "message": "Notion sync initiated"}
    except Exception as e:
        logger.error(f"Error triggering Notion sync: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate Notion sync",
        )


@router.post("/github-sync")
async def trigger_github_sync(
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Trigger synchronization of GitHub data to the vector store."""
    try:
        logger.info(f"Triggering GitHub sync for user {user.id}")
        background_tasks.add_task(sync_github_data, user_id=user.id)
        return {"status": "success", "message": "GitHub sync initiated"}
    except Exception as e:
        logger.error(f"Error triggering GitHub sync: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate GitHub sync",
        )


@router.post("/slack-sync")
async def trigger_slack_sync(
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Trigger synchronization of Slack data to the vector store."""
    try:
        logger.info(f"Triggering Slack sync for user {user.id}")
        background_tasks.add_task(sync_slack_data, user_id=user.id)
        return {"status": "success", "message": "Slack sync initiated"}
    except Exception as e:
        logger.error(f"Error triggering Slack sync: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate Slack sync",
        )