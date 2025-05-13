"""
Tool-related endpoints for the AI Task Copilot API.
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, Body
from typing import List, Optional

from app.core.logging import logger
from app.api.deps import get_current_user
from app.services.notion import sync_notion_data
from app.services.github import sync_github_data
from app.services.slack import sync_slack_data
from app.tools.rag_tools import semantic_search
from pydantic import BaseModel

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


class SearchRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = None
    limit: Optional[int] = 5

@router.post("/search")
async def search(
    request: SearchRequest = Body(...),
    user=Depends(get_current_user)
):
    """Perform semantic search across data sources (Notion, GitHub, Slack)."""
    try:
        source_type = None
        if request.sources:
            # Only support one source at a time for now
            if len(request.sources) == 1:
                source_type = request.sources[0]
            else:
                # If multiple, treat as None (search all)
                source_type = None
        results = await semantic_search(
            query=request.query,
            limit=request.limit or 5,
            source_type=source_type
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in /search endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )