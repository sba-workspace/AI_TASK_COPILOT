"""
Notion tools for the LangChain agent.
"""
from typing import Optional
from pydantic import BaseModel, Field

from langchain.tools import StructuredTool

from app.services.notion import get_page, create_page
from app.core.logging import logger


class GetNotionPageInput(BaseModel):
    """Input for getting a Notion page."""
    page_id: str = Field(..., description="The ID of the Notion page to retrieve")


class CreateNotionPageInput(BaseModel):
    """Input for creating a Notion page."""
    parent_id: str = Field(..., description="The ID of the parent page or database")
    title: str = Field(..., description="The title of the new page")
    content: str = Field(..., description="The content to add to the page")


async def get_notion_page(page_id: str) -> str:
    """
    Get a Notion page by ID.
    
    Args:
        page_id: The ID of the Notion page
        
    Returns:
        The page content as text
    """
    try:
        page = await get_page(page_id)
        return f"Title: {page.title}\n\n{page.content}"
    except Exception as e:
        logger.error(f"Error in get_notion_page tool: {e}", exc_info=True)
        return f"Error getting Notion page: {str(e)}"


async def create_notion_page(parent_id: str, title: str, content: str) -> str:
    """
    Create a new Notion page.
    
    Args:
        parent_id: The ID of the parent page or database
        title: The title of the new page
        content: The content to add to the page
        
    Returns:
        The URL of the created page
    """
    try:
        page_id = await create_page(parent_id, title, content)
        return f"Created Notion page: https://notion.so/{page_id.replace('-', '')}"
    except Exception as e:
        logger.error(f"Error in create_notion_page tool: {e}", exc_info=True)
        return f"Error creating Notion page: {str(e)}"


# Create LangChain tools
get_notion_page_tool = StructuredTool.from_function(
    func=get_notion_page,
    name="get_notion_page",
    description="Get the content of a Notion page by its ID",
    args_schema=GetNotionPageInput,
)

create_notion_page_tool = StructuredTool.from_function(
    func=create_notion_page,
    name="create_notion_page",
    description="Create a new Notion page with the given title and content",
    args_schema=CreateNotionPageInput,
) 