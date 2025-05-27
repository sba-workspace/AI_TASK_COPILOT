"""
Notion API service for AI Task Copilot.
"""
from typing import Dict, Any, Optional, List
from functools import lru_cache
import asyncio

from notion_client import Client
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger


class NotionPage(BaseModel):
    """Notion page model with basic properties."""
    id: str
    title: str
    content: str
    url: str
    last_updated: str


class NotionDatabase(BaseModel):
    """Notion database model."""
    id: str
    title: str
    description: str
    url: str
    properties: Dict[str, Any]


@lru_cache()
def get_notion_client() -> Client:
    """
    Create and return a Notion client instance.
    Uses lru_cache to ensure only one client is created.
    """
    try:
        logger.debug("Creating Notion client")
        client = Client(auth=settings.NOTION_API_KEY)
        return client
    except Exception as e:
        logger.error(f"Failed to create Notion client: {e}", exc_info=True)
        raise


async def get_page(page_id: str) -> NotionPage:
    """
    Get a Notion page by ID.
    
    Args:
        page_id: The ID of the Notion page
        
    Returns:
        NotionPage object with page data
    """
    try:
        client = get_notion_client()
        loop = asyncio.get_running_loop()
        
        # Retrieve page metadata
        page = await loop.run_in_executor(None, client.pages.retrieve, page_id)
        
        # Extract title
        title = ""
        if "properties" in page and "title" in page["properties"]:
            title_obj = page["properties"]["title"]
            if "title" in title_obj and title_obj["title"]:
                title = "".join([text_obj.get("plain_text", "") for text_obj in title_obj["title"]])
        
        # Get page URL
        url = f"https://notion.so/{page_id.replace('-', '')}"
        
        # Get last updated time
        last_updated = page.get("last_edited_time", "")
        
        # Retrieve page blocks (content)
        blocks_response = await loop.run_in_executor(None, client.blocks.children.list, page_id)
        
        # Extract text content from blocks
        content = await _extract_blocks_content(blocks_response.get("results", []))
        
        return NotionPage(
            id=page_id,
            title=title,
            content=content,
            url=url,
            last_updated=last_updated
        )
    except Exception as e:
        logger.error(f"Error getting Notion page: {e}", exc_info=True)
        raise


async def create_page(parent_id: str, title: str, content: str) -> str:
    """
    Create a new Notion page.
    
    Args:
        parent_id: The ID of the parent page or database
        title: The title of the new page
        content: The content to add to the page
        
    Returns:
        The ID of the created page
    """
    try:
        client = get_notion_client()
        loop = asyncio.get_running_loop()

        # Check if parent_id is a database or page
        is_database = False
        try:
            # Offload the synchronous DB retrieval
            await loop.run_in_executor(None, client.databases.retrieve, parent_id) # Use actual parent_id
            is_database = True
        except Exception: # Catch specific Notion exceptions if possible, e.g., APIResponseError
            # If not a database, assume it's a page
            pass
        
        # Create page properties based on parent type
        if is_database:
            # Database item requires properties matching the database schema
            # For simplicity, we assume a 'Name' property exists
            properties = {
                "Name": {
                    "title": [
                        {
                            "text": {
                                "content": title
                            }
                        }
                    ]
                }
            }
            parent = {"database_id": parent_id}
        else:
            # Page as parent is simpler
            properties = {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            }
            parent = {"page_id": parent_id}
        
        # Offload the synchronous page creation
        response = await loop.run_in_executor(
            None, 
            lambda: client.pages.create(parent=parent, properties=properties)
        )
        
        page_id = response["id"]
        
        # Add content as blocks - _convert_text_to_blocks is synchronous
        # client.blocks.children.append is also synchronous
        blocks_data = _convert_text_to_blocks(content)
        if blocks_data: # Only append if there are blocks to add
            await loop.run_in_executor(
                None,
                lambda: client.blocks.children.append(block_id=page_id, children=blocks_data)
            )
        
        return page_id
    except Exception as e:
        logger.error(f"Error creating Notion page: {e}", exc_info=True)
        raise


async def _extract_blocks_content(blocks: list) -> str:
    """
    Extract text content from Notion blocks.
    
    Args:
        blocks: List of Notion blocks
        
    Returns:
        Extracted text content
    """
    content_parts = []
    loop = asyncio.get_running_loop()

    for block in blocks:
        block_type = block.get("type")
        text_content = ""
        
        if block_type == "paragraph":
            text_content = "".join([text_obj.get("plain_text", "") for text_obj in block.get("paragraph", {}).get("rich_text", [])])
        elif block_type == "heading_1":
            text_content = f"# {"".join([text_obj.get("plain_text", "") for text_obj in block.get("heading_1", {}).get("rich_text", [])])}"
        elif block_type == "heading_2":
            text_content = f"## {"".join([text_obj.get("plain_text", "") for text_obj in block.get("heading_2", {}).get("rich_text", [])])}"
        elif block_type == "heading_3":
            text_content = f"### {"".join([text_obj.get("plain_text", "") for text_obj in block.get("heading_3", {}).get("rich_text", [])])}"
        elif block_type == "bulleted_list_item":
            text_content = f"• {"".join([text_obj.get("plain_text", "") for text_obj in block.get("bulleted_list_item", {}).get("rich_text", [])])}"
        elif block_type == "numbered_list_item":
            text_content = f"1. {"".join([text_obj.get("plain_text", "") for text_obj in block.get("numbered_list_item", {}).get("rich_text", [])])}"
        elif block_type == "to_do":
            checked = block.get("to_do", {}).get("checked", False)
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("to_do", {}).get("rich_text", [])])
            checkbox = "[x]" if checked else "[ ]"
            text_content = f"{checkbox} {text}"
        elif block_type == "code":
            language = block.get("code", {}).get("language", "")
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("code", {}).get("rich_text", [])])
            text_content = f"```{language}\n{text}\n```"
        
        if text_content.strip():
            content_parts.append(text_content)

        if block.get("has_children"):
            try:
                client = get_notion_client()
                child_blocks_response = await loop.run_in_executor(None, client.blocks.children.list, block["id"])
                child_content = await _extract_blocks_content(child_blocks_response.get("results", []))
                if child_content.strip():
                    content_parts.append(child_content)
            except Exception as e:
                logger.error(f"Error getting child blocks for {block['id']}: {e}", exc_info=True)
    
    return "\n\n".join(filter(None, content_parts))


def _convert_text_to_blocks(content: str) -> list:
    """
    Convert plain text content to Notion blocks.
    This is a simplified implementation that handles common formatting.
    
    Args:
        content: Text content to convert
        
    Returns:
        List of Notion blocks
    """
    lines = content.split("\n")
    blocks = []
    
    current_block = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            # Add a separator for empty lines
            if current_block:
                blocks.append(current_block)
                current_block = None
            continue
        
        # Check for heading patterns
        if line.startswith("# "):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                }
            })
        elif line.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": line[3:]}}]
                }
            })
        elif line.startswith("### "):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": line[4:]}}]
                }
            })
        # Check for list patterns
        elif line.startswith("• ") or line.startswith("- "):
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                }
            })
        elif line.startswith("1. ") or (len(line) > 2 and line[0].isdigit() and line[1] == "." and line[2] == " "):
            # Extract the text after the number
            text = line[line.find(" ")+1:]
            blocks.append({
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                }
            })
        # Check for checkbox patterns
        elif line.startswith("[ ] "):
            blocks.append({
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"type": "text", "text": {"content": line[4:]}}],
                    "checked": False
                }
            })
        elif line.startswith("[x] ") or line.startswith("[X] "):
            blocks.append({
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"type": "text", "text": {"content": line[4:]}}],
                    "checked": True
                }
            })
        else:
            # Default to paragraph for any other text
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": line}}]
                }
            })
    
    # Add any remaining block
    if current_block:
        blocks.append(current_block)
    
    return blocks 


async def get_database(database_id: str) -> NotionDatabase:
    """
    Get a Notion database by ID.
    
    Args:
        database_id: The ID of the Notion database
        
    Returns:
        NotionDatabase object with database data
    """
    try:
        client = get_notion_client()
        loop = asyncio.get_running_loop()
        
        # Offload synchronous database retrieval
        database = await loop.run_in_executor(None, client.databases.retrieve, database_id)
        
        # Extract title
        title = ""
        if "title" in database:
            title = "".join([text_obj.get("plain_text", "") for text_obj in database["title"]])
        
        # Extract description
        description = ""
        if "description" in database:
            description = "".join([text_obj.get("plain_text", "") for text_obj in database["description"]])
        
        # Get database URL
        url = f"https://notion.so/{database_id.replace('-', '')}"
        
        return NotionDatabase(
            id=database_id,
            title=title,
            description=description,
            url=url,
            properties=database.get("properties", {})
        )
    except Exception as e:
        logger.error(f"Error getting Notion database {database_id}: {e}", exc_info=True)
        raise


async def query_database(
    database_id: str,
    filter_conditions: Optional[Dict] = None,
    sorts: Optional[List[Dict]] = None,
    page_size: int = 100
) -> List[NotionPage]:
    """
    Query a Notion database with optional filters and sorting.
    
    Args:
        database_id: The ID of the database to query
        filter_conditions: Optional filter conditions following Notion's filter syntax
        sorts: Optional sort conditions following Notion's sort syntax
        page_size: Maximum number of results to return
        
    Returns:
        List of NotionPage objects matching the query
    """
    try:
        client = get_notion_client()
        loop = asyncio.get_running_loop()
        
        query_params = {
            "database_id": database_id,
            "page_size": page_size
        }
        if filter_conditions:
            query_params["filter"] = filter_conditions
        if sorts:
            query_params["sorts"] = sorts
        
        # Offload synchronous database query
        response = await loop.run_in_executor(None, lambda: client.databases.query(**query_params))
        
        pages = []
        for page_summary in response.get("results", []):
            page_id = page_summary["id"]
            try:
                # get_page is already refactored
                page_data = await get_page(page_id)
                if page_data:
                    pages.append(page_data)
            except Exception as e:
                logger.error(f"Error getting page {page_id} during database query: {e}", exc_info=True)
                continue
        
        return pages
    except Exception as e:
        logger.error(f"Error querying Notion database {database_id}: {e}", exc_info=True)
        raise


async def create_database(
    parent_page_id: str,
    title: str,
    description: str,
    properties: Dict[str, Any]
) -> NotionDatabase:
    """
    Create a new Notion database.
    
    Args:
        parent_page_id: The ID of the parent page
        title: The title of the database
        description: The description of the database
        properties: The database properties schema
        
    Returns:
        NotionDatabase object for the created database
    """
    try:
        client = get_notion_client()
        loop = asyncio.get_running_loop()
        
        db_payload = {
            "parent": {"page_id": parent_page_id},
            "title": [{
                "type": "text",
                "text": {"content": title}
            }],
            "description": [{
                "type": "text",
                "text": {"content": description}
            }],
            "properties": properties
        }
        
        # Offload synchronous database creation
        response = await loop.run_in_executor(None, lambda: client.databases.create(**db_payload))
        
        return NotionDatabase(
            id=response["id"],
            title=title, # Title from input, as response might not have it in the same format
            description=description, # Description from input
            url=f"https://notion.so/{response['id'].replace('-', '')}",
            properties=response.get("properties", {})
        )
    except Exception as e:
        logger.error(f"Error creating Notion database: {e}", exc_info=True)
        raise


async def sync_notion_data(user_id: str) -> Dict[str, Any]:
    """
    Synchronize Notion data for a user to the vector store.
    This function fetches data from Notion and prepares it for embedding.
    
    Args:
        user_id: The ID of the user to sync data for
        
    Returns:
        A dictionary with sync results
    """
    try:
        logger.info(f"Starting Notion data sync for user {user_id}")
        client = get_notion_client()
        loop = asyncio.get_running_loop()
        
        # Search for pages the integration has access to
        search_params = {"filter": {"property": "object", "value": "page"}}
        response = await loop.run_in_executor(None, lambda: client.search(**search_params))
        pages_summary = response.get("results", [])
        
        logger.info(f"Found {len(pages_summary)} pages/databases summary records via search.")

        # Get details for each page
        synced_pages_data = []
        errors = []
        
        for page_summary in pages_summary:
            try:
                page_id = page_summary["id"]
                page_data = await get_page(page_id)
                if page_data:
                    synced_pages_data.append(page_data)
                    logger.debug(f"Successfully fetched Notion page: {page_data.title}")
                else:
                    logger.warning(f"Could not retrieve data for page {page_id}")
                    errors.append({"page_id": page_id, "error": "Failed to retrieve page data (get_page returned None)"})
            except Exception as e:
                logger.error(f"Error processing Notion page_summary {page_summary.get('id', 'UnknownID')}: {e}", exc_info=True)
                errors.append({"page_id": page_summary.get('id', 'UnknownID'), "error": str(e)})
        
        # TODO: Add code to embed and store the synced pages in a vector store
        # This would typically call a function like:
        # await store_notion_data_in_weaviate(user_id, synced_pages_data) # Assuming such function exists
        
        logger.info(f"Finished Notion data sync for user {user_id}. Pages processed: {len(synced_pages_data)}, Errors: {len(errors)}")
        return {
            "status": "success" if not errors else "partial_success",
            "user_id": user_id,
            "pages_synced_count": len(synced_pages_data),
            "errors_count": len(errors),
            "error_details": errors if errors else None
        }
    except Exception as e:
        logger.error(f"Critical error during Notion data sync for user {user_id}: {e}", exc_info=True)
        # It might be better to return a distinct error status or re-raise for a global error handler
        return {
            "status": "failed",
            "user_id": user_id,
            "message": str(e),
            "pages_synced_count": 0,
            "errors_count": 1, # Indicates the main sync function failed
            "error_details": [{ "error": str(e) }]
        } 