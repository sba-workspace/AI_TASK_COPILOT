"""
Notion API service for AI Task Copilot.
"""
from typing import Dict, Any, Optional, List
from functools import lru_cache

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
        
        # Retrieve page metadata
        page = client.pages.retrieve(page_id=page_id)
        
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
        blocks = client.blocks.children.list(block_id=page_id)
        
        # Extract text content from blocks
        content = await _extract_blocks_content(blocks.get("results", []))
        
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
        
        # Check if parent_id is a database or page
        is_database = False
        try:
            client.databases.retrieve(database_id=parent_id)
            is_database = True
        except:
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
        
        # Create the page
        response = client.pages.create(
            parent=parent,
            properties=properties,
        )
        
        page_id = response["id"]
        
        # Add content as blocks
        blocks = _convert_text_to_blocks(content)
        client.blocks.children.append(
            block_id=page_id,
            children=blocks
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
    content = []
    
    for block in blocks:
        block_type = block.get("type")
        
        if block_type == "paragraph":
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("paragraph", {}).get("rich_text", [])])
            if text:
                content.append(text)
                
        elif block_type == "heading_1":
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("heading_1", {}).get("rich_text", [])])
            if text:
                content.append(f"# {text}")
                
        elif block_type == "heading_2":
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("heading_2", {}).get("rich_text", [])])
            if text:
                content.append(f"## {text}")
                
        elif block_type == "heading_3":
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("heading_3", {}).get("rich_text", [])])
            if text:
                content.append(f"### {text}")
                
        elif block_type == "bulleted_list_item":
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("bulleted_list_item", {}).get("rich_text", [])])
            if text:
                content.append(f"• {text}")
                
        elif block_type == "numbered_list_item":
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("numbered_list_item", {}).get("rich_text", [])])
            if text:
                content.append(f"1. {text}")  # Not preserving actual numbers, just listing as 1.
                
        elif block_type == "to_do":
            checked = block.get("to_do", {}).get("checked", False)
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("to_do", {}).get("rich_text", [])])
            if text:
                checkbox = "[x]" if checked else "[ ]"
                content.append(f"{checkbox} {text}")
                
        elif block_type == "code":
            language = block.get("code", {}).get("language", "")
            text = "".join([text_obj.get("plain_text", "") for text_obj in block.get("code", {}).get("rich_text", [])])
            if text:
                content.append(f"```{language}\n{text}\n```")
                
        # Handle child blocks recursively if they exist
        if "has_children" in block and block["has_children"]:
            try:
                child_blocks = get_notion_client().blocks.children.list(block_id=block["id"]).get("results", [])
                child_content = await _extract_blocks_content(child_blocks)
                content.append(child_content)
            except Exception as e:
                logger.error(f"Error getting child blocks: {e}", exc_info=True)
    
    return "\n\n".join(content)


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
        
        # Retrieve database metadata
        database = client.databases.retrieve(database_id=database_id)
        
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
        logger.error(f"Error getting Notion database: {e}", exc_info=True)
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
        
        # Prepare query parameters
        query_params = {
            "database_id": database_id,
            "page_size": page_size
        }
        
        if filter_conditions:
            query_params["filter"] = filter_conditions
            
        if sorts:
            query_params["sorts"] = sorts
        
        # Query the database
        response = client.databases.query(**query_params)
        
        # Process results
        pages = []
        for page in response.get("results", []):
            # Extract page data
            page_id = page["id"]
            
            # Get page content
            try:
                page_data = await get_page(page_id)
                pages.append(page_data)
            except Exception as e:
                logger.error(f"Error getting page {page_id}: {e}")
                continue
        
        return pages
    except Exception as e:
        logger.error(f"Error querying Notion database: {e}", exc_info=True)
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
        
        # Create database
        response = client.databases.create(
            parent={"page_id": parent_page_id},
            title=[{
                "type": "text",
                "text": {"content": title}
            }],
            description=[{
                "type": "text",
                "text": {"content": description}
            }],
            properties=properties
        )
        
        return NotionDatabase(
            id=response["id"],
            title=title,
            description=description,
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
        
        # Search for pages the integration has access to
        response = client.search(filter={"property": "object", "value": "page"})
        pages = response.get("results", [])
        
        # Get details for each page
        synced_pages = []
        errors = []
        
        for page in pages:
            try:
                page_id = page["id"]
                page_data = await get_page(page_id)
                synced_pages.append(page_data)
                logger.debug(f"Synced Notion page: {page_data.title}")
            except Exception as e:
                logger.error(f"Error syncing Notion page {page['id']}: {e}")
                errors.append({"page_id": page["id"], "error": str(e)})
        
        # TODO: Add code to embed and store the synced pages in a vector store
        # This would typically call a function like:
        # await store_notion_data(user_id, synced_pages)
        
        return {
            "status": "success",
            "user_id": user_id,
            "pages_synced": len(synced_pages),
            "errors": len(errors),
            "error_details": errors if errors else None
        }
    except Exception as e:
        logger.error(f"Error syncing Notion data for user {user_id}: {e}", exc_info=True)
        raise 