import pytest
from unittest.mock import patch, AsyncMock

from app.tools.notion_tools import get_notion_page, create_notion_page


@pytest.mark.asyncio
@patch("app.tools.notion_tools.get_page")
async def test_get_notion_page(mock_get_page):
    """Test get_notion_page tool."""
    # Setup mock
    mock_page = AsyncMock()
    mock_page.title = "Test Page"
    mock_page.content = "This is a test page content."
    mock_get_page.return_value = mock_page
    
    # Call the function
    result = await get_notion_page("test-page-id")
    
    # Verify the result
    assert "Test Page" in result
    assert "This is a test page content." in result
    mock_get_page.assert_called_once_with("test-page-id")


@pytest.mark.asyncio
@patch("app.tools.notion_tools.create_page")
async def test_create_notion_page(mock_create_page):
    """Test create_notion_page tool."""
    # Setup mock
    mock_create_page.return_value = "test-new-page-id"
    
    # Call the function
    result = await create_notion_page(
        parent_id="test-parent-id",
        title="New Test Page",
        content="New test page content."
    )
    
    # Verify the result
    assert "Created Notion page:" in result
    assert "test-new-page-id" in result
    mock_create_page.assert_called_once_with(
        "test-parent-id", "New Test Page", "New test page content."
    )