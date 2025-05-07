import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
@patch("app.api.deps.get_current_user")
@patch("app.agent.agent.get_agent")
async def test_run_task(mock_get_agent, mock_get_current_user):
    """Test run task endpoint."""
    # Mock user
    mock_user = MagicMock()
    mock_user.id = "test-user-id"
    mock_user.email = "test@example.com"
    mock_get_current_user.return_value = mock_user
    
    # Mock agent
    mock_agent = MagicMock()
    mock_agent.arun.return_value = "Task completed successfully"
    mock_get_agent.return_value = mock_agent
    
    # Test API call
    response = client.post(
        "/api/v1/run-task", 
        json={"description": "Test task"}
    )
    
    assert response.status_code == 200
    assert "task_id" in response.json()
    assert response.json()["status"] == "completed"
    assert response.json()["result"] == "Task completed successfully"
    
    # Verify agent was called with the task description
    mock_agent.arun.assert_called_once_with("Test task")
