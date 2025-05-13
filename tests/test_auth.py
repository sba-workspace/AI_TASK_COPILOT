import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.api.deps import User, get_current_user
from app.core.config import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_dependency_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()

def test_oauth_login_github():
    """Test OAuth login with GitHub provider."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_client.auth.sign_in_with_oauth.return_value = MagicMock(
            url="https://ytopycjpelmtqrdbsxcr.supabase.co/auth/v1/authorize?redirect_to=your-frontend-url/auth/callback&provider=github"
        )
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(f"{settings.API_PREFIX}/auth/oauth", json={"provider": "github"})
        
        assert response.status_code == 200
        assert "url" in response.json()
        assert "supabase" in response.json()["url"]
        assert "github" in response.json()["url"]

def test_oauth_login_invalid_provider():
    """Test OAuth login with invalid provider."""
    response = client.post(f"{settings.API_PREFIX}/auth/oauth", json={"provider": "invalid"})
    
    assert response.status_code == 400
    assert "Unsupported provider" in response.json()["detail"]

def test_token_exchange():
    """Test token exchange endpoint."""
    response = client.post(f"{settings.API_PREFIX}/auth/token", json={"code": "test_code"})
    
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()
    assert response.json()["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_protected_endpoint():
    """Test a protected endpoint using the auth middleware."""
    # Create a mock user
    mock_user = User(id="test-user", email="test@example.com")
    
    # Override the dependency
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Test endpoint with auth
    task_id = "task_test-user_123"
    response = client.get(
        f"{settings.API_PREFIX}/tasks/{task_id}",
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    assert "task_id" in response.json()
    assert response.json()["task_id"] == task_id
    assert response.json()["status"] == "in_progress"
    assert response.json()["result"] is None

def test_unauthorized_access():
    """Test accessing protected endpoint without auth token."""
    # Make sure no dependency overrides are active
    app.dependency_overrides.clear()
    
    task_id = "task_123"
    response = client.get(f"{settings.API_PREFIX}/tasks/{task_id}")
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"] 