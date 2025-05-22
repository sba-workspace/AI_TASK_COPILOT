import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time
from jose import jwt

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
            url="https://ytopycjpelmtqrdbsxcr.supabase.co/auth/v1/authorize?redirect_to=/docs/oauth2-redirect&provider=github"
        )
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.get(f"{settings.API_PREFIX}/auth/oauth/login/github")
        
        assert response.status_code == 200
        assert "url" in response.json()
        assert "supabase" in response.json()["url"]
        assert "github" in response.json()["url"]
        assert "docs/oauth2-redirect" in response.json()["url"]

def test_oauth_login_invalid_provider():
    """Test OAuth login with invalid provider."""
    response = client.get(f"{settings.API_PREFIX}/auth/oauth/login/invalid")
    
    assert response.status_code == 400
    assert "Unsupported provider" in response.json()["detail"]
    assert "Supported providers are" in response.json()["detail"]

def test_token_exchange():
    """Test token exchange endpoint."""
    response = client.post(f"{settings.API_PREFIX}/auth/token", json={"code": "test_code"})
    
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()
    assert response.json()["token_type"] == "bearer"
    assert "refresh_token" in response.json()
    assert "expires_in" in response.json()

def test_login_success():
    """Test successful login with email and password."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_session.access_token = "mock_access_token"
        mock_session.refresh_token = "mock_refresh_token"
        mock_session.expires_in = 3600
        
        mock_client.auth.sign_in_with_password.return_value = MagicMock(
            session=mock_session
        )
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/login", 
            json={"email": "user@example.com", "password": "password123"}
        )
        
        assert response.status_code == 200
        assert response.json()["access_token"] == "mock_access_token"
        assert response.json()["token_type"] == "bearer"
        assert response.json()["refresh_token"] == "mock_refresh_token"
        assert response.json()["expires_in"] == 3600

def test_login_invalid_credentials():
    """Test login with invalid credentials."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client to raise an exception
        mock_client = MagicMock()
        mock_client.auth.sign_in_with_password.side_effect = Exception("Invalid credentials")
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/login", 
            json={"email": "user@example.com", "password": "wrong_password"}
        )
        
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]

def test_signup_success():
    """Test successful signup with email and password."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_session.access_token = "mock_access_token"
        mock_session.refresh_token = "mock_refresh_token"
        mock_session.expires_in = 3600
        
        mock_user = MagicMock()
        mock_user.id = "new-user-id"
        
        mock_client.auth.sign_up.return_value = MagicMock(
            user=mock_user,
            session=mock_session
        )
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/signup", 
            json={
                "email": "newuser@example.com", 
                "password": "password123",
                "name": "New User"
            }
        )
        
        assert response.status_code == 200
        assert response.json()["access_token"] == "mock_access_token"
        assert response.json()["token_type"] == "bearer"
        assert response.json()["refresh_token"] == "mock_refresh_token"
        assert response.json()["expires_in"] == 3600

def test_signup_email_confirmation():
    """Test signup with email confirmation required."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "new-user-id"
        
        # No session returned (email confirmation required)
        mock_client.auth.sign_up.return_value = MagicMock(
            user=mock_user,
            session=None
        )
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/signup", 
            json={
                "email": "newuser@example.com", 
                "password": "password123"
            }
        )
        
        assert response.status_code == 200
        assert response.json()["access_token"] == "verification_needed"
        assert response.json()["refresh_token"] is None
        assert response.json()["expires_in"] is None

def test_signup_existing_user():
    """Test signup with an email that already exists."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_client.auth.sign_up.side_effect = Exception("User already registered")
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/signup", 
            json={
                "email": "existing@example.com", 
                "password": "password123"
            }
        )
        
        assert response.status_code == 409
        assert "User already exists" in response.json()["detail"]

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

@pytest.mark.asyncio
async def test_forbidden_access():
    """Test accessing a resource belonging to a different user."""
    # Create a mock user with ID "different-user"
    mock_user = User(id="different-user", email="different@example.com")
    
    # Override the dependency
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Try to access a resource that belongs to "test-user"
    task_id = "task_test-user_123"
    response = client.get(
        f"{settings.API_PREFIX}/tasks/{task_id}",
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 403
    assert "permission" in response.json()["detail"]

def test_missing_code_parameter():
    """Test token exchange with missing required code parameter."""
    response = client.post(f"{settings.API_PREFIX}/auth/token", json={})
    
    assert response.status_code == 422  # Unprocessable Entity
    assert "code" in response.json()["detail"][0]["loc"]

def test_oauth_login_missing_provider():
    """Test OAuth login with missing provider parameter."""
    response = client.post(f"{settings.API_PREFIX}/auth/oauth", json={})
    
    assert response.status_code == 422  # Unprocessable Entity
    assert "provider" in response.json()["detail"][0]["loc"]

@pytest.mark.asyncio
async def test_expired_token():
    """Test accessing a protected endpoint with an expired token."""
    # Create an expired JWT
    with patch("app.api.deps.get_supabase_client") as mock_get_client:
        # Create a mock exception for expired token
        mock_client = MagicMock()
        mock_client.auth.get_user.side_effect = Exception("Token expired")
        mock_get_client.return_value = mock_client
        
        # Create a mock JWT decode function that raises a JWTError
        with patch("app.api.deps.jwt.decode") as mock_jwt_decode:
            mock_jwt_decode.side_effect = jwt.JWTError("Token expired")
            
            # Test endpoint with expired token
            response = client.get(
                f"{settings.API_PREFIX}/tasks/task_123",
                headers={"Authorization": "Bearer expired_token"}
            )
            
            assert response.status_code == 401
            assert "credentials" in response.json()["detail"]

@pytest.mark.asyncio
async def test_token_with_invalid_signature():
    """Test accessing a protected endpoint with a token having invalid signature."""
    with patch("app.api.deps.get_supabase_client") as mock_get_client:
        # Create a mock exception for invalid signature
        mock_client = MagicMock()
        mock_client.auth.get_user.side_effect = Exception("Invalid signature")
        mock_get_client.return_value = mock_client
        
        # Create a mock JWT decode function that raises a JWTError
        with patch("app.api.deps.jwt.decode") as mock_jwt_decode:
            mock_jwt_decode.side_effect = jwt.JWTError("Signature verification failed")
            
            # Test endpoint with invalid token
            response = client.get(
                f"{settings.API_PREFIX}/tasks/task_123",
                headers={"Authorization": "Bearer invalid_signature_token"}
            )
            
            assert response.status_code == 401
            assert "credentials" in response.json()["detail"]

@pytest.mark.asyncio
async def test_supabase_fallback_to_jwt():
    """Test fallback to manual JWT verification when Supabase client fails."""
    with patch("app.api.deps.get_supabase_client") as mock_get_client:
        # Make Supabase client fail
        mock_client = MagicMock()
        mock_client.auth.get_user.side_effect = Exception("Supabase unavailable")
        mock_get_client.return_value = mock_client
        
        # But make JWT verification succeed
        with patch("app.api.deps.jwt.decode") as mock_jwt_decode:
            mock_jwt_decode.return_value = {
                "sub": "jwt-user-id", 
                "email": "jwt@example.com",
                "exp": int(time.time()) + 3600  # Valid for 1 hour
            }
            
            # Override the dependency to use the real function
            app.dependency_overrides.clear()
            
            # Test endpoint with valid JWT but Supabase down
            response = client.get(
                f"{settings.API_PREFIX}/tasks/task_jwt-user-id_123",
                headers={"Authorization": "Bearer valid_jwt_token"}
            )
            
            assert response.status_code == 200
            assert response.json()["task_id"] == "task_jwt-user-id_123"

@pytest.mark.asyncio
async def test_api_prefix_correctness():
    """Test that the API_PREFIX setting is correctly applied to all routes."""
    # Try with a wrong prefix
    wrong_prefix = "/wrong/prefix"
    response = client.post(f"{wrong_prefix}/auth/oauth", json={"provider": "github"})
    
    # Should return 404 Not Found
    assert response.status_code == 404
    
    # Now try with the correct prefix
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.auth.sign_in_with_oauth.return_value = MagicMock(
            url="https://supabase-url/auth/v1/authorize?provider=github"
        )
        mock_get_client.return_value = mock_client

        response = client.post(f"{settings.API_PREFIX}/auth/oauth", json={"provider": "github"})
        
        # Should succeed
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_rate_limit_simulation():
    """Test simulation of rate-limiting behavior on repeated login attempts."""
    # This test simulates how rate limiting would be tested
    # In a real implementation, you would have actual rate limiting middleware
    
    # Create a counter to simulate rate limit
    request_count = 0
    
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        mock_client = MagicMock()
        
        # Simulate rate limiting after 5 requests
        def side_effect(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            if request_count > 5:
                raise Exception("Too many requests")
            return MagicMock(url="https://supabase-url/auth/v1/authorize?provider=github")
        
        mock_client.auth.sign_in_with_oauth.side_effect = side_effect
        mock_get_client.return_value = mock_client
        
        # First 5 requests should succeed
        for _ in range(5):
            response = client.post(f"{settings.API_PREFIX}/auth/oauth", json={"provider": "github"})
            assert response.status_code == 200
        
        # 6th request should fail
        response = client.post(f"{settings.API_PREFIX}/auth/oauth", json={"provider": "github"})
        assert response.status_code == 500
        assert "service unavailable" in response.json()["detail"]

def test_pydantic_model_validation():
    """Test validation of Pydantic models for task results."""
    # Create a mock user
    mock_user = User(id="test-user", email="test@example.com")
    
    # Override the dependency
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Send invalid data to the run-task endpoint
    response = client.post(
        f"{settings.API_PREFIX}/tasks/run-task",
        json={"invalid_field": "This should fail validation"}
    )
    
    assert response.status_code == 422  # Unprocessable Entity
    assert "description" in response.json()["detail"][0]["loc"]

def test_refresh_token_success():
    """Test successful refresh token endpoint."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_session.access_token = "new_access_token"
        mock_session.refresh_token = "new_refresh_token"
        mock_session.expires_in = 3600
        
        mock_client.auth.refresh_session.return_value = MagicMock(
            session=mock_session
        )
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/refresh", 
            json={"refresh_token": "old_refresh_token"}
        )
        
        assert response.status_code == 200
        assert response.json()["access_token"] == "new_access_token"
        assert response.json()["refresh_token"] == "new_refresh_token"
        assert response.json()["expires_in"] == 3600

def test_refresh_token_invalid():
    """Test refresh token endpoint with invalid refresh token."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client to raise an exception
        mock_client = MagicMock()
        mock_client.auth.refresh_session.side_effect = Exception("Invalid refresh token")
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/refresh", 
            json={"refresh_token": "invalid_refresh_token"}
        )
        
        assert response.status_code == 401
        assert "Invalid or expired refresh token" in response.json()["detail"]

def test_logout():
    """Test logout endpoint."""
    # Create a mock user
    mock_user = User(id="test-user", email="test@example.com")
    
    # Override the dependency
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/logout",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["detail"]
        mock_client.auth.sign_out.assert_called_once()

def test_logout_error_handling():
    """Test logout endpoint with Supabase error."""
    # Create a mock user
    mock_user = User(id="test-user", email="test@example.com")
    
    # Override the dependency
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client to raise an exception
        mock_client = MagicMock()
        mock_client.auth.sign_out.side_effect = Exception("Supabase error")
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/logout",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should still return success
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["detail"]

def test_reset_password():
    """Test reset password endpoint."""
    with patch("app.db.supabase.get_supabase_client") as mock_get_client:
        # Setup mock Supabase client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Test the endpoint
        response = client.post(
            f"{settings.API_PREFIX}/auth/reset-password",
            json={"email": "user@example.com"}
        )
        
        assert response.status_code == 200
        # Should not confirm if email exists or not
        assert "If the email exists" in response.json()["detail"]
        mock_client.auth.reset_password_email.assert_called_once_with("user@example.com") 