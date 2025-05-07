import pytest
from unittest.mock import patch, MagicMock

from app.services.llm import get_llm


@pytest.fixture
def mock_chat_google_generative_ai():
    """Mock for ChatGoogleGenerativeAI."""
    with patch("app.services.llm.ChatGoogleGenerativeAI") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


def test_get_llm(mock_chat_google_generative_ai):
    """Test getting LLM client."""
    # Call the function
    llm = get_llm()
    
    # Verify the ChatGoogleGenerativeAI was created with correct parameters
    mock_chat_google_generative_ai.assert_called_once()
    call_kwargs = mock_chat_google_generative_ai.call_args.kwargs
    
    assert call_kwargs["model"] == "gemini-1.5-flash"
    assert "google_api_key" in call_kwargs
    assert call_kwargs["temperature"] == 0.2
    
    # Verify we got back the mock instance
    assert llm == mock_chat_google_generative_ai.return_value
