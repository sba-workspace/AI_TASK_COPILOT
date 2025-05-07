"""
Slack tools for the LangChain agent.
"""
from typing import Optional
from pydantic import BaseModel, Field

from langchain.tools import StructuredTool

from app.services.slack import get_channel_messages, send_message
from app.core.logging import logger


class GetSlackMessagesInput(BaseModel):
    """Input for getting Slack messages."""
    channel_id: str = Field(..., description="The ID of the Slack channel")
    limit: Optional[int] = Field(100, description="Maximum number of messages to retrieve")


class SendSlackMessageInput(BaseModel):
    """Input for sending a Slack message."""
    channel_id: str = Field(..., description="The ID of the Slack channel")
    text: str = Field(..., description="The message text to send")


async def get_slack_messages(channel_id: str, limit: int = 100) -> str:
    """
    Get messages from a Slack channel.
    
    Args:
        channel_id: The ID of the Slack channel
        limit: Maximum number of messages to retrieve
        
    Returns:
        The messages as formatted text
    """
    try:
        messages = await get_channel_messages(channel_id, limit)
        
        if not messages:
            return "No messages found in the channel."
            
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(
                f"From: {msg.sender}\n"
                f"Time: {msg.timestamp}\n"
                f"Message: {msg.content}\n"
            )
            
        return "\n---\n".join(formatted_messages)
        
    except Exception as e:
        logger.error(f"Error in get_slack_messages tool: {e}", exc_info=True)
        return f"Error getting Slack messages: {str(e)}"


async def send_slack_message(channel_id: str, text: str) -> str:
    """
    Send a message to a Slack channel.
    
    Args:
        channel_id: The ID of the Slack channel
        text: The message text to send
        
    Returns:
        Confirmation message
    """
    try:
        timestamp = await send_message(channel_id, text)
        return f"Message sent successfully at {timestamp}"
    except Exception as e:
        logger.error(f"Error in send_slack_message tool: {e}", exc_info=True)
        return f"Error sending Slack message: {str(e)}"


# Create LangChain tools
get_slack_messages_tool = StructuredTool.from_function(
    func=get_slack_messages,
    name="get_slack_messages",
    description="Get messages from a Slack channel",
    args_schema=GetSlackMessagesInput,
)

send_slack_message_tool = StructuredTool.from_function(
    func=send_slack_message,
    name="send_slack_message",
    description="Send a message to a Slack channel",
    args_schema=SendSlackMessageInput,
) 