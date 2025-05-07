"""
Slack API service for AI Task Copilot.
"""
from functools import lru_cache
from typing import Dict, List, Optional
from datetime import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger
from app.db.weaviate import get_weaviate_client
from app.services.embeddings import embed_texts


class SlackMessage(BaseModel):
    """Slack message model."""
    id: str
    text: str
    user: str
    channel_id: str
    channel_name: str
    timestamp: str
    permalink: Optional[str] = None
    thread_ts: Optional[str] = None
    reactions: List[str] = []


class SlackChannel(BaseModel):
    """Slack channel model."""
    id: str
    name: str
    is_private: bool
    member_count: int
    topic: str
    purpose: str


@lru_cache()
def get_slack_client() -> WebClient:
    """
    Create and return a Slack Web API client instance.
    Uses lru_cache to ensure only one client is created.
    """
    try:
        logger.debug("Creating Slack client")
        client = WebClient(token=settings.SLACK_BOT_TOKEN)
        return client
    except Exception as e:
        logger.error(f"Failed to create Slack client: {e}", exc_info=True)
        raise


async def get_channel_messages(
    channel_id: str, 
    limit: int = 10, 
    oldest: Optional[str] = None,
    latest: Optional[str] = None
) -> List[SlackMessage]:
    """
    Get messages from a Slack channel.
    
    Args:
        channel_id: The ID of the channel
        limit: Maximum number of messages to return
        oldest: Timestamp of the oldest message to include
        latest: Timestamp of the latest message to include
        
    Returns:
        List of SlackMessage objects
    """
    try:
        client = get_slack_client()
        
        # Get channel info for the name
        channel_info = client.conversations_info(channel=channel_id)
        channel_name = channel_info["channel"]["name"]
        
        # Get messages
        response = client.conversations_history(
            channel=channel_id,
            limit=limit,
            oldest=oldest,
            latest=latest
        )
        
        messages = []
        for msg in response["messages"]:
            # Skip bot messages if they don't have text
            if "subtype" in msg and msg["subtype"] == "bot_message" and not msg.get("text"):
                continue
                
            # Get user display name when possible
            user_display = msg.get("user", "unknown")
            if "user" in msg and msg["user"]:
                try:
                    user_info = client.users_info(user=msg["user"])
                    user_display = user_info["user"]["real_name"] or user_info["user"]["name"]
                except:
                    # Fall back to user ID if we can't get the name
                    pass
            
            # Get message permalink if possible
            permalink = None
            try:
                if "ts" in msg:
                    permalink_resp = client.chat_getPermalink(
                        channel=channel_id,
                        message_ts=msg["ts"]
                    )
                    permalink = permalink_resp.get("permalink")
            except:
                # Permalinks are not essential, so continue if we can't get one
                pass
                
            # Get reactions
            reactions = []
            if "reactions" in msg:
                for reaction in msg["reactions"]:
                    reactions.append(f":{reaction['name']}:")
            
            messages.append(SlackMessage(
                id=msg["ts"],
                text=msg.get("text", ""),
                user=user_display,
                channel_id=channel_id,
                channel_name=channel_name,
                timestamp=msg["ts"],
                permalink=permalink,
                thread_ts=msg.get("thread_ts"),
                reactions=reactions
            ))
        
        return messages
        
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error getting Slack messages: {e}", exc_info=True)
        raise


async def get_thread_messages(
    channel_id: str, 
    thread_ts: str, 
    limit: int = 20
) -> List[SlackMessage]:
    """
    Get thread messages from a Slack conversation.
    
    Args:
        channel_id: The ID of the channel
        thread_ts: Timestamp of the parent message
        limit: Maximum number of messages to return
        
    Returns:
        List of SlackMessage objects
    """
    try:
        client = get_slack_client()
        
        # Get channel info for the name
        channel_info = client.conversations_info(channel=channel_id)
        channel_name = channel_info["channel"]["name"]
        
        # Get thread replies
        response = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=limit
        )
        
        messages = []
        for msg in response["messages"]:
            # Skip the parent message (it will be the first one)
            if msg["ts"] == thread_ts and len(messages) > 0:
                continue
                
            # Get user display name when possible
            user_display = msg.get("user", "unknown")
            if "user" in msg and msg["user"]:
                try:
                    user_info = client.users_info(user=msg["user"])
                    user_display = user_info["user"]["real_name"] or user_info["user"]["name"]
                except:
                    # Fall back to user ID if we can't get the name
                    pass
            
            # Get message permalink if possible
            permalink = None
            try:
                permalink_resp = client.chat_getPermalink(
                    channel=channel_id,
                    message_ts=msg["ts"]
                )
                permalink = permalink_resp.get("permalink")
            except:
                # Permalinks are not essential, so continue if we can't get one
                pass
                
            # Get reactions
            reactions = []
            if "reactions" in msg:
                for reaction in msg["reactions"]:
                    reactions.append(f":{reaction['name']}:")
            
            messages.append(SlackMessage(
                id=msg["ts"],
                text=msg.get("text", ""),
                user=user_display,
                channel_id=channel_id,
                channel_name=channel_name,
                timestamp=msg["ts"],
                permalink=permalink,
                thread_ts=thread_ts,
                reactions=reactions
            ))
        
        return messages
        
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error getting Slack thread messages: {e}", exc_info=True)
        raise


async def search_messages(query: str, limit: int = 5) -> List[SlackMessage]:
    """
    Search for Slack messages using the Slack search API.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of SlackMessage objects matching the query
    """
    try:
        client = get_slack_client()
        
        # Perform the search
        response = client.search_messages(
            query=query,
            count=limit,
            sort="timestamp",
            sort_dir="desc"
        )
        
        messages = []
        for match in response["messages"]["matches"]:
            channel_id = match["channel"]["id"]
            channel_name = match["channel"]["name"]
            
            # Get user display name when possible
            user_display = match.get("user", "unknown")
            if "user" in match and match["user"]:
                try:
                    user_info = client.users_info(user=match["user"])
                    user_display = user_info["user"]["real_name"] or user_info["user"]["name"]
                except:
                    # Fall back to user ID if we can't get the name
                    pass
            
            # Get permalink if available, otherwise construct from ts and channel
            permalink = match.get("permalink", None)
            
            messages.append(SlackMessage(
                id=match["ts"],
                text=match.get("text", ""),
                user=user_display,
                channel_id=channel_id,
                channel_name=channel_name,
                timestamp=match["ts"],
                permalink=permalink,
                thread_ts=match.get("thread_ts")
            ))
        
        return messages
        
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error searching Slack messages: {e}", exc_info=True)
        raise


async def send_message(
    channel_id: str, 
    text: str, 
    thread_ts: Optional[str] = None,
    blocks: Optional[List[Dict]] = None
) -> SlackMessage:
    """
    Send a message to a Slack channel or thread.
    
    Args:
        channel_id: The ID of the channel
        text: The message text
        thread_ts: Optional timestamp of the thread to reply to
        blocks: Optional blocks for rich formatting
        
    Returns:
        SlackMessage object for the sent message
    """
    try:
        client = get_slack_client()
        
        # Get channel info for the name
        try:
            channel_info = client.conversations_info(channel=channel_id)
            channel_name = channel_info["channel"]["name"]
        except:
            # If we can't get channel info, use the ID as the name
            channel_name = channel_id
        
        # Prepare message parameters
        msg_params = {
            "channel": channel_id,
            "text": text
        }
        
        # Add thread_ts if replying to a thread
        if thread_ts:
            msg_params["thread_ts"] = thread_ts
            
        # Add blocks if provided
        if blocks:
            msg_params["blocks"] = blocks
            
        # Send the message
        response = client.chat_postMessage(**msg_params)
        
        # Get bot user info
        bot_info = client.auth_test()
        bot_user_id = bot_info["user_id"]
        
        # Create and return message object
        return SlackMessage(
            id=response["ts"],
            text=text,
            user=bot_user_id,
            channel_id=channel_id,
            channel_name=channel_name,
            timestamp=response["ts"],
            thread_ts=thread_ts
        )
        
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error sending Slack message: {e}", exc_info=True)
        raise


async def list_channels(limit: int = 100, exclude_archived: bool = True) -> List[SlackChannel]:
    """
    List available Slack channels.
    
    Args:
        limit: Maximum number of channels to return
        exclude_archived: Whether to exclude archived channels
        
    Returns:
        List of SlackChannel objects
    """
    try:
        client = get_slack_client()
        
        # Get public channels
        response = client.conversations_list(
            limit=limit,
            exclude_archived=exclude_archived,
            types="public_channel,private_channel"
        )
        
        channels = []
        for channel in response["channels"]:
            channels.append(SlackChannel(
                id=channel["id"],
                name=channel["name"],
                is_private=channel["is_private"],
                member_count=channel["num_members"],
                topic=channel.get("topic", {}).get("value", ""),
                purpose=channel.get("purpose", {}).get("value", "")
            ))
        
        return channels
        
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error listing Slack channels: {e}", exc_info=True)
        raise


async def sync_slack_data(user_id: str, channel_ids: List[str] = None):
    """
    Sync Slack data to the vector store.
    
    Args:
        user_id: The ID of the user whose data to sync
        channel_ids: Optional list of channel IDs to sync
    """
    try:
        logger.info(f"Starting Slack sync for user {user_id}")
        
        client = get_slack_client()
        weaviate_client = get_weaviate_client()
        
        # If no channels specified, get all channels the bot has access to
        if not channel_ids:
            try:
                response = client.conversations_list(types="public_channel,private_channel")
                channel_ids = [c["id"] for c in response["channels"]]
            except SlackApiError as e:
                logger.error(f"Error listing channels: {e}", exc_info=True)
                raise
        
        for channel_id in channel_ids:
            try:
                # Get channel info
                channel_info = client.conversations_info(channel=channel_id)["channel"]
                channel_name = channel_info["name"]
                
                logger.info(f"Syncing channel: {channel_name}")
                
                # Get messages
                messages = await get_channel_messages(channel_id, limit=1000)  # Adjust limit as needed
                
                for msg in messages:
                    try:
                        # Embed the message content
                        embedding = (await embed_texts([msg.text]))[0]
                        
                        # Store in Weaviate
                        weaviate_client.data_object.create(
                            class_name="SlackMessage",
                            data_object={
                                "channel": msg.channel_id,
                                "content": msg.text,
                                "sender": msg.user,
                                "timestamp": msg.timestamp,
                                "userId": user_id,
                            },
                            vector=embedding
                        )
                        
                        logger.debug(f"Synced message {msg.id} from {channel_name}")
                        
                    except Exception as e:
                        logger.error(f"Error syncing message {msg.id}: {e}", exc_info=True)
                        continue
                
                logger.info(f"Completed sync for channel: {channel_name}")
                
            except Exception as e:
                logger.error(f"Error syncing channel {channel_id}: {e}", exc_info=True)
                continue
        
        logger.info(f"Completed Slack sync for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error in Slack sync for user {user_id}: {e}", exc_info=True)
        raise 