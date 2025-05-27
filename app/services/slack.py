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
    """Retrieve and process messages from a Slack channel with robust error handling."""
    logger.debug(f"Starting message retrieval for channel: {channel_id}")
    client = get_slack_client()
    messages = []

    try:
        # Get channel metadata with enhanced validation
        channel_info = client.conversations_info(channel=channel_id).data
        if not channel_info.get('ok'):
            logger.error(f"Channel info request failed for {channel_id}: {channel_info.get('error')}")
            return []

        channel_data = channel_info.get('channel', {})
        if not isinstance(channel_data, dict):
            logger.error(f"Invalid channel data format for {channel_id}")
            return []

        channel_name = channel_data.get('name', 'unknown-channel')
        logger.debug(f"Processing channel: {channel_name}")

        # Retrieve message history with pagination support
        history_data = client.conversations_history(
            channel=channel_id,
            limit=limit,
            oldest=oldest,
            latest=latest
        ).data

        if not history_data.get('ok'):
            logger.error(f"History request failed for {channel_name}: {history_data.get('error')}")
            return []

        raw_messages = history_data.get('messages', [])
        logger.debug(f"Found {len(raw_messages)} raw messages in {channel_name}")

        # Process messages with comprehensive validation
        for idx, msg in enumerate(raw_messages):
            try:
                if not isinstance(msg, dict):
                    logger.warning(f"Skipping invalid message format at index {idx}")
                    continue

                # User resolution with fallback
                user_id = msg.get('user', 'unknown-user')
                user_display = await resolve_user_display(client, user_id)

                # Message metadata
                message_id = msg.get('ts', f"no-ts-{idx}")
                message_text = msg.get('text', '').strip()
                
                if not message_text:
                    logger.debug(f"Skipping empty message {message_id}")
                    continue

                # Permalink retrieval
                permalink = await get_message_permalink(client, channel_id, message_id)

                # Reaction processing
                reactions = process_reactions(msg.get('reactions', []))

                messages.append(SlackMessage(
                    id=message_id,
                    text=message_text,
                    user=user_display,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    timestamp=message_id,
                    permalink=permalink,
                    thread_ts=msg.get('thread_ts'),
                    reactions=reactions
                ))

            except Exception as msg_error:
                logger.warning(f"Error processing message {idx}: {str(msg_error)}")

        logger.info(f"Successfully processed {len(messages)}/{len(raw_messages)} messages from {channel_name}")
        return messages

    except SlackApiError as api_error:
        logger.error(f"Slack API failure: {api_error.response['error']}")
        raise
    except Exception as general_error:
        logger.error(f"Unexpected error: {str(general_error)}", exc_info=True)
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


async def sync_slack_data(user_id: str, channel_ids: Optional[List[str]] = None):
    """Orchestrate Slack data synchronization with fault-tolerant processing."""
    logger.info(f"Initiating Slack sync for user: {user_id}")
    client = get_slack_client()
    
    try:
        weaviate_client = get_weaviate_client()
    except Exception as weaviate_error:
        logger.warning("Weaviate unavailable: Vector storage disabled")
        weaviate_client = None

    try:
        channel_ids = channel_ids or await fetch_accessible_channels(client)
        logger.debug(f"Processing {len(channel_ids)} channels")

        success_count = 0
        for channel_id in channel_ids:
            try:
                channel_info = client.conversations_info(channel=channel_id).data
                if not channel_info.get('ok'):
                    logger.warning(f"Skipping channel {channel_id}: {channel_info.get('error')}")
                    continue

                channel_name = channel_info['channel'].get('name', 'unknown-channel')
                logger.info(f"Syncing channel: {channel_name}")

                messages = await get_channel_messages(channel_id, limit=1000)
                if not messages:
                    logger.debug(f"No messages found in {channel_name}")
                    continue

                if weaviate_client:
                    await store_messages_in_weaviate(weaviate_client, messages, user_id)
                
                success_count += 1
                logger.debug(f"Completed sync for {channel_name}")

            except Exception as channel_error:
                logger.warning(f"Channel {channel_id} sync failed: {str(channel_error)}")
                continue

        logger.info(f"Sync complete: {success_count}/{len(channel_ids)} channels processed")

    except SlackApiError as api_error:
        logger.error(f"Slack API failure: {api_error.response['error']}")
        raise
    except Exception as general_error:
        logger.error(f"Critical sync failure: {str(general_error)}", exc_info=True)
        raise


# Helper functions
async def resolve_user_display(client: WebClient, user_id: str) -> str:
    """Resolve user display name with caching and fallback."""
    try:
        user_info = client.users_info(user=user_id).data
        if user_info.get('ok'):
            user = user_info.get('user', {})
            return user.get('real_name') or user.get('name', user_id)
    except Exception as user_error:
        logger.debug(f"User resolution failed: {str(user_error)}")
    return user_id

async def get_message_permalink(client: WebClient, channel_id: str, ts: str) -> Optional[str]:
    """Retrieve message permalink with error suppression."""
    try:
        permalink_data = client.chat_getPermalink(channel=channel_id, message_ts=ts).data
        return permalink_data.get('permalink') if permalink_data.get('ok') else None
    except Exception:
        return None

def process_reactions(raw_reactions: List[dict]) -> List[str]:
    """Normalize reaction data."""
    return [
        f":{r['name']}:" 
        for r in raw_reactions 
        if isinstance(r, dict) and 'name' in r
    ]

async def fetch_accessible_channels(client: WebClient) -> List[str]:
    """Retrieve list of accessible channels with error handling."""
    try:
        response = client.conversations_list(types="public_channel,private_channel").data
        return [c['id'] for c in response.get('channels', [])] if response.get('ok') else []
    except Exception as channel_error:
        logger.error(f"Channel listing failed: {str(channel_error)}")
        return []

async def store_messages_in_weaviate(client, messages: List[SlackMessage], user_id: str):
    """Batch store messages in Weaviate with error tracking."""
    success = 0
    for msg in messages:
        try:
            embedding = (await embed_texts([msg.text]))[0]
            client.data_object.create(
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
            success += 1
        except Exception as store_error:
            logger.debug(f"Storage failed for {msg.id}: {str(store_error)}")
    logger.info(f"Stored {success}/{len(messages)} messages in Weaviate") 