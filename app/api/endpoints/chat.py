from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import uuid

from app.agent.router import get_router_agent
from app.core.logging import logger
from app.api.deps import get_current_user, User

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

@router.post("/", response_model=ChatResponse)
async def process_chat(request: ChatRequest, user: User = Depends(get_current_user)):
    """
    Process a chat message using the AI agent.
    """
    try:
        logger.info(f"Processing chat message from user {user.id}: {request.message[:50]}...")
        
        # Get the router agent
        router = get_router_agent()
        
        # Generate a conversation ID if not provided
        conversation_id = request.conversation_id or f"chat_{user.id}_{uuid.uuid4()}"
        
        # Process the message using the appropriate agent
        result = await router.route_and_execute({
            "input": request.message,
            "conversation_id": conversation_id,
            "user_id": user.id
        })
        
        # Extract the response
        response = result.get("output", "I'm sorry, I couldn't process your request.")
        
        return {
            "response": response,
            "conversation_id": conversation_id
        }
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")
