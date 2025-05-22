"""
Router agent that decides which agent to use for a given query.
"""
from typing import Dict, Any, Optional
import re

from app.services.llm import get_llm
from app.agent.agent import get_agent as get_simple_agent
from app.agent.Langraph_agent import execute_graph_agent
from app.core.logging import logger

# Simple patterns to help identify complex queries
COMPLEX_PATTERNS = [
    r"multiple\s+step", r"multi\s*step", r"sequence", r"workflow",
    r"(first|then|after|next|finally)",
    r"(\d+)[\.\)]", # Numbered lists
    r"create.*then.*update",
    r"search.*create",
    r"compare",
    r"analyze.*then",
]

class AgentRouter:
    """
    Router agent that decides which agent to use for a given query.
    """
    
    def __init__(self):
        """Initialize the router agent."""
        self.simple_agent = get_simple_agent()
    
    async def analyze_query(self, query: str) -> str:
        """
        Analyze the query to determine which agent to use.
        
        Args:
            query: The user query
            
        Returns:
            str: Either "simple" or "langraph" indicating which agent to use
        """
        # Simple heuristic approach based on patterns
        for pattern in COMPLEX_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return "langraph"
        
        # Default to the simple agent for most queries
        return "simple"
    
    async def route_and_execute(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route the query to the appropriate agent and execute it.
        
        Args:
            input_dict: Dictionary containing the input query
            
        Returns:
            Dict[str, Any]: The agent's response
        """
        try:
            input_text = input_dict["input"]
            
            # Determine which agent to use
            agent_type = await self.analyze_query(input_text)
            logger.info(f"Router selected agent type: {agent_type}")
            
            if agent_type == "langraph":
                # Use the LangGraph agent
                result = await execute_graph_agent(input_text)
                return {"output": result}
            else:
                # Use the simple agent
                return await self.simple_agent.async_invoke(input_dict)
                
        except Exception as e:
            logger.error(f"Error in agent router: {e}", exc_info=True)
            
            # Fallback to the simple agent if routing fails
            try:
                return await self.simple_agent.async_invoke(input_dict)
            except Exception as fallback_error:
                logger.error(f"Fallback to simple agent also failed: {fallback_error}", exc_info=True)
                return {"output": f"I encountered an error while processing your request: {str(e)}"}


# Singleton instance
_router_instance = None

def get_router_agent():
    """
    Get or create the router agent.
    
    Returns:
        AgentRouter: The router agent instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = AgentRouter()
    return _router_instance 