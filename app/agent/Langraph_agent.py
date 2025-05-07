"""
LangGraph agent for more complex, multi-step workflows.
"""
from typing import Dict, List, Optional, TypedDict, Annotated, Sequence, Any
import operator
from functools import partial

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolNode

from app.services.llm import get_llm
from app.tools.notion_tools import get_notion_page_tool, create_notion_page_tool
from app.tools.github_tools import (
    get_github_issue_tool,
    get_github_pr_tool,
    get_github_file_tool,
    create_github_issue_tool,
)
from app.tools.slack_tools import get_slack_messages_tool, send_slack_message_tool
from app.tools.rag_tools import semantic_search_tool
from app.core.logging import logger


# Define state
class AgentState(TypedDict):
    """State for the agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tools_to_use: Optional[List[str]]
    tool_results: Optional[Dict[str, Any]]
    next_steps: Optional[List[str]]
    final_response: Optional[str]


# System prompt
SYSTEM_PROMPT = """You are an AI Task Copilot, a helpful assistant that can manage tasks across Notion, GitHub, and Slack.
You have access to the following capabilities:

1. Notion:
   - Read pages
   - Create new pages
   
2. GitHub:
   - Read issues and pull requests
   - Create new issues
   - Read file contents
   
3. Slack:
   - Read channel messages
   - Send messages
   
4. Semantic Search:
   - Search across all data sources using natural language

When helping users, follow these guidelines:
1. Use semantic search to find relevant context before taking actions
2. Break down complex tasks into smaller steps
3. Provide clear explanations of what you're doing
4. Handle errors gracefully and suggest alternatives
5. Ask for clarification when needed

Remember to maintain context across the conversation and refer back to previous interactions when relevant.
"""


def create_tools():
    """Create the list of available tools."""
    return [
        semantic_search_tool,
        get_notion_page_tool,
        create_notion_page_tool,
        get_github_issue_tool,
        get_github_pr_tool,
        get_github_file_tool,
        create_github_issue_tool,
        get_slack_messages_tool,
        send_slack_message_tool,
    ]


def create_agent():
    """Create the LangGraph agent."""
    try:
        logger.info("Creating LangGraph agent")
        
        # Get LLM
        llm = get_llm()
        
        # Create tools
        tools = create_tools()
        
        # Convert to OpenAI functions
        functions = [convert_to_openai_function(t) for t in tools]
        
        # Create tool mapping
        tool_map = {tool.name: tool for tool in tools}
        
        # Define the initial prompt for the graph
        assessment_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create assessment chain
        assessment_chain = assessment_prompt | llm.bind_functions(functions=functions)
        
        # Tool execution node
        tools_node = ToolNode(tools=tool_map)
        
        # Create nodes for the workflow
        def should_continue(state: AgentState) -> str:
            """Determine if the agent should continue or finish."""
            last_message = state["messages"][-1]
            
            # Check if this is a function call
            if hasattr(last_message, "function_call") and last_message.function_call:
                return "continue"
            
            # Otherwise, we're done
            return "end"
            
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", assessment_chain)
        workflow.add_node("tools", tools_node)
        
        # Add edges
        workflow.add_edge("agent", should_continue)
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        
        # Set the entry point
        workflow.set_entry_point("agent")
        
        # Compile the workflow
        app = workflow.compile(checkpointer=MemorySaver())
        
        return app
        
    except Exception as e:
        logger.error(f"Failed to create LangGraph agent: {e}", exc_info=True)
        raise


# Create executor function for the LangGraph agent
async def execute_graph_agent(input_text: str, chat_history: List[BaseMessage] = None) -> str:
    """
    Execute the LangGraph agent with the given input.
    
    Args:
        input_text: User input text
        chat_history: Optional chat history
        
    Returns:
        Agent response
    """
    try:
        # Create the agent
        agent = create_agent()
        
        # Initialize messages
        messages = []
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
            
        # Add current input
        messages.append(HumanMessage(content=input_text))
        
        # Execute the agent
        result = await agent.ainvoke({"messages": messages})
        
        # Get the final messages
        final_messages = result["messages"]
        
        # Return the last AI message
        for message in reversed(final_messages):
            if not isinstance(message, HumanMessage):
                return message.content
                
        return "Unable to get a response from the agent."
        
    except Exception as e:
        logger.error(f"Error executing graph agent: {e}", exc_info=True)
        return f"Error: {str(e)}"
