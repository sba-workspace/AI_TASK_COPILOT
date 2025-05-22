"""
LangChain agent for AI Task Copilot.
"""
from functools import lru_cache
from typing import List, Dict, Any, Callable
import re
import asyncio
import inspect

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain.tools import BaseTool

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


# System prompt for the agent
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

# Human message template
HUMAN_TEMPLATE = """User Request: {input}

Please help me with this task. You can use your tools to search for information and take actions across Notion, GitHub, and Slack."""


class SimpleAgent:
    """A simple agent that processes input and returns a response."""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            semantic_search_tool,  # Always try semantic search first
            get_notion_page_tool,
            create_notion_page_tool,
            get_github_issue_tool,
            get_github_pr_tool,
            get_github_file_tool,
            create_github_issue_tool,
            get_slack_messages_tool,
            send_slack_message_tool,
        ]
        
    def extract_github_info(self, text: str) -> tuple[str, int]:
        """Extract repository name and issue number from text."""
        # Extract repository name
        repo_match = re.search(r'github\.com/([^/\s]+/[^/\s]+)', text)
        if not repo_match:
            raise ValueError("Could not find GitHub repository URL")
        repo_name = repo_match.group(1)
        
        # Extract issue number
        issue_match = re.search(r'issue\s*#?(\d+)', text.lower())
        if not issue_match:
            raise ValueError("Could not find issue number")
        issue_number = int(issue_match.group(1))
        
        return repo_name, issue_number
        
    async def async_invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input and return a response asynchronously."""
        try:
            # Extract repository and issue number from the input
            input_text = input_dict["input"]
            
            # Check if this is a GitHub issue request
            if "github.com" in input_text and "issue" in input_text.lower():
                try:
                    # Extract repository name and issue number
                    repo_name, issue_number = self.extract_github_info(input_text)
                    
                    # Use the GitHub issue tool with a dictionary of arguments
                    # issue_info = await get_github_issue_tool.ainvoke({
                    # "repo_name": repo_name,
                    # "issue_number": issue_number
                    # })
                    
                    # Defensive awaiting:
                    # Call ainvoke and await its result
                    invoked_result = await get_github_issue_tool.ainvoke({
                        "repo_name": repo_name,
                        "issue_number": issue_number
                    })
                    
                    # Check if the result itself is a coroutine
                    if inspect.iscoroutine(invoked_result):
                        logger.info("The result of ainvoke was a coroutine, awaiting it.")
                        issue_info = await invoked_result
                    else:
                        issue_info = invoked_result
                    
                    # Format the response
                    response = f"I've retrieved the GitHub issue information:\\n\\n{issue_info}"
                except ValueError as e:
                    response = f"I couldn't process your request: {str(e)}. Please provide a valid GitHub repository URL and issue number."
                except Exception as e:
                    logger.error(f"Error in GitHub tool: {e}", exc_info=True)
                    response = f"Error accessing GitHub: {str(e)}. Please check if the repository and issue exist and are accessible."
            else:
                # Format the input for general queries
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=HUMAN_TEMPLATE.format(input=input_text))
                ]
                
                # Call the LLM
                response = self.llm.invoke(messages).content
            
            return {"output": response}
            
        except Exception as e:
            logger.error(f"Error in agent invoke: {e}", exc_info=True)
            return {"output": f"I encountered an error while processing your request: {str(e)}"}
            
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input and return a response.
        This is a sync interface that just invokes the LLM directly for non-async contexts.
        For async contexts, use async_invoke.
        """
        try:
            # Extract input text
            input_text = input_dict["input"]
            
            # For synchronous contexts, just use the LLM directly
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=HUMAN_TEMPLATE.format(input=input_text))
            ]
            
            # Call the LLM
            response = self.llm.invoke(messages).content
            
            return {"output": response}
        except Exception as e:
            logger.error(f"Error in agent invoke: {e}", exc_info=True)
            return {"output": f"I encountered an error while processing your request: {str(e)}"}


@lru_cache()
def get_agent():
    """
    Create and return a simple agent that processes input and returns a response.
    Uses lru_cache to ensure only one instance is created.
    """
    try:
        logger.info("Creating LangChain agent")
        
        # Get LLM
        llm = get_llm()
        
        # Create and return the agent
        return SimpleAgent(llm)
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise 