"""
LangChain agent for AI Task Copilot.
"""
from functools import lru_cache
from typing import List

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
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


@lru_cache()
def get_agent() -> AgentExecutor:
    """
    Create and return a LangChain agent instance.
    Uses lru_cache to ensure only one agent is created.
    """
    try:
        logger.info("Creating LangChain agent")
        
        # Get LLM
        llm = get_llm()
        
        # Create tools list
        tools: List[BaseTool] = [
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
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", HUMAN_TEMPLATE),
        ])
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        
        # Create agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent_executor
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise 