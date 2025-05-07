"""
RAG tools for semantic search using Weaviate.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain.tools import StructuredTool

from app.db.weaviate import get_weaviate_client
from app.services.embeddings import embed_query
from app.core.logging import logger


class SemanticSearchInput(BaseModel):
    """Input for semantic search."""
    query: str = Field(..., description="The search query")
    limit: Optional[int] = Field(5, description="Maximum number of results to return")
    source_type: Optional[str] = Field(None, description="Optional source type filter (notion, github, slack)")


async def semantic_search(query: str, limit: int = 5, source_type: Optional[str] = None) -> str:
    """
    Perform semantic search across all data sources.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        source_type: Optional source type filter (notion, github, slack)
        
    Returns:
        Search results as formatted text
    """
    try:
        # Get query embedding
        embedding = await embed_query(query)
        
        # Prepare search
        weaviate_client = get_weaviate_client()
        results = []
        
        # Define classes to search based on source type
        if source_type:
            if source_type.lower() == "notion":
                classes = ["NotionDocument"]
            elif source_type.lower() == "github":
                classes = ["GitHubItem"]
            elif source_type.lower() == "slack":
                classes = ["SlackMessage"]
            else:
                raise ValueError(f"Invalid source type: {source_type}")
        else:
            classes = ["NotionDocument", "GitHubItem", "SlackMessage"]
        
        # Search each class
        for class_name in classes:
            try:
                response = (
                    weaviate_client.query
                    .get(class_name, ["title", "content", "url", "type", "channel", "sender", "lastUpdated"])
                    .with_near_vector({
                        "vector": embedding
                    })
                    .with_limit(limit)
                    .do()
                )
                
                if "data" in response and "Get" in response["data"]:
                    class_results = response["data"]["Get"].get(class_name, [])
                    
                    for result in class_results:
                        # Format result based on class type
                        if class_name == "NotionDocument":
                            results.append(
                                f"[Notion] {result['title']}\n"
                                f"URL: {result['url']}\n"
                                f"Last Updated: {result['lastUpdated']}\n"
                                f"Content Preview: {result['content'][:200]}...\n"
                            )
                        elif class_name == "GitHubItem":
                            results.append(
                                f"[GitHub - {result.get('type', 'item')}] {result['title']}\n"
                                f"URL: {result['url']}\n"
                                f"Last Updated: {result['lastUpdated']}\n"
                                f"Content Preview: {result['content'][:200]}...\n"
                            )
                        elif class_name == "SlackMessage":
                            results.append(
                                f"[Slack - {result['channel']}] From: {result['sender']}\n"
                                f"Time: {result['lastUpdated']}\n"
                                f"Message: {result['content']}\n"
                            )
                
            except Exception as e:
                logger.error(f"Error searching {class_name}: {e}", exc_info=True)
                continue
        
        if not results:
            return "No results found."
            
        return "\n---\n".join(results)
        
    except Exception as e:
        logger.error(f"Error in semantic_search tool: {e}", exc_info=True)
        return f"Error performing semantic search: {str(e)}"


# Create LangChain tool
semantic_search_tool = StructuredTool.from_function(
    func=semantic_search,
    name="semantic_search",
    description="Search across Notion, GitHub, and Slack data using semantic similarity",
    args_schema=SemanticSearchInput,
) 