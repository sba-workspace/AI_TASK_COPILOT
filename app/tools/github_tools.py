"""
GitHub tools for the LangChain agent.
"""
from typing import Optional
from pydantic import BaseModel, Field

from langchain.tools import StructuredTool

from app.services.github import get_issue, get_pull_request, get_file, create_issue
from app.core.logging import logger


class GetGitHubIssueInput(BaseModel):
    """Input for getting a GitHub issue."""
    repo_name: str = Field(..., description="Full repository name (e.g., 'owner/repo')")
    issue_number: int = Field(..., description="Issue number")


class GetGitHubPRInput(BaseModel):
    """Input for getting a GitHub pull request."""
    repo_name: str = Field(..., description="Full repository name (e.g., 'owner/repo')")
    pr_number: int = Field(..., description="Pull request number")


class GetGitHubFileInput(BaseModel):
    """Input for getting a GitHub file."""
    repo_name: str = Field(..., description="Full repository name (e.g., 'owner/repo')")
    path: str = Field(..., description="Path to the file in the repository")
    ref: Optional[str] = Field(None, description="Optional git reference (branch, tag, commit)")


class CreateGitHubIssueInput(BaseModel):
    """Input for creating a GitHub issue."""
    repo_name: str = Field(..., description="Full repository name (e.g., 'owner/repo')")
    title: str = Field(..., description="Issue title")
    body: str = Field(..., description="Issue body/content")


async def get_github_issue(repo_name: str, issue_number: int) -> str:
    """
    Get a GitHub issue by repository and number.
    
    Args:
        repo_name: Full repository name (e.g., 'owner/repo')
        issue_number: Issue number
        
    Returns:
        The issue content as text
    """
    try:
        issue = await get_issue(repo_name, issue_number)
        return f"Title: {issue.title}\n\n{issue.content}\n\nURL: {issue.url}"
    except Exception as e:
        logger.error(f"Error in get_github_issue tool: {e}", exc_info=True)
        return f"Error getting GitHub issue: {str(e)}"


async def get_github_pr(repo_name: str, pr_number: int) -> str:
    """
    Get a GitHub pull request by repository and number.
    
    Args:
        repo_name: Full repository name (e.g., 'owner/repo')
        pr_number: Pull request number
        
    Returns:
        The PR content as text
    """
    try:
        pr = await get_pull_request(repo_name, pr_number)
        return f"Title: {pr.title}\n\n{pr.content}\n\nURL: {pr.url}"
    except Exception as e:
        logger.error(f"Error in get_github_pr tool: {e}", exc_info=True)
        return f"Error getting GitHub PR: {str(e)}"


async def get_github_file(repo_name: str, path: str, ref: Optional[str] = None) -> str:
    """
    Get a GitHub file's content by repository and path.
    
    Args:
        repo_name: Full repository name (e.g., 'owner/repo')
        path: Path to the file in the repository
        ref: Optional git reference (branch, tag, commit)
        
    Returns:
        The file content as text
    """
    try:
        file = await get_file(repo_name, path, ref)
        return f"File: {file.name}\n\n{file.content}\n\nURL: {file.html_url}"
    except Exception as e:
        logger.error(f"Error in get_github_file tool: {e}", exc_info=True)
        return f"Error getting GitHub file: {str(e)}"


async def create_github_issue(repo_name: str, title: str, body: str) -> str:
    """
    Create a new GitHub issue.
    
    Args:
        repo_name: Full repository name (e.g., 'owner/repo')
        title: Issue title
        body: Issue body/content
        
    Returns:
        The URL of the created issue
    """
    try:
        url = await create_issue(repo_name, title, body)
        return f"Created GitHub issue: {url}"
    except Exception as e:
        logger.error(f"Error in create_github_issue tool: {e}", exc_info=True)
        return f"Error creating GitHub issue: {str(e)}"


# Create LangChain tools
get_github_issue_tool = StructuredTool.from_function(
    func=get_github_issue,
    name="get_github_issue",
    description="Get a GitHub issue by repository name and issue number",
    args_schema=GetGitHubIssueInput,
)

get_github_pr_tool = StructuredTool.from_function(
    func=get_github_pr,
    name="get_github_pr",
    description="Get a GitHub pull request by repository name and PR number",
    args_schema=GetGitHubPRInput,
)

get_github_file_tool = StructuredTool.from_function(
    func=get_github_file,
    name="get_github_file",
    description="Get a GitHub file's content by repository name and file path",
    args_schema=GetGitHubFileInput,
)

create_github_issue_tool = StructuredTool.from_function(
    func=create_github_issue,
    name="create_github_issue",
    description="Create a new GitHub issue with the given title and body",
    args_schema=CreateGitHubIssueInput,
) 