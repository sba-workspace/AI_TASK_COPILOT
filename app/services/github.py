"""
GitHub API service for AI Task Copilot.
"""
from functools import lru_cache
from typing import Dict, List, Optional, Any

from github import Github, GithubException
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger


class GitHubIssue(BaseModel):
    """GitHub issue model."""
    id: int
    number: int
    title: str
    body: str
    state: str
    html_url: str
    created_at: str
    updated_at: str
    labels: List[str]
    assignees: List[str]
    repository: str


class GitHubPR(BaseModel):
    """GitHub pull request model."""
    id: int
    number: int
    title: str
    body: str
    state: str
    html_url: str
    created_at: str
    updated_at: str
    labels: List[str]
    assignees: List[str]
    repository: str
    branch: str
    merged: bool


class GitHubFile(BaseModel):
    """GitHub file model."""
    name: str
    path: str
    content: str
    html_url: str
    repository: str
    sha: str


@lru_cache()
def get_github_client() -> Github:
    """
    Create and return a GitHub client instance.
    Uses lru_cache to ensure only one client is created.
    """
    try:
        logger.debug("Creating GitHub client")
        client = Github(settings.GITHUB_API_TOKEN)
        return client
    except Exception as e:
        logger.error(f"Failed to create GitHub client: {e}", exc_info=True)
        raise


async def get_issue(repo_name: str, issue_number: int) -> GitHubIssue:
    """
    Get a GitHub issue by repository and issue number.
    
    Args:
        repo_name: The name of the repository (e.g., "username/repo")
        issue_number: The issue number
        
    Returns:
        GitHubIssue object with issue data
    """
    try:
        client = get_github_client()
        repo = client.get_repo(repo_name)
        issue = repo.get_issue(issue_number)
        
        return GitHubIssue(
            id=issue.id,
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            state=issue.state,
            html_url=issue.html_url,
            created_at=issue.created_at.isoformat(),
            updated_at=issue.updated_at.isoformat(),
            labels=[label.name for label in issue.labels],
            assignees=[assignee.login for assignee in issue.assignees],
            repository=repo_name
        )
    except GithubException as e:
        status = getattr(e, 'status', None)
        if status == 404:
            logger.error(f"GitHub issue not found: {repo_name}#{issue_number}")
            raise ValueError(f"Issue {issue_number} not found in repository {repo_name}")
        logger.error(f"GitHub API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error getting GitHub issue: {e}", exc_info=True)
        raise


async def get_pull_request(repo_name: str, pr_number: int) -> GitHubPR:
    """
    Get a GitHub pull request by repository and PR number.
    
    Args:
        repo_name: The name of the repository (e.g., "username/repo")
        pr_number: The pull request number
        
    Returns:
        GitHubPR object with pull request data
    """
    try:
        client = get_github_client()
        repo = client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        
        return GitHubPR(
            id=pr.id,
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            state=pr.state,
            html_url=pr.html_url,
            created_at=pr.created_at.isoformat(),
            updated_at=pr.updated_at.isoformat(),
            labels=[label.name for label in pr.labels],
            assignees=[assignee.login for assignee in pr.assignees],
            repository=repo_name,
            branch=pr.head.ref,
            merged=pr.merged
        )
    except GithubException as e:
        status = getattr(e, 'status', None)
        if status == 404:
            logger.error(f"GitHub PR not found: {repo_name}#{pr_number}")
            raise ValueError(f"Pull request {pr_number} not found in repository {repo_name}")
        logger.error(f"GitHub API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error getting GitHub pull request: {e}", exc_info=True)
        raise


async def get_file(repo_name: str, file_path: str, branch: Optional[str] = None) -> GitHubFile:
    """
    Get a file from a GitHub repository.
    
    Args:
        repo_name: The name of the repository (e.g., "username/repo")
        file_path: The path to the file in the repository
        branch: The branch name (default: repository's default branch)
        
    Returns:
        GitHubFile object with file data
    """
    try:
        client = get_github_client()
        repo = client.get_repo(repo_name)
        
        try:
            file_content = repo.get_contents(file_path, ref=branch)
            
            # Handle case where get_contents returns a list (directory)
            if isinstance(file_content, list):
                raise ValueError(f"{file_path} is a directory, not a file")
                
            # Get the raw content
            if file_content.encoding == "base64":
                content = file_content.decoded_content.decode('utf-8')
            else:
                content = file_content.content
                
            return GitHubFile(
                name=file_content.name,
                path=file_content.path,
                content=content,
                html_url=file_content.html_url,
                repository=repo_name,
                sha=file_content.sha
            )
            
        except GithubException as e:
            if e.status == 404:
                raise ValueError(f"File {file_path} not found in repository {repo_name}")
            raise
            
    except ValueError:
        # Re-raise ValueError
        raise
    except GithubException as e:
        logger.error(f"GitHub API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error getting GitHub file: {e}", exc_info=True)
        raise


async def create_issue(repo_name: str, title: str, body: str, labels: Optional[List[str]] = None) -> GitHubIssue:
    """
    Create a new GitHub issue.
    
    Args:
        repo_name: The name of the repository (e.g., "username/repo")
        title: The issue title
        body: The issue body/description
        labels: Optional list of label names to apply
        
    Returns:
        GitHubIssue object for the created issue
    """
    try:
        client = get_github_client()
        repo = client.get_repo(repo_name)
        
        # Create the issue
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=labels or []
        )
        
        return GitHubIssue(
            id=issue.id,
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            state=issue.state,
            html_url=issue.html_url,
            created_at=issue.created_at.isoformat(),
            updated_at=issue.updated_at.isoformat(),
            labels=[label.name for label in issue.labels],
            assignees=[assignee.login for assignee in issue.assignees],
            repository=repo_name
        )
    except GithubException as e:
        logger.error(f"GitHub API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error creating GitHub issue: {e}", exc_info=True)
        raise


async def search_issues(query: str, max_results: int = 5) -> List[GitHubIssue]:
    """
    Search for GitHub issues using the GitHub search API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of GitHubIssue objects matching the query
    """
    try:
        client = get_github_client()
        
        # Perform the search
        issues_data = client.search_issues(query=query)
        results = []
        
        # Process results
        for i, issue in enumerate(issues_data):
            if i >= max_results:
                break
                
            # Extract repository name from issue
            repo_name = issue.repository.full_name
            
            results.append(GitHubIssue(
                id=issue.id,
                number=issue.number,
                title=issue.title,
                body=issue.body or "",
                state=issue.state,
                html_url=issue.html_url,
                created_at=issue.created_at.isoformat(),
                updated_at=issue.updated_at.isoformat(),
                labels=[label.name for label in issue.labels],
                assignees=[assignee.login for assignee in issue.assignees],
                repository=repo_name
            ))
            
        return results
    except GithubException as e:
        logger.error(f"GitHub API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error searching GitHub issues: {e}", exc_info=True)
        raise


async def create_pull_request(
    repo_name: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str = "main",
    draft: bool = False,
    labels: Optional[List[str]] = None
) -> GitHubPR:
    """
    Create a new GitHub pull request.
    
    Args:
        repo_name: The name of the repository (e.g., "username/repo")
        title: The PR title
        body: The PR description
        head_branch: The name of the branch with changes
        base_branch: The name of the branch to merge into (default: main)
        draft: Whether to create as a draft PR
        labels: Optional list of label names to apply
        
    Returns:
        GitHubPR object for the created pull request
    """
    try:
        client = get_github_client()
        repo = client.get_repo(repo_name)
        
        # Create the pull request
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=draft
        )
        
        # Add labels if provided
        if labels:
            pr.add_to_labels(*labels)
        
        return GitHubPR(
            id=pr.id,
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            state=pr.state,
            html_url=pr.html_url,
            created_at=pr.created_at.isoformat(),
            updated_at=pr.updated_at.isoformat(),
            labels=[label.name for label in pr.labels],
            assignees=[assignee.login for assignee in pr.assignees],
            repository=repo_name,
            branch=pr.head.ref,
            merged=pr.merged
        )
    except GithubException as e:
        logger.error(f"GitHub API error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error creating GitHub pull request: {e}", exc_info=True)
        raise


async def sync_github_data(user_id: str) -> Dict[str, Any]:
    """
    Synchronize GitHub data for a user to the vector store.
    This function fetches data from GitHub and prepares it for embedding.
    
    Args:
        user_id: The ID of the user to sync data for
        
    Returns:
        A dictionary with sync results
    """
    try:
        logger.info(f"Starting GitHub data sync for user {user_id}")
        client = get_github_client()
        
        # Get user's repositories
        user = client.get_user()
        repos = list(user.get_repos())
        
        synced_issues = []
        synced_prs = []
        errors = []
        
        # For each repo, get issues and PRs
        for repo in repos:
            repo_name = repo.full_name
            logger.debug(f"Processing repo: {repo_name}")
            
            # Get issues
            try:
                issues = list(repo.get_issues(state="all", sort="updated", direction="desc")[:10])
                for issue in issues:
                    if not issue.pull_request:  # Skip PRs, they'll be handled separately
                        try:
                            issue_data = await get_issue(repo_name, issue.number)
                            synced_issues.append(issue_data)
                        except Exception as e:
                            logger.error(f"Error syncing issue {repo_name}#{issue.number}: {e}")
                            errors.append({"type": "issue", "repo": repo_name, "number": issue.number, "error": str(e)})
            except Exception as e:
                logger.error(f"Error getting issues for repo {repo_name}: {e}")
                errors.append({"type": "repo_issues", "repo": repo_name, "error": str(e)})
                
            # Get PRs
            try:
                prs = list(repo.get_pulls(state="all", sort="updated", direction="desc")[:10])
                for pr in prs:
                    try:
                        pr_data = await get_pull_request(repo_name, pr.number)
                        synced_prs.append(pr_data)
                    except Exception as e:
                        logger.error(f"Error syncing PR {repo_name}#{pr.number}: {e}")
                        errors.append({"type": "pr", "repo": repo_name, "number": pr.number, "error": str(e)})
            except Exception as e:
                logger.error(f"Error getting PRs for repo {repo_name}: {e}")
                errors.append({"type": "repo_prs", "repo": repo_name, "error": str(e)})
        
        # TODO: Add code to embed and store the synced GitHub data in a vector store
        
        return {
            "status": "success",
            "user_id": user_id,
            "repos_processed": len(repos),
            "issues_synced": len(synced_issues),
            "prs_synced": len(synced_prs),
            "errors": len(errors),
            "error_details": errors if errors else None
        }
    except Exception as e:
        logger.error(f"Error syncing GitHub data for user {user_id}: {e}", exc_info=True)
        raise
