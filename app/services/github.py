"""
GitHub API service for AI Task Copilot.
"""
import asyncio
from functools import lru_cache
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
from threading import Lock

from github import Github, GithubException, Auth
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


# Global rate limiter state
github_rate_limiter = {"last_call": datetime.now() - timedelta(seconds=0.5), "lock": Lock()}


@lru_cache()
def get_github_client() -> Github:
    """
    Create and return a GitHub client instance.
    Uses lru_cache to ensure only one client is created.
    """
    try:
        logger.debug("Attempting to create GitHub client...")
        auth = Auth.Token(settings.GITHUB_API_TOKEN)
        logger.debug("GitHub Auth object created.")
        client = Github(auth=auth)
        logger.debug("GitHub client object created successfully.")
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
        loop = asyncio.get_running_loop()
        
        repo_obj = await loop.run_in_executor(None, client.get_repo, repo_name)
        issue = await loop.run_in_executor(None, repo_obj.get_issue, issue_number)
        
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
        loop = asyncio.get_running_loop()

        repo_obj = await loop.run_in_executor(None, client.get_repo, repo_name)
        pr = await loop.run_in_executor(None, repo_obj.get_pull, pr_number)
        
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
        loop = asyncio.get_running_loop()

        repo_obj = await loop.run_in_executor(None, client.get_repo, repo_name)
        
        def _get_contents_sync(repo_object, path, ref_branch):
            return repo_object.get_contents(path, ref=ref_branch)

        file_content = await loop.run_in_executor(None, _get_contents_sync, repo_obj, file_path, branch)
            
        if isinstance(file_content, list):
            raise ValueError(f"{file_path} is a directory, not a file")
            
        if file_content.encoding == "base64":
            content = await loop.run_in_executor(None, file_content.decoded_content.decode, 'utf-8')
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
            raise ValueError(f"File {file_path} not found in repository {repo_name} on branch {branch or 'default'}")
        logger.error(f"GitHub API error getting file {file_path} from {repo_name}: {e}", exc_info=True)
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error getting GitHub file {file_path} from {repo_name}: {e}", exc_info=True)
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
        loop = asyncio.get_running_loop()

        repo_obj = await loop.run_in_executor(None, client.get_repo, repo_name)
        
        created_issue = await loop.run_in_executor(
            None,
            repo_obj.create_issue,
            title,
            body=body,
            labels=labels or []
        )
        
        return GitHubIssue(
            id=created_issue.id,
            number=created_issue.number,
            title=created_issue.title,
            body=created_issue.body or "",
            state=created_issue.state,
            html_url=created_issue.html_url,
            created_at=created_issue.created_at.isoformat(),
            updated_at=created_issue.updated_at.isoformat(),
            labels=[label.name for label in created_issue.labels],
            assignees=[assignee.login for assignee in created_issue.assignees],
            repository=repo_name
        )
    except GithubException as e:
        logger.error(f"GitHub API error creating issue in {repo_name}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error creating GitHub issue in {repo_name}: {e}", exc_info=True)
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
        loop = asyncio.get_running_loop()
        
        issues_paginated_list = await loop.run_in_executor(None, client.search_issues, query=query)
        
        results = []
        count = 0
        for issue in issues_paginated_list:
            if count >= max_results:
                break
            
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
            count += 1
            
        return results
    except GithubException as e:
        logger.error(f"GitHub API error searching issues with query '{query}': {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error searching GitHub issues with query '{query}': {e}", exc_info=True)
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
    """
    try:
        client = get_github_client()
        loop = asyncio.get_running_loop()

        repo_obj = await loop.run_in_executor(None, client.get_repo, repo_name)
        
        created_pr = await loop.run_in_executor(
            None,
            repo_obj.create_pull,
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=draft
        )
        
        if labels:
            await loop.run_in_executor(None, created_pr.add_to_labels, *labels)
        
        return GitHubPR(
            id=created_pr.id,
            number=created_pr.number,
            title=created_pr.title,
            body=created_pr.body or "",
            state=created_pr.state,
            html_url=created_pr.html_url,
            created_at=created_pr.created_at.isoformat(),
            updated_at=created_pr.updated_at.isoformat(),
            labels=[label.name for label in created_pr.labels],
            assignees=[assignee.login for assignee in created_pr.assignees],
            repository=repo_name,
            branch=created_pr.head.ref,
            merged=created_pr.merged
        )
    except GithubException as e:
        logger.error(f"GitHub API error creating PR in {repo_name}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error creating GitHub pull request in {repo_name}: {e}", exc_info=True)
        raise


async def sync_github_data(user_id: str) -> Dict[str, Any]:
    """
    Synchronize GitHub data for a user to the vector store.
    This function fetches data from GitHub and prepares it for embedding.
    """
    try:
        logger.info(f"Starting GitHub data sync for user {user_id}")
        logger.debug(f"[{user_id}] Calling get_github_client()...")
        client = get_github_client()
        logger.debug(f"[{user_id}] get_github_client() returned. Client: {'present' if client else 'None'}")
        
        logger.debug(f"[{user_id}] Attempting to get running asyncio loop...")
        loop = asyncio.get_running_loop()
        logger.debug(f"[{user_id}] Successfully got asyncio loop.")
        
        logger.debug(f"[{user_id}] Attempting to get GitHub user...")
        user = await loop.run_in_executor(None, client.get_user)
        logger.debug(f"[{user_id}] Successfully got GitHub user: {user.login if user else 'None'}")
        
        if not user:
            logger.error(f"[{user_id}] Failed to retrieve GitHub user object.")
            return {
                "status": "failed",
                "user_id": user_id,
                "message": "Failed to retrieve GitHub user object.",
                "repos_processed": 0,
                "issues_synced": 0,
                "prs_synced": 0,
                "errors_count": 1,
                "error_details": [{"type": "critical_sync_failure", "repo": "N/A", "error": "GitHub user object is None"}]
            }

        def _get_all_repos_sync(user_obj):
            logger.debug(f"[{user_id}] Executor: Getting repos for user {user_obj.login}")
            repos_list = list(user_obj.get_repos())
            logger.debug(f"[{user_id}] Executor: Found {len(repos_list)} repos for user {user_obj.login}")
            return repos_list
        
        logger.debug(f"[{user_id}] Attempting to get repositories for user {user.login}...")
        repos = await loop.run_in_executor(None, _get_all_repos_sync, user)
        logger.debug(f"[{user_id}] Successfully got {len(repos)} repositories for user {user.login}.")
        
        synced_issues = []
        synced_prs = []
        errors = []
        
        # Process repositories in parallel
        repo_tasks = []
        for repo in repos:
            repo_tasks.append(process_repository(repo, user_id, synced_issues, synced_prs, errors))
        
        # Run all repo processing tasks concurrently
        await asyncio.gather(*repo_tasks)
        
        final_status = "success"
        if errors:
            final_status = "partial_success"

        return {
            "status": final_status,
            "user_id": user_id,
            "repos_processed": len(repos),
            "issues_synced": len(synced_issues),
            "prs_synced": len(synced_prs),
            "errors_count": len(errors),
            "error_details": errors if errors else None
        }
    except Exception as e:
        logger.error(f"Critical error during GitHub data sync for user {user_id}: {e}", exc_info=True)
        return {
            "status": "failed",
            "user_id": user_id,
            "message": str(e),
            "repos_processed": 0,
            "issues_synced": 0,
            "prs_synced": 0,
            "errors_count": 1,
            "error_details": [{"type": "critical_sync_failure", "repo": "N/A", "error": str(e)}]
        }


async def process_repository(repo, user_id, synced_issues, synced_prs, errors):
    repo_name = repo.full_name
    logger.debug(f"[{user_id}] Processing repo: {repo_name}")
    
    try:
        # Run issues and PRs processing concurrently
        issues_task = asyncio.create_task(process_issues(repo, repo_name, synced_issues))
        prs_task = asyncio.create_task(process_prs(repo, repo_name, synced_prs))
        await asyncio.gather(issues_task, prs_task)
        
    except Exception as e:
        logger.error(f"Error processing repository {repo_name}: {e}", exc_info=True)
        errors.append({"type": "repo_processing", "repo": repo_name, "error": str(e)})


async def process_issues(repo, repo_name, synced_issues):
    issues_list = await asyncio.get_running_loop().run_in_executor(
        None, 
        lambda: make_rate_limited_call(lambda: repo.get_issues(state="all", sort="updated", direction="desc"))
    )
    # Process issues
    for issue in issues_list:
        synced_issues.append(GitHubIssue(
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


async def process_prs(repo, repo_name, synced_prs):
    prs_list = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: make_rate_limited_call(lambda: repo.get_pulls(state="all", sort="updated", direction="desc"))
    )
    # Process PRs
    for pr in prs_list:
        synced_prs.append(GitHubPR(
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
        ))


def make_rate_limited_call(callable_fn):
    with github_rate_limiter["lock"]:
        time_since_last = (datetime.now() - github_rate_limiter["last_call"]).total_seconds()
        if time_since_last < 0.5:  # Allow 2 requests per second
            sleep_time = 0.5 - time_since_last
            time.sleep(sleep_time)
        result = callable_fn()
        github_rate_limiter["last_call"] = datetime.now()
    return result
