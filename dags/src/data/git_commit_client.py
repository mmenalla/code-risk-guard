"""
Git Commit Data Collector

Extracts commit data from local git repository as an alternative to GitHub PR analysis.
This allows risk analysis without GitHub API dependency.

Usage:
    collector = GitCommitCollector(repo_path="/path/to/repo", branch="main")
    df = collector.fetch_commit_data(max_commits=300)
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

try:
    from git import Repo, Commit
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logging.warning("GitPython not installed. Install with: pip install GitPython")


logger = logging.getLogger(__name__)


class GitCommitCollector:
    """
    Collects commit data from local git repository.
    
    Extracts file-level statistics similar to PR analysis but from commits:
    - Lines added/removed per file
    - File modifications over time
    - Author activity
    - Commit messages (for bug detection)
    """
    
    def __init__(self, repo_path: str, branch: str = "main"):
        """
        Initialize Git Commit Collector.
        
        Args:
            repo_path: Absolute path to local git repository
            branch: Branch name to analyze (default: main)
        """
        if not GIT_AVAILABLE:
            raise ImportError("GitPython is required. Install with: pip install GitPython")
        
        self.repo_path = Path(repo_path)
        self.branch = branch
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        try:
            self.repo = Repo(self.repo_path)
            logger.info(f"Initialized git repository: {self.repo_path}")
        except Exception as e:
            raise ValueError(f"Invalid git repository: {e}")
        
        # Verify branch exists (no checkout needed - read-only access)
        if branch not in self.repo.heads:
            available_branches = [h.name for h in self.repo.heads]
            raise ValueError(f"Branch '{branch}' not found. Available: {available_branches}")
        
        logger.info(f"Using branch: {branch}")
    
    def fetch_commit_data(self, max_commits: int = 300, since_days: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch commit data and aggregate by file.
        
        Args:
            max_commits: Maximum number of commits to analyze
            since_days: Only analyze commits from last N days (optional)
        
        Returns:
            DataFrame with columns:
                - filename: File path
                - lines_added: Total lines added
                - lines_removed: Total lines removed  
                - commits: Number of commits touching this file
                - unique_authors: Number of unique authors
                - bug_commits: Commits with 'fix', 'bug', 'patch' in message
                - created_at: First commit timestamp
                - last_modified: Last commit timestamp
                - repo_name: Repository name
        """
        logger.info(f"Fetching commits from {self.repo_path} (branch: {self.branch})")
        logger.info(f"Max commits: {max_commits}, Since days: {since_days}")
        
        # Get commits from the current branch
        commits = list(self.repo.iter_commits(self.branch, max_count=max_commits))
        logger.info(f"Found {len(commits)} commits")
        
        if since_days:
            from datetime import timedelta
            cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None)
            cutoff_date = cutoff_date - timedelta(days=since_days)
            # Convert commit dates to naive UTC for comparison
            commits = [c for c in commits 
                      if (c.committed_datetime.astimezone(timezone.utc).replace(tzinfo=None) > cutoff_date)]
            logger.info(f"Filtered to {len(commits)} commits since {since_days} days ago")
        
        # Aggregate file-level stats
        file_stats = {}
        
        for commit in commits:
            # Convert commit datetime to UTC to avoid timezone issues with parquet
            commit_date = commit.committed_datetime
            if commit_date.tzinfo is not None:
                commit_date = commit_date.astimezone(timezone.utc).replace(tzinfo=None)
            
            author = commit.author.email
            message = commit.message.lower()
            is_bug_fix = any(keyword in message for keyword in ['fix', 'bug', 'patch', 'hotfix', 'bugfix'])
            
            # Get parent commit for diff
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
                
                for diff in diffs:
                    # Handle renamed files
                    if diff.b_path:
                        filepath = diff.b_path
                    elif diff.a_path:
                        filepath = diff.a_path
                    else:
                        continue
                    
                    # Skip non-source files
                    if not self._is_source_file(filepath):
                        continue
                    
                    # Initialize file stats if first time seeing this file
                    if filepath not in file_stats:
                        file_stats[filepath] = {
                            'lines_added': 0,
                            'lines_removed': 0,
                            'commits': 0,
                            'authors': set(),
                            'bug_commits': 0,
                            'first_commit': commit_date,
                            'last_commit': commit_date
                        }
                    
                    stats = file_stats[filepath]
                    
                    # Count line changes
                    if diff.diff:
                        diff_text = diff.diff.decode('utf-8', errors='ignore')
                        lines_added = sum(1 for line in diff_text.split('\n') if line.startswith('+') and not line.startswith('+++'))
                        lines_removed = sum(1 for line in diff_text.split('\n') if line.startswith('-') and not line.startswith('---'))
                        
                        stats['lines_added'] += lines_added
                        stats['lines_removed'] += lines_removed
                    
                    # Update stats
                    stats['commits'] += 1
                    stats['authors'].add(author)
                    if is_bug_fix:
                        stats['bug_commits'] += 1
                    
                    # Update timestamps
                    if commit_date < stats['first_commit']:
                        stats['first_commit'] = commit_date
                    if commit_date > stats['last_commit']:
                        stats['last_commit'] = commit_date
        
        # Convert to DataFrame
        if not file_stats:
            logger.warning("No source files found in commits")
            return pd.DataFrame()
        
        rows = []
        repo_name = self.repo_path.name
        
        for filepath, stats in file_stats.items():
            rows.append({
                'filename': filepath,
                'lines_added': stats['lines_added'],
                'lines_removed': stats['lines_removed'],
                'churn': stats['lines_added'] + stats['lines_removed'],  # Total code churn
                'prs': stats['commits'],  # Use 'prs' for compatibility with existing pipeline
                'unique_authors': len(stats['authors']),
                'bug_prs': stats['bug_commits'],  # Use 'bug_prs' for compatibility
                'created_at': stats['first_commit'],
                'last_modified': stats['last_commit'],
                'repo_name': repo_name,
                'module': filepath  # For compatibility
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"Extracted data for {len(df)} files")
        logger.info(f"Total commits analyzed: {sum(df['prs'])}")
        logger.info(f"Total lines added: {df['lines_added'].sum()}")
        logger.info(f"Total lines removed: {df['lines_removed'].sum()}")
        
        return df
    
    def _is_source_file(self, filepath: str) -> bool:
        """
        Check if file is a source code file (not test, config, or generated file).
        
        Args:
            filepath: File path to check
        
        Returns:
            True if source file, False otherwise
        """
        # Skip test files
        if any(test_dir in filepath.lower() for test_dir in ['test', '__test__', 'tests', '__tests__', 'spec']):
            return False
        
        # Skip common non-source paths
        skip_paths = [
            'node_modules/', 'vendor/', 'dist/', 'build/', '.git/',
            'coverage/', '.next/', '__pycache__/', '.pytest_cache/',
            'target/', 'out/', 'bin/', 'obj/'
        ]
        if any(skip_path in filepath for skip_path in skip_paths):
            return False
        
        # Only include source code extensions
        source_extensions = {
            # Programming languages
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.kt', '.scala',
            '.go', '.rs', '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php',
            '.swift', '.m', '.mm', '.r', '.jl', '.lua', '.pl', '.sh',
            # Web
            '.vue', '.svelte', '.html', '.css', '.scss', '.sass', '.less',
            # Data/Config that matters
            '.sql', '.graphql', '.proto'
        }
        
        ext = Path(filepath).suffix.lower()
        return ext in source_extensions
    
    def get_file_history(self, filepath: str, max_commits: int = 100) -> List[Dict]:
        """
        Get detailed commit history for a specific file.
        
        Args:
            filepath: File path relative to repository root
            max_commits: Maximum commits to retrieve
        
        Returns:
            List of commit dictionaries with metadata
        """
        try:
            commits = list(self.repo.iter_commits(self.branch, paths=filepath, max_count=max_commits))
            
            history = []
            for commit in commits:
                history.append({
                    'sha': commit.hexsha[:8],
                    'author': commit.author.name,
                    'email': commit.author.email,
                    'date': commit.committed_datetime,
                    'message': commit.message.strip(),
                    'files_changed': len(commit.stats.files)
                })
            
            return history
        except Exception as e:
            logger.error(f"Error getting history for {filepath}: {e}")
            return []
    
    def get_file_age_days(self, filepath: str) -> Optional[int]:
        """
        Get age of file in days (from first commit).
        
        Args:
            filepath: File path relative to repository root
        
        Returns:
            Age in days, or None if file not found
        """
        try:
            commits = list(self.repo.iter_commits(self.branch, paths=filepath))
            if not commits:
                return None
            
            first_commit = commits[-1]  # Last in list is oldest
            age = datetime.now(timezone.utc) - first_commit.committed_datetime
            return age.days
        except Exception as e:
            logger.error(f"Error getting age for {filepath}: {e}")
            return None


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python git_commit_client.py /path/to/repo [branch]")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    branch = sys.argv[2] if len(sys.argv) > 2 else "main"
    
    collector = GitCommitCollector(repo_path, branch)
    df = collector.fetch_commit_data(max_commits=100)
    
    print(f"\n📊 Commit Data Summary:")
    print(f"   Files analyzed: {len(df)}")
    print(f"   Total commits: {df['prs'].sum()}")
    print(f"   Unique authors: {df['unique_authors'].sum()}")
    print(f"   Bug fix commits: {df['bug_prs'].sum()}")
    print(f"\nTop 10 files by commit count:")
    print(df.nlargest(10, 'prs')[['filename', 'prs', 'lines_added', 'lines_removed']])


if __name__ == "__main__":
    main()
