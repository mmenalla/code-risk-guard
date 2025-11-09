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
from datetime import datetime, timezone, timedelta
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
    
    def calculate_temporal_features(
        self,
        start_date: datetime,
        end_date: datetime,
        max_commits: int = 10000
    ) -> pd.DataFrame:
        """
        Calculate Git features for a specific time window (temporal analysis).
        
        This is the main method used for degradation prediction - it analyzes
        commits within a date range to understand code activity patterns.
        
        Args:
            start_date: Beginning of time window
            end_date: End of time window
            max_commits: Maximum commits to analyze
            
        Returns:
            DataFrame with Git features per file for the time window
        """
        logger.info(f"Calculating temporal features from {start_date.date()} to {end_date.date()}")
        
        days_delta = (end_date - start_date).days
        
        # Get commits in date range
        commits = list(self.repo.iter_commits(
            self.branch,
            max_count=max_commits,
            since=start_date.strftime('%Y-%m-%d'),
            until=end_date.strftime('%Y-%m-%d')
        ))
        
        logger.info(f"Found {len(commits)} commits in date range")
        
        if not commits:
            logger.warning("No commits found in specified date range")
            return pd.DataFrame()
        
        # Extract features from commits
        features_df = self._extract_temporal_features(commits, days_delta)
        
        return features_df
    
    def _extract_temporal_features(
        self, 
        commits: List[Commit], 
        days_in_window: int
    ) -> pd.DataFrame:
        """
        Extract Git features from commits in a time window.
        
        Args:
            commits: List of Git commit objects
            days_in_window: Number of days in the time window
            
        Returns:
            DataFrame with features per file
        """
        from collections import defaultdict
        
        file_stats = defaultdict(lambda: {
            'commits': 0,
            'authors': set(),
            'lines_added': 0,
            'lines_deleted': 0,
            'churn': 0,
            'bug_related': 0,
            'refactor_related': 0,
            'feature_related': 0,
        })
        
        for commit in commits:
            message = commit.message.lower()
            is_bug = any(keyword in message for keyword in ['fix', 'bug', 'issue', 'error'])
            is_refactor = any(keyword in message for keyword in ['refactor', 'cleanup', 'improve'])
            is_feature = any(keyword in message for keyword in ['feat', 'add', 'implement', 'new'])
            
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
                
                for diff in diffs:
                    if diff.b_path:
                        filepath = diff.b_path
                    elif diff.a_path:
                        filepath = diff.a_path
                    else:
                        continue
                    
                    if not filepath or filepath == '/dev/null':
                        continue
                    
                    if not self._is_source_file(filepath):
                        continue
                    
                    # Count line changes
                    insertions = 0
                    deletions = 0
                    
                    if diff.diff:
                        diff_text = diff.diff.decode('utf-8', errors='ignore')
                        for line in diff_text.split('\n'):
                            if line.startswith('+') and not line.startswith('+++'):
                                insertions += 1
                            elif line.startswith('-') and not line.startswith('---'):
                                deletions += 1
                    
                    file_stats[filepath]['commits'] += 1
                    file_stats[filepath]['authors'].add(commit.author.name)
                    file_stats[filepath]['lines_added'] += insertions
                    file_stats[filepath]['lines_deleted'] += deletions
                    file_stats[filepath]['churn'] += insertions + deletions
                    
                    if is_bug:
                        file_stats[filepath]['bug_related'] += 1
                    if is_refactor:
                        file_stats[filepath]['refactor_related'] += 1
                    if is_feature:
                        file_stats[filepath]['feature_related'] += 1
        
        # Convert to DataFrame
        rows = []
        for file_path, stats in file_stats.items():
            num_authors = len(stats['authors'])
            num_commits = stats['commits']
            
            rows.append({
                'module': file_path,
                'commits': num_commits,
                'authors': num_authors,
                'lines_added': stats['lines_added'],
                'lines_deleted': stats['lines_deleted'],
                'churn': stats['churn'],
                'bug_commits': stats['bug_related'],
                'refactor_commits': stats['refactor_related'],
                'feature_commits': stats['feature_related'],
                'lines_per_author': stats['lines_added'] / num_authors if num_authors > 0 else 0,
                'churn_per_commit': stats['churn'] / num_commits if num_commits > 0 else 0,
                'bug_ratio': stats['bug_related'] / num_commits if num_commits > 0 else 0,
                'days_active': days_in_window,
                'commits_per_day': num_commits / days_in_window if days_in_window > 0 else 0,
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"Extracted features for {len(df)} files from temporal window")
        
        return df
    
    def fetch_commit_data(
        self, 
        max_commits: int = 300, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        since_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch commit data and aggregate by file with temporal window support.
        
        Args:
            max_commits: Maximum number of commits to analyze
            start_date: Start of temporal window (optional)
            end_date: End of temporal window (optional)
            since_days: Only analyze commits from last N days (legacy, optional)
        
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
        logger.info(f"Max commits: {max_commits}")
        if start_date and end_date:
            logger.info(f"Temporal window: {start_date} to {end_date}")
        elif since_days:
            logger.info(f"Since days: {since_days}")
        
        # Get commits from the current branch
        commits = list(self.repo.iter_commits(self.branch, max_count=max_commits))
        logger.info(f"Found {len(commits)} commits")
        
        # Filter by temporal window if specified
        if start_date and end_date:
            # Ensure dates are timezone-aware for comparison
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            
            commits = [c for c in commits 
                      if start_date <= c.committed_datetime.astimezone(timezone.utc) <= end_date]
            logger.info(f"Filtered to {len(commits)} commits in temporal window")
        
        # Legacy: filter by since_days if specified (for backward compatibility)
        elif since_days:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)
            commits = [c for c in commits 
                      if c.committed_datetime.astimezone(timezone.utc) > cutoff_date]
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
            # Calculate base features with NEW consolidated column names
            num_authors = len(stats['authors'])
            num_commits = stats['commits']
            days_active_val = max((stats['last_commit'] - stats['first_commit']).days, 1)
            
            rows.append({
                'module': filepath,
                'commits': num_commits,
                'authors': num_authors,
                'lines_added': stats['lines_added'],
                'lines_deleted': stats['lines_removed'],  # Keep internal stats['lines_removed'], but output as 'lines_deleted'
                'churn': stats['lines_added'] + stats['lines_removed'],
                'bug_commits': stats['bug_commits'],
                'refactor_commits': stats.get('refactor_commits', 0),
                'feature_commits': stats.get('feature_commits', 0),
                'lines_per_author': stats['lines_added'] / num_authors if num_authors > 0 else 0,
                'churn_per_commit': (stats['lines_added'] + stats['lines_removed']) / num_commits if num_commits > 0 else 0,
                'bug_ratio': stats['bug_commits'] / num_commits if num_commits > 0 else 0,
                'days_active': days_active_val,
                'commits_per_day': num_commits / days_active_val,
                'repo_name': repo_name
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"Extracted data for {len(df)} files")
        logger.info(f"Total commits analyzed: {df['commits'].sum():.0f}")
        logger.info(f"Total lines added: {df['lines_added'].sum():.0f}")
        logger.info(f"Total lines deleted: {df['lines_deleted'].sum():.0f}")
        
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
    
    def get_commits_in_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        max_commits: int = 10000
    ) -> List[Commit]:
        """
        Get commits within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            max_commits: Maximum commits to retrieve
        
        Returns:
            List of Commit objects
        """
        try:
            # Format dates for git log
            after = start_date.strftime('%Y-%m-%d')
            before = end_date.strftime('%Y-%m-%d')
            
            commits = list(self.repo.iter_commits(
                self.branch,
                after=after,
                before=before,
                max_count=max_commits
            ))
            
            logger.info(f"Found {len(commits)} commits between {after} and {before}")
            return commits
            
        except Exception as e:
            logger.error(f"Error getting commits in date range: {e}")
            return []
    
    def calculate_features_from_commits(self, commits: List[Commit]) -> pd.DataFrame:
        """
        Calculate file-level features from a list of commits.
        
        Args:
            commits: List of Commit objects
        
        Returns:
            DataFrame with features per file
        """
        file_stats = {}
        
        for commit in commits:
            try:
                # Check if commit has parents (skip initial commit)
                if not commit.parents:
                    continue
                
                # Get diff stats
                diff = commit.parents[0].diff(commit, create_patch=False)
                
                for change in diff:
                    # Skip deleted files
                    if change.deleted_file:
                        continue
                    
                    filepath = change.b_path if change.b_path else change.a_path
                    
                    if filepath not in file_stats:
                        file_stats[filepath] = {
                            'lines_added': 0,
                            'lines_removed': 0,
                            'commits': 0,
                            'authors': set(),
                            'bug_commits': 0,
                            'refactor_commits': 0,
                            'feature_commits': 0,
                            'first_commit': commit.committed_datetime,
                            'last_commit': commit.committed_datetime
                        }
                    
                    stats = file_stats[filepath]
                    
                    # Update stats
                    if hasattr(change.diff, 'decode'):
                        # Count lines added/removed from diff
                        diff_text = change.diff.decode('utf-8', errors='ignore')
                        for line in diff_text.split('\n'):
                            if line.startswith('+') and not line.startswith('+++'):
                                stats['lines_added'] += 1
                            elif line.startswith('-') and not line.startswith('---'):
                                stats['lines_removed'] += 1
                    
                    stats['commits'] += 1
                    stats['authors'].add(commit.author.email)
                    
                    # Classify commit type
                    message_lower = commit.message.lower()
                    if any(word in message_lower for word in ['fix', 'bug', 'patch', 'hotfix']):
                        stats['bug_commits'] += 1
                    elif any(word in message_lower for word in ['refactor', 'clean', 'improve']):
                        stats['refactor_commits'] += 1
                    elif any(word in message_lower for word in ['feat', 'feature', 'add', 'new']):
                        stats['feature_commits'] += 1
                    
                    # Update timestamps
                    if commit.committed_datetime > stats['last_commit']:
                        stats['last_commit'] = commit.committed_datetime
                    if commit.committed_datetime < stats['first_commit']:
                        stats['first_commit'] = commit.committed_datetime
                        
            except Exception as e:
                logger.warning(f"Error processing commit {commit.hexsha[:8]}: {e}")
                continue
        
        # Convert to DataFrame
        rows = []
        for filepath, stats in file_stats.items():
            row = {
                'module': filepath,
                'commits': stats['commits'],
                'authors': len(stats['authors']),
                'lines_added': stats['lines_added'],
                'lines_deleted': stats['lines_deleted'],
                'churn': stats['lines_added'] + stats['lines_deleted'],
                'bug_commits': stats['bug_commits'],
                'refactor_commits': stats['refactor_commits'],
                'feature_commits': stats['feature_commits'],
                'lines_per_author': stats['lines_added'] / len(stats['authors']) if stats['authors'] else 0,
                'churn_per_commit': (stats['lines_added'] + stats['lines_deleted']) / stats['commits'] if stats['commits'] > 0 else 0,
                'bug_ratio': stats['bug_commits'] / stats['commits'] if stats['commits'] > 0 else 0,
                'days_active': (stats['last_commit'] - stats['first_commit']).days,
                'commits_per_day': stats['commits'] / max((stats['last_commit'] - stats['first_commit']).days, 1)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Calculated features for {len(df)} files")
        
        return df
    
    def calculate_temporal_features(
        self,
        start_date: datetime,
        end_date: datetime,
        max_commits: int = 10000
    ) -> pd.DataFrame:
        """
        Calculate features for a specific temporal window.
        
        This is the main method used by the training DAG for temporal analysis.
        
        Args:
            start_date: Start of temporal window
            end_date: End of temporal window
            max_commits: Maximum commits to process
            
        Returns:
            DataFrame with calculated features for the temporal window
        """
        logger.info(f"Calculating temporal features from {start_date} to {end_date}")
        
        # Use the enhanced fetch_commit_data with temporal filtering
        df = self.fetch_commit_data(
            max_commits=max_commits,
            start_date=start_date,
            end_date=end_date
        )
        
        # Note: fetch_commit_data already calculates all base features including:
        # lines_per_author, churn_per_commit, bug_ratio, days_active, commits_per_day
        
        # Convert any timestamp columns to strings for JSON serialization
        timestamp_columns = ['created_at', 'last_modified']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        logger.info(f"Temporal features calculated for {len(df)} files")
        return df
    
    @staticmethod
    def get_historical_date(days_back: int, reference_date: Optional[datetime] = None) -> datetime:
        """
        Calculate a historical date relative to a reference point.
        
        Args:
            days_back: Number of days to go back
            reference_date: Reference date (defaults to now)
            
        Returns:
            Historical datetime
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)
        
        return reference_date - timedelta(days=days_back)


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
