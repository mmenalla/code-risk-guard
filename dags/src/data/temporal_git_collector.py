"""
Temporal Git Data Collector for Degradation Analysis

Collects Git features from a specific time window (e.g., 100 days)
to enable temporal prediction of quality degradation.

Key difference from regular GitCommitClient:
- Filters commits to a specific date range (from historical_date to current_date)
- Used to collect features that PRECEDED the quality change
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from git import Repo
import pandas as pd

from src.data.git_commit_client import GitCommitCollector

logger = logging.getLogger(__name__)


class TemporalGitDataCollector:
    """
    Collects Git metrics from a specific time window for temporal analysis.
    
    Example use case:
    - Historical SonarQube scan: 100 days ago (July 31, 2024)
    - Current SonarQube scan: Today (November 8, 2024)
    - Collect Git features: From July 31 to November 8 (100 days of activity)
    - Label: quality_degradation = score_today - score_100_days_ago
    - Model learns: "These Git patterns led to quality degradation"
    """
    
    def __init__(self, repo_path: str, branch: str = "main"):
        """
        Initialize temporal Git data collector
        
        Args:
            repo_path: Path to Git repository
            branch: Branch to analyze
        """
        self.repo_path = repo_path
        self.branch = branch
        self.client = GitCommitCollector(repo_path, branch)
    
    def get_temporal_commits(
        self, 
        start_date: datetime, 
        end_date: datetime,
        max_commits: int = 10000
    ) -> List[Dict]:
        """
        Get commits within a specific date range with file-level changes.
        
        Args:
            start_date: Beginning of time window (e.g., 100 days ago)
            end_date: End of time window (e.g., today)
            max_commits: Maximum number of commits to fetch
            
        Returns:
            List of commit dictionaries with metadata and file changes
        """
        logger.info(f"Fetching commits from {start_date.date()} to {end_date.date()}")
        logger.info(f"Repository: {self.repo_path}, Branch: {self.branch}")
        
        repo = Repo(self.repo_path)
        branch_ref = repo.heads[self.branch]
        
        # Get commits in date range
        commits = list(repo.iter_commits(
            branch_ref,
            max_count=max_commits,
            since=start_date.strftime('%Y-%m-%d'),
            until=end_date.strftime('%Y-%m-%d')
        ))
        
        logger.info(f"Found {len(commits)} commits in date range")
        
        commit_data = []
        for commit in commits:
            try:
                # Get file-level changes for this commit
                file_changes = {}
                
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
                        
                        # Count insertions and deletions
                        insertions = 0
                        deletions = 0
                        
                        if diff.diff:
                            diff_text = diff.diff.decode('utf-8', errors='ignore')
                            for line in diff_text.split('\n'):
                                if line.startswith('+') and not line.startswith('+++'):
                                    insertions += 1
                                elif line.startswith('-') and not line.startswith('---'):
                                    deletions += 1
                        
                        file_changes[filepath] = {
                            'insertions': insertions,
                            'deletions': deletions
                        }
                
                commit_data.append({
                    'sha': commit.hexsha,
                    'author': commit.author.name,
                    'email': commit.author.email,
                    'date': datetime.fromtimestamp(commit.committed_date),
                    'message': commit.message.strip(),
                    'file_changes': file_changes
                })
            except Exception as e:
                logger.warning(f"Error processing commit {commit.hexsha}: {e}")
                continue
        
        return commit_data
    
    def calculate_temporal_features(
        self,
        start_date: datetime,
        end_date: datetime,
        max_commits: int = 10000
    ) -> pd.DataFrame:
        """
        Calculate Git features for the time window.
        
        Args:
            start_date: Beginning of time window
            end_date: End of time window  
            max_commits: Maximum commits to analyze
            
        Returns:
            DataFrame with Git features per file for the time window
        """
        logger.info(f"Calculating temporal features from {start_date.date()} to {end_date.date()}")
        
        # Calculate days in window
        days_delta = (end_date - start_date).days
        
        # Get commits from the temporal window
        temporal_commits = self.get_temporal_commits(start_date, end_date, max_commits)
        
        if not temporal_commits:
            logger.warning("No commits found in specified date range")
            return pd.DataFrame()
        
        # Extract features from temporal commits
        features_df = self._extract_features_from_commits(temporal_commits, days_delta)
        
        return features_df
    
    def _extract_features_from_commits(
        self, 
        commits: List[Dict], 
        days_in_window: int
    ) -> pd.DataFrame:
        """
        Extract Git features from filtered commits.
        
        Reuses logic from GitCommitClient but adapted for temporal window.
        
        Args:
            commits: List of commit dictionaries
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
            message = commit.get('message', '').lower()
            is_bug = any(keyword in message for keyword in ['fix', 'bug', 'issue', 'error'])
            is_refactor = any(keyword in message for keyword in ['refactor', 'cleanup', 'improve'])
            is_feature = any(keyword in message for keyword in ['feat', 'add', 'implement', 'new'])
            
            for file_path, stats in commit.get('file_changes', {}).items():
                if not file_path or file_path == '/dev/null':
                    continue
                
                file_stats[file_path]['commits'] += 1
                file_stats[file_path]['authors'].add(commit.get('author', 'unknown'))
                file_stats[file_path]['lines_added'] += stats.get('insertions', 0)
                file_stats[file_path]['lines_deleted'] += stats.get('deletions', 0)
                file_stats[file_path]['churn'] += stats.get('insertions', 0) + stats.get('deletions', 0)
                
                if is_bug:
                    file_stats[file_path]['bug_related'] += 1
                if is_refactor:
                    file_stats[file_path]['refactor_related'] += 1
                if is_feature:
                    file_stats[file_path]['feature_related'] += 1
        
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
    
    @staticmethod
    def get_historical_date(days_ago: int, reference_date: Optional[datetime] = None) -> datetime:
        """
        Calculate historical date for temporal analysis.
        
        Args:
            days_ago: Number of days to go back
            reference_date: Reference date (defaults to today)
            
        Returns:
            Historical datetime
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        return reference_date - timedelta(days=days_ago)
