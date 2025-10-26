import base64
import os

from github import Github, Auth
from collections import defaultdict
import datetime
import pandas as pd
import logging
import requests

from src.utils.config import Config

logger = logging.getLogger(__name__)


class GitHubDataCollector:
    # Bug-related labels to check for
    BUG_LABELS = {'bug', 'bugfix', 'hotfix', 'regression', 'critical-bug', 'defect', 'fix'}
    
    # Enhanced keyword detection with confidence levels
    BUG_KEYWORDS = {
        'high_confidence': ['crash', 'exception', 'error', 'regression', 'broken', 'null pointer', 'memory leak'],
        'medium_confidence': ['bug', 'bugfix', 'hotfix', 'defect'],
        'low_confidence': ['fix', 'issue', 'problem']
    }
    
    def __init__(self, token: str = None):
        self.token = token or Config.GITHUB_TOKEN
        if not self.token:
            raise ValueError("GitHub token must be provided")
        self.client = Github(auth=Auth.Token(self.token))
        self.headers = {"Authorization": f"token {self.token}"} if self.token else {}

    @staticmethod
    def module_from_path(path: str) -> str:
        """
        Convert file path to module identifier.
        Returns the full path to properly identify files in nested directories.
        """
        # Return the full path - no truncation
        # This ensures files like 'dags/src/models/train.py' are properly identified
        return path
    
    def is_bug_fix_pr(self, pr) -> bool:
        """
        Determine if a PR is a bug fix using multiple signals:
        1. PR labels (highest confidence)
        2. Enhanced keyword detection with confidence levels
        3. Issue references (if available)
        
        Returns True if PR is classified as a bug fix.
        """
        # Method 1: Check PR labels (most reliable)
        try:
            pr_labels = {label.name.lower() for label in pr.labels}
            if self.BUG_LABELS & pr_labels:
                logger.debug(f"PR #{pr.number} identified as bug fix via labels: {pr_labels}")
                return True
        except Exception as e:
            logger.debug(f"Could not check labels for PR #{pr.number}: {e}")
        
        # Method 2: Enhanced keyword detection with confidence levels
        title_body = (pr.title or '').lower() + ' ' + (pr.body or '').lower()
        
        # High confidence keywords - immediate match
        if any(kw in title_body for kw in self.BUG_KEYWORDS['high_confidence']):
            logger.debug(f"PR #{pr.number} identified as bug fix via high-confidence keywords")
            return True
        
        # Medium confidence keywords - immediate match
        if any(kw in title_body for kw in self.BUG_KEYWORDS['medium_confidence']):
            logger.debug(f"PR #{pr.number} identified as bug fix via medium-confidence keywords")
            return True
        
        # Low confidence keywords - require additional signals
        if any(kw in title_body for kw in self.BUG_KEYWORDS['low_confidence']):
            # Additional signals that boost confidence
            signals = []
            
            # Signal: References an issue (closes #123, fixes #456, resolves #789)
            if any(pattern in title_body for pattern in ['closes #', 'fixes #', 'resolves #', 'close #', 'fix #', 'resolve #']):
                signals.append('issue_reference')
            
            # Signal: Small, focused change (likely a targeted fix) - but not documentation-only
            try:
                if pr.changed_files <= 3:
                    # Exclude docs-only changes
                    if not any(doc_word in title_body for doc_word in ['doc', 'readme', 'typo', 'comment']):
                        signals.append('small_change')
            except:
                pass
            
            # Signal: Contains stack trace or error patterns
            if any(pattern in title_body for pattern in ['traceback', 'stack trace', 'error:', 'exception:']):
                signals.append('error_pattern')
            
            # If we have at least one additional signal, classify as bug fix
            if signals:
                logger.debug(f"PR #{pr.number} identified as bug fix via low-confidence keyword + signals: {signals}")
                return True
        
        return False
    
    def fetch_pr_data_for_repo(self, repo_name: str, since_days: int = 365, max_prs: int = 100) -> pd.DataFrame:
        """
        Fetch PRs for a single repo using the GitHub search API, filtering only merged PRs in the last `since_days`.
        """
        if not repo_name:
            raise ValueError("repo_name must be provided")

        since = (datetime.datetime.utcnow() - datetime.timedelta(days=since_days)).date()
        query = f'repo:{repo_name} is:pr is:merged merged:>{since.isoformat()}'
        logger.info(f"Fetching PRs for {repo_name} since {since.isoformat()} using search API")

        prs = list(self.client.search_issues(query=query))
        logger.info(f"Total PRs found by search: {len(prs)}")

        # Limit to max_prs
        if len(prs) > max_prs:
            import random
            prs = random.sample(prs, max_prs)
            logger.info(f"Sampling {max_prs} PRs to process")

        module_stats = defaultdict(lambda: {
            "lines_added": 0, "lines_removed": 0, "prs": 0,
            "unique_authors": set(), "bug_prs": 0, "churn": 0, "created_at": None,
            "repo_name": repo_name, "first_seen": None, "last_modified": None,
            "days_tracked": since_days
        })

        for pr_issue in prs:
            try:
                pr = self.client.get_repo(repo_name).get_pull(pr_issue.number)
            except Exception as e:
                logger.warning(f"Failed to fetch PR {pr_issue.number}: {e}")
                continue

            touched_modules = set()
            for f in pr.get_files():
                module = self.module_from_path(f.filename)
                touched_modules.add(module)
                module_stats[module]['filename'] = os.path.basename(f.filename)
                module_stats[module]['lines_added'] += f.additions
                module_stats[module]['lines_removed'] += f.deletions
                module_stats[module]['churn'] += f.additions + f.deletions
                module_stats[module]['created_at'] = pr.merged_at
                
                # Track first and last modification timestamps for temporal features
                if module_stats[module]['first_seen'] is None or pr.merged_at < module_stats[module]['first_seen']:
                    module_stats[module]['first_seen'] = pr.merged_at
                if module_stats[module]['last_modified'] is None or pr.merged_at > module_stats[module]['last_modified']:
                    module_stats[module]['last_modified'] = pr.merged_at

            for module in touched_modules:
                module_stats[module]['prs'] += 1
                module_stats[module]['unique_authors'].add(pr.user.login)

            # Use enhanced bug detection
            if self.is_bug_fix_pr(pr):
                for module in touched_modules:
                    module_stats[module]['bug_prs'] += 1

        rows = []
        for module, stats in module_stats.items():
            # Calculate temporal features
            file_age_days = None
            last_modified_days = None
            
            if stats['first_seen'] and stats['last_modified']:
                now = datetime.datetime.utcnow()
                # Use timezone-aware comparison if merged_at has timezone
                if stats['first_seen'].tzinfo:
                    now = datetime.datetime.now(datetime.timezone.utc)
                
                file_age_days = (now - stats['first_seen']).days
                last_modified_days = (now - stats['last_modified']).days
            
            rows.append({
                "module": module,
                "filename": module_stats[module]['filename'],
                "lines_added": stats["lines_added"],
                "lines_removed": stats["lines_removed"],
                "prs": stats["prs"],
                "unique_authors": len(stats["unique_authors"]),
                "bug_prs": stats["bug_prs"],
                "churn": stats["churn"],
                "created_at": stats["created_at"],
                "repo_name": stats["repo_name"],
                "days_tracked": stats["days_tracked"],
                "file_age_days": file_age_days,
                "last_modified_days": last_modified_days
            })

        return pd.DataFrame(rows)

    def get_code_snippet_from_github(self, repo: str, module_path: str, ref: str = "main") -> str:
        """
        Fetches the code of a file/module from GitHub.
        """
        import base64
        import logging
        import requests

        url = f"https://api.github.com/repos/{repo}/contents/{module_path}?ref={ref}"

        logging.info(f"Fetching GitHub file: {url}")
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            content_json = response.json()

            # Case 1: GitHub returned a directory listing (list of files)
            if isinstance(content_json, list):
                logging.warning(f"'{module_path}' is a directory, not a file. Skipping.")
                return ""

            # Case 2: GitHub returned a single file
            if content_json.get("encoding") == "base64":
                file_content = base64.b64decode(content_json["content"]).decode("utf-8")
                return file_content

            logging.warning(f"Unexpected encoding or format for '{module_path}', returning empty string.")
            return ""

        elif response.status_code == 404:
            logging.warning(f"File not found: {module_path} at ref {ref}")
            return ""

        else:
            logging.error(f"Failed to fetch file: {response.status_code} - {response.text}")
            return ""
