import base64

from github import Github, Auth
from collections import defaultdict
import datetime
import pandas as pd
import logging
import requests

from src.utils.config import Config

logger = logging.getLogger(__name__)


class GitHubDataCollector:
    def __init__(self, token: str = None):
        self.token = token or Config.GITHUB_TOKEN
        if not self.token:
            raise ValueError("GitHub token must be provided")
        self.client = Github(auth=Auth.Token(self.token))
        self.headers = {"Authorization": f"token {self.token}"} if self.token else {}

    @staticmethod
    def module_from_path(path: str) -> str:
        parts = path.split('/')
        return '/'.join(parts[:2]) if len(parts) >= 2 else parts[0]

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
            "repo_name": repo_name
        })

        for pr_issue in prs:
            # Convert issue to PR object
            try:
                pr = self.client.get_repo(repo_name).get_pull(pr_issue.number)
            except Exception as e:
                logger.warning(f"Failed to fetch PR {pr_issue.number}: {e}")
                continue

            touched_modules = set()
            for f in pr.get_files():
                module = self.module_from_path(f.filename)
                touched_modules.add(module)
                module_stats[module]['lines_added'] += f.additions
                module_stats[module]['lines_removed'] += f.deletions
                module_stats[module]['churn'] += f.additions + f.deletions
                module_stats[module]['created_at'] = pr.merged_at

            for module in touched_modules:
                module_stats[module]['prs'] += 1
                module_stats[module]['unique_authors'].add(pr.user.login)

            title_body = (pr.title or '') + ' ' + (pr.body or '')
            if any(w in title_body.lower() for w in ['fix', 'bug', 'bugfix', 'hotfix']):
                for module in touched_modules:
                    module_stats[module]['bug_prs'] += 1

        # Convert to DataFrame
        rows = []
        for module, stats in module_stats.items():
            rows.append({
                "module": module,
                "lines_added": stats["lines_added"],
                "lines_removed": stats["lines_removed"],
                "prs": stats["prs"],
                "unique_authors": len(stats["unique_authors"]),
                "bug_prs": stats["bug_prs"],
                "churn": stats["churn"],
                "created_at": stats["created_at"],
                "repo_name": stats["repo_name"]
            })

        return pd.DataFrame(rows)

    def get_code_snippet_from_github(self, module_path: str, ref: str = "main") -> str:
        """
        Fetches the code of a file/module from GitHub.

        :param module_path: path in repo, e.g., "src/models/train.py"
        :param ref: branch, tag, or commit, e.g., "main"
        :return: string containing the file content
        """
        # Extract owner/repo from module_path if needed
        # Example assumes a fixed repo for simplicity
        owner = "your-github-org-or-user"
        repo = "your-repo-name"

        # Construct GitHub API URL for file contents
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{module_path}?ref={ref}"

        logging.info(f"Fetching GitHub file: {url}")
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            content_json = response.json()
            if content_json.get("encoding") == "base64":
                file_content = base64.b64decode(content_json["content"]).decode("utf-8")
                return file_content
            else:
                logging.warning("Unexpected encoding for file content, returning empty string")
                return ""
        elif response.status_code == 404:
            logging.warning(f"File not found: {module_path} at ref {ref}")
            return ""
        else:
            logging.error(f"Failed to fetch file: {response.status_code} - {response.text}")
            return ""
