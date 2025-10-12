import os
from atlassian import Jira
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class JiraClient:
    def __init__(self):
        self.base_url = os.getenv("JIRA_BASE_URL")
        self.email = os.getenv("JIRA_USER_EMAIL")
        self.api_token = os.getenv("JIRA_API_TOKEN")

        if not all([self.base_url, self.email, self.api_token]):
            raise ValueError("JIRA_BASE_URL, JIRA_USER_EMAIL, and JIRA_API_TOKEN must be set in .env")

        self.jira = Jira(
            url=self.base_url,
            username=self.email,
            password=self.api_token
        )

        self.default_project = os.getenv("JIRA_PROJECT_KEY", "SCRUM")

    def create_ticket(self, title: str, description: str, project_key: str = None) -> Dict:
        """
        Create a single Jira issue.
        """
        project_key = project_key or self.default_project

        issue_fields = {
            "project": {"key": project_key},
            "summary": title.strip("**Title:** ").strip(),
            "description": description.strip("**Description:** ").strip(),
            "issuetype": {"name": "Task"}  # or "Bug", "Maintenance", etc.
        }

        issue = self.jira.issue_create(fields=issue_fields)
        return {"key": issue["key"], "url": f"{self.base_url}/browse/{issue['key']}"}

    def create_tickets_bulk(self, tickets: List[Dict], project_key: str = None) -> List[Dict]:
        """
        Create multiple Jira issues from a list of drafts.
        Each dict in tickets must have 'title' and 'description'.
        """
        created = []
        for t in tickets:
            issue = self.create_ticket(t['title'], t['description'], project_key)
            created.append(issue)
        return created
