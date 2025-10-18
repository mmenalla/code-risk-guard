import os
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models" / "artifacts"


    GITHUB_REPO = os.getenv("GITHUB_REPO", 'mmenalla/readlike-me').strip()
    GITHUB_REPOS = os.getenv("GITHUB_REPOS", '["scikit-learn/scikit-learn"]').strip()
    try:
        GITHUB_REPOS = json.loads(GITHUB_REPOS)
    except json.JSONDecodeError:
        GITHUB_REPOS = [repo.strip() for repo in GITHUB_REPOS.split(",")]

    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    # Jira
    JIRA_USERNAME = None
    JIRA_SERVER = None
    JIRA_URL = os.getenv("JIRA_URL")
    JIRA_USER = os.getenv("JIRA_USER")
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
    JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "TECH")

    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "xgboost_risk_model.json")

    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
