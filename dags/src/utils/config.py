import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load .env file from src directory (dags/src/.env)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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

    # GitHub Data Collection
    SINCE_DAYS = int(os.getenv("SINCE_DAYS", "150"))
    MAX_PRS = int(os.getenv("MAX_PRS", "50"))

    # Ticket Generation
    GENERATE_TICKETS = os.getenv("GENERATE_TICKETS", "True").lower() in ("true", "1", "yes")
    RISK_SCORE_THRESHOLD = float(os.getenv("RISK_SCORE_THRESHOLD", "0.5"))
    NUM_TICKETS_PER_MODULE = int(os.getenv("NUM_TICKETS_PER_MODULE", "1"))

    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # SonarQube
    # Use http://sonarqube:9000 for Docker container-to-container communication
    # Use http://localhost:9000 for local testing outside Docker
    SONARQUBE_URL = os.getenv("SONARQUBE_URL", "http://sonarqube:9000")
    SONARQUBE_TOKEN = os.getenv("SONARQUBE_TOKEN")
    SONARQUBE_PROJECT_KEYS = os.getenv("SONARQUBE_PROJECT_KEYS", "").strip()
    try:
        SONARQUBE_PROJECT_KEYS = json.loads(SONARQUBE_PROJECT_KEYS) if SONARQUBE_PROJECT_KEYS else []
    except json.JSONDecodeError:
        SONARQUBE_PROJECT_KEYS = [key.strip() for key in SONARQUBE_PROJECT_KEYS.split(",") if key.strip()]
