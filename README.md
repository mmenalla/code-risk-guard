# Code Risk Guard

A Python-based platform for automated risk assessment of code changes, leveraging machine learning and LLMs to generate actionable Jira tickets for high-risk modules.

## Features

- Automated data collection from GitHub PRs
- Risk model training and inference using XGBoost
- Streamlit dashboard for ticket review and approval
- Integration with Jira for ticket creation
- Dockerized deployment with Airflow, MongoDB, and PostgreSQL

## Requirements

- Python 3.10+
- Docker & Docker Compose
- pip

## Setup

**Clone the repository:**
   ```sh
   git clone https://github.com/mmenalla/code-risk-guard.git
   cd code-risk-guard
   ```

** Follow the Makefile instructions to set up the environment:**
   ```sh
   make build
   make run
   ```

Airflow UI:
Visit http://localhost:8080 after starting containers.

SonnarQube Labeling:
```
cd /Users/megi/Documents/Other/TechDebtPOC/TDGPTRepos && ./scan_all_repos.sh
```
