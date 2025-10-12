from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.save_incremental_labeled_data import push_to_mongo
from src.data.github_client import GitHubDataCollector
from src.data.feature_engineering import FeatureEngineer
from src.models.predict import RiskPredictor
from src.llm.ticket_generator import JiraTicketGenerator
from src.utils.config import Config

import os
import logging
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MONGO_URI = "mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin"


# ---------------------------------------------------------------------
# Task Functions
# ---------------------------------------------------------------------

def fetch_latest_github_data(**context):
    """Fetch latest PRs using GitHub search API (merged in last N days)."""
    days_back = 30
    logger.info(f"Fetching PRs merged in the last {days_back} days...")
    collector = GitHubDataCollector()
    raw_df = collector.fetch_pr_data_for_repo(repo_name="mmenalla/readlike-me", since_days=15)

    if raw_df.empty:
        logger.warning("No PRs fetched from GitHub.")
        return

    os.makedirs(Config.DATA_DIR, exist_ok=True)
    raw_path = Config.DATA_DIR / "inference_raw_data.parquet"
    raw_df.to_parquet(raw_path, index=False)

    context['ti'].xcom_push(key='inference_raw_path', value=str(raw_path))
    logger.info(f"Fetched {len(raw_df)} PRs and saved to {raw_path}")


def generate_features(**context):
    """Transform raw GitHub PR data into model-ready features."""
    raw_path = context['ti'].xcom_pull(key='inference_raw_path')
    if not raw_path or not os.path.exists(raw_path):
        raise FileNotFoundError("Missing raw data file for feature generation.")

    df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(df)} raw PR records for feature engineering.")

    fe = FeatureEngineer()
    feature_df = fe.transform(df)

    feature_path = Config.DATA_DIR / "inference_features.parquet"
    feature_df.to_parquet(feature_path, index=False)

    context['ti'].xcom_push(key='inference_feature_path', value=str(feature_path))
    logger.info(f"Generated features for {len(feature_df)} records. Saved to {feature_path}")


def predict_risk(**context):
    """Use the latest trained model to predict PR risk scores."""
    feature_path = context['ti'].xcom_pull(key='inference_feature_path')
    feature_df = pd.read_parquet(feature_path)
    logger.info(f"Running inference on {len(feature_df)} records...")

    model_dir = Path(Config.MODELS_DIR)
    models = sorted(model_dir.glob("xgboost_risk_model_v*.pkl"))
    if not models:
        raise FileNotFoundError("No trained model versions found in models directory!")

    latest_model_path = max(models, key=lambda p: int(p.stem.split("_v")[-1]))
    logger.info(f"Using latest model: {latest_model_path.name}")

    predictor = RiskPredictor(model_path=str(latest_model_path))
    predictions_df = predictor.predict(feature_df)

    # Log distribution of predicted risk scores
    logger.info(
        f"Predicted risk scores — mean: {predictions_df['risk_score'].mean():.3f}, "
        f"std: {predictions_df['risk_score'].std():.3f}"
    )

    predictions_path = Config.DATA_DIR / "inference_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    context['ti'].xcom_push(key='predictions_path', value=str(predictions_path))

    logger.info(f"Predictions saved to {predictions_path}")


def generate_jira_tickets(**context):
    """Generate AI-based Jira tickets for high-risk modules."""
    predictions_path = context['ti'].xcom_pull(key='predictions_path')
    if not predictions_path or not os.path.exists(predictions_path):
        raise FileNotFoundError("Missing predictions file for Jira ticket generation.")

    predictions_df = pd.read_parquet(predictions_path)
    high_risk_df = predictions_df[predictions_df['risk_score'] >= 0.5]
    logger.info(f"Identified {len(high_risk_df)} high-risk PRs for potential tickets.")

    if high_risk_df.empty:
        logger.info("No high-risk PRs detected. Skipping Jira ticket creation.")
        return

    ticket_generator = JiraTicketGenerator()
    tickets = []

    for _, row in high_risk_df.iterrows():
        context_data = {
            "recent_churn": row.get("lines_changed", 0),
            "bug_ratio": row.get("bug_ratio", 0),
            "recent_prs": row.get("pr_count", 0),
            "code_snippet": GitHubDataCollector().get_code_snippet_from_github(
                row["module"], ref=row.get("base_ref", "main")
            ),
        }

        tickets.append({
            "module": row["module"],
            "risk_score": float(row["risk_score"]),
            "context": context_data,
        })

    logger.info(f"Generating LLM-based Jira ticket drafts for {len(tickets)} high-risk modules...")
    ticket_drafts = ticket_generator.generate_tickets_bulk(tickets, num_of_tickets=None)
    ticket_drafts = [{**t, "is_deleted": False} for t in ticket_drafts]

    # Store ticket drafts to Mongo for review / downstream processing
    push_to_mongo(
        pd.DataFrame(ticket_drafts),
        mongo_uri=MONGO_URI,
        mongo_db="risk_model_db",
        mongo_collection="ticket_drafts",
    )

    logger.info(f"Stored {len(ticket_drafts)} ticket drafts to MongoDB (ticket_drafts collection).")


# ---------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------

default_args = {
    "owner": "megi",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 0,
}

with DAG(
    dag_id="risk_model_inference_dag",
    description="Weekly inference using the latest risk model to generate Jira ticket drafts for risky PRs.",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["risk", "inference", "github", "jira"],
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_github_data",
        python_callable=fetch_latest_github_data,
        provide_context=True,
    )

    feature_task = PythonOperator(
        task_id="generate_features",
        python_callable=generate_features,
        provide_context=True,
    )

    predict_task = PythonOperator(
        task_id="predict_risk",
        python_callable=predict_risk,
        provide_context=True,
    )

    jira_task = PythonOperator(
        task_id="generate_jira_tickets",
        python_callable=generate_jira_tickets,
        provide_context=True,
    )

    fetch_task >> feature_task >> predict_task >> jira_task
