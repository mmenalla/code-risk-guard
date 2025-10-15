from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os
import pandas as pd
import logging

from src.utils.config import Config
from src.data.github_client import GitHubDataCollector
from src.data.save_incremental_labeled_data import push_to_mongo
from src.data.feature_engineering import FeatureEngineer
from src.data.labels import LabelCreator
from src.models.train import RiskModelTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MONGO_URI = "mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin"


def fetch_repo_pr_data(repo_name: str, **context):
    collector = GitHubDataCollector()
    logger.info(f"REPO: {repo_name}")
    df = collector.fetch_pr_data_for_repo(repo_name, since_days=150, max_prs=200)

    os.makedirs(Config.DATA_DIR, exist_ok=True)
    repo_path = Config.DATA_DIR / f"raw_pr_{repo_name.replace('/', '_')}.parquet"
    df.to_parquet(repo_path, index=False)
    context['ti'].xcom_push(key=f'raw_{repo_name}', value=str(repo_path))
    logger.info(f"Saved PR data for {repo_name}, {len(df)} rows")


def aggregate_raw_data(**context):
    all_paths = []
    for repo in Config.GITHUB_REPOS:
        path = context['ti'].xcom_pull(key=f'raw_{repo}')
        if path:
            all_paths.append(path)

    if not all_paths:
        logger.info("No PR data found. Exiting.")
        return

    df_list = [pd.read_parquet(p) for p in all_paths]
    raw_df = pd.concat(df_list, ignore_index=True)

    raw_path = Config.DATA_DIR / "raw_pr_data.parquet"
    raw_df.to_parquet(raw_path, index=False)
    context['ti'].xcom_push(key='raw_data_path', value=str(raw_path))
    logger.info(f"Aggregated PR data saved, {len(raw_df)} rows from {len(all_paths)} repos")


def feature_engineering(**context):
    raw_path = context['ti'].xcom_pull(key='raw_data_path')
    if not raw_path:
        logger.info("No data to process for feature engineering")
        return

    df = pd.read_parquet(raw_path)
    fe = FeatureEngineer()
    feature_df = fe.transform(df)

    feature_path = Config.DATA_DIR / "features.parquet"
    feature_df.to_parquet(feature_path, index=False)
    context['ti'].xcom_push(key='feature_data_path', value=str(feature_path))
    logger.info(f"Features saved, {feature_df.shape[0]} rows")


def label_data(**context):
    feature_path = context['ti'].xcom_pull(key='feature_data_path')
    if not feature_path:
        logger.info("No data to label")
        return

    df = pd.read_parquet(feature_path)
    labeler = LabelCreator(bug_threshold=0.2, churn_threshold=50)
    labeled_df = labeler.create_labels(df)

    labeled_path = Config.DATA_DIR / "labeled_data.parquet"
    labeled_df.to_parquet(labeled_path, index=False)
    context['ti'].xcom_push(key='labeled_data_path', value=str(labeled_path))

    push_to_mongo(
        labeled_df,
        mongo_uri=MONGO_URI,
        mongo_db="risk_model_db",
        mongo_collection="labeled_pr_data"
    )
    logger.info(f"Labeled data saved and pushed to MongoDB, {len(labeled_df)} rows")


def train_model(**context):
    labeled_path = context['ti'].xcom_pull(key='labeled_data_path')
    if not labeled_path:
        logger.info("No labeled data, skipping training")
        return

    df = pd.read_parquet(labeled_path)
    df.drop(inplace=True, columns=['repo_name', 'created_at'], errors='ignore')
    trainer = RiskModelTrainer()
    model, model_name, X_test, y_test = trainer.train(df)

    context['ti'].xcom_push(key='X_test', value=X_test.to_dict(orient='list'))
    context['ti'].xcom_push(key='y_test', value=y_test.tolist())
    context['ti'].xcom_push(key='model_name', value=model_name)
    logger.info("Model training completed")


def evaluate_model(**context):
    model_name = context['ti'].xcom_pull(key='model_name')
    if not model_name:
        logger.info("No trained model to evaluate")
        return

    X_test = pd.DataFrame.from_dict(context['ti'].xcom_pull(key='X_test'))
    y_test = pd.Series(context['ti'].xcom_pull(key='y_test'))

    trainer = RiskModelTrainer(model_path=model_name)
    trainer.model = trainer.load_model(model_name)
    metrics = trainer.evaluate(X_test, y_test, model_name)
    logger.info(f"Evaluation metrics: {metrics}")


default_args = {
    "owner": "megi",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 0,
}

with DAG(
    dag_id="risk_model_training_dag",
    description="Train XGBoost risk model from multiple GitHub repos",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    # --- TaskGroup: fetch all repos ---
    with TaskGroup("github_pr_fetch") as fetch_group:
        for repo in Config.GITHUB_REPOS:
            PythonOperator(
                task_id=f"fetch_{repo.replace('/', '_')}",
                python_callable=fetch_repo_pr_data,
                op_kwargs={"repo_name": repo},
                provide_context=True
            )

    aggregate_task = PythonOperator(
        task_id="aggregate_raw_data",
        python_callable=aggregate_raw_data,
        provide_context=True
    )

    feature_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
        provide_context=True
    )

    label_task = PythonOperator(
        task_id="label_data",
        python_callable=label_data,
        provide_context=True
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        provide_context=True
    )

    fetch_group >> aggregate_task >> feature_task >> label_task >> train_task >> evaluate_task
