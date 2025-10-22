import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.save_incremental_labeled_data import log_model_metrics
from src.models.train import RiskModelTrainer
from src.utils.config import Config

# --- Mongo config ---
MONGO_URI = "mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin"
DB_NAME = "risk_model_db"
PREDICTION_COLLECTION = "model_predictions"
FEEDBACK_COLLECTION = "risk_feedback"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    # "retry_delay": timedelta(minutes=5),
}

# --- DAG ---
with DAG(
    dag_id="risk_model_finetune_dag",
    default_args=default_args,
    description="Fine-tune risk model with human feedback",
    schedule_interval=None,
    start_date=datetime(2025, 10, 18),
    catchup=False,
    tags=["risk_model", "finetune"],
) as dag:

    def prepare_finetune_dataset(**context):
        """Pull predictions and human feedback, join on prediction_id"""
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        # Load predictions
        predictions = list(db[PREDICTION_COLLECTION].find({}))
        if not predictions:
            raise ValueError("No predictions found in MongoDB!")
        pred_df = pd.DataFrame(predictions)
        features_df = pd.json_normalize(pred_df['features'])
        pred_df = pd.concat([pred_df.reset_index(drop=True), features_df], axis=1)
        pred_df.rename(columns={'_id': 'prediction_id'}, inplace=True)
        pred_df['prediction_id'] = pred_df['prediction_id'].astype(str)

        # Load feedback
        feedback = list(db[FEEDBACK_COLLECTION].find({}))
        if not feedback:
            raise ValueError("No human feedback found in MongoDB!")
        feedback_df = pd.DataFrame(feedback)
        feedback_df['prediction_id'] = feedback_df['prediction_id'].astype(str)

        # Mark deleted feedback as 0 risk
        # if 'is_deleted' in feedback_df.columns:
        #     feedback_df.loc[feedback_df['is_deleted'], 'manager_risk'] = 0.0

        # Join
        merged_df = pred_df.merge(
            feedback_df[['prediction_id', 'manager_risk']],
            on='prediction_id',
            how='inner'
        )

        # Prepare training dataset
        drop_cols = ['_id', 'model_name', 'module', 'features', 'repo_name', 'created_at', 'source', 'needs_maintenance', 'prediction_id', 'predicted_risk']
        train_df = merged_df.drop(columns=[c for c in drop_cols if c in merged_df.columns])
        train_df.rename(columns={'manager_risk': 'needs_maintenance'}, inplace=True)
        cols = [c for c in train_df.columns if c != 'needs_maintenance'] + ['needs_maintenance']
        train_df = train_df[cols]

        dataset_path = Config.DATA_DIR / "finetune_dataset.parquet"
        train_df.to_parquet(dataset_path, index=False)
        context['ti'].xcom_push(key='finetune_dataset_path', value=str(dataset_path))
        logging.info(f"Finetune dataset prepared: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

    def finetune_model(**context):
        """Load dataset and fine-tune latest model"""
        dataset_path = context['ti'].xcom_pull(key='finetune_dataset_path')
        if not dataset_path or not Path(dataset_path).exists():
            raise FileNotFoundError("Finetune dataset not found!")

        df = pd.read_parquet(dataset_path)
        trainer = RiskModelTrainer()

        # Load the latest model if available
        latest_model_dir = Path(Config.MODELS_DIR)
        latest_models = sorted(latest_model_dir.glob("xgboost_risk_model_v*.pkl"))
        if latest_models:
            latest_model_path = max(latest_models, key=lambda p: int(p.stem.split("_v")[-1]))
            trainer.model_path = latest_model_path
            trainer.load_existing_model()

        # Train (fine-tune)
        model, model_name, X_test, y_test = trainer.train(df)

        # Evaluate and log metrics
        metrics = trainer.evaluate(X_test, y_test, model_name)
        logging.info(f"Fine-tuning complete. Model saved at {model_name}")
        log_model_metrics(metrics, model_name=model_name.split("/")[-1].rsplit(".", 1)[0])
        context['ti'].xcom_push(key='finetuned_model_path', value=model_name)

    t1 = PythonOperator(
        task_id="prepare_finetune_dataset",
        python_callable=prepare_finetune_dataset,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id="finetune_model",
        python_callable=finetune_model,
        provide_context=True,
    )

    t1 >> t2
