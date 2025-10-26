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
        """
        Prepare fine-tuning dataset by joining model predictions with manager feedback.
        
        Strategy:
        1. Load ALL predictions from model_predictions collection
        2. LEFT JOIN with risk_feedback on: model_predictions._id = risk_feedback.prediction_id
        3. Where feedback exists: use manager_risk as ground truth
        4. Where no feedback: use original predicted_risk as ground truth
        5. Train on the complete dataset (all predictions)
        """
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        # 1. Load model predictions
        predictions = list(db[PREDICTION_COLLECTION].find({}))
        if not predictions:
            raise ValueError("No predictions found in model_predictions collection!")
        
        pred_df = pd.DataFrame(predictions)
        pred_df['prediction_id'] = pred_df['_id'].astype(str)  # Convert ObjectId to string for join
        pred_df = pred_df.drop(columns=['_id'], errors='ignore')
        
        # Expand nested 'features' dictionary into separate columns
        if 'features' in pred_df.columns:
            features_df = pd.json_normalize(pred_df['features'])
            pred_df = pd.concat([pred_df.drop(columns=['features']), features_df], axis=1)
            logging.info(f"Expanded features into {len(features_df.columns)} columns")
        
        logging.info(f"Loaded {len(pred_df)} predictions from {PREDICTION_COLLECTION}")

        # 2. Load manager feedback (if any)
        feedback = list(db[FEEDBACK_COLLECTION].find({}))
        
        if feedback:
            feedback_df = pd.DataFrame(feedback)
            feedback_df = feedback_df.drop(columns=['_id'], errors='ignore')
            
            # Get latest feedback for each prediction_id
            feedback_df = feedback_df.sort_values('created_at', ascending=False)
            feedback_df = feedback_df.drop_duplicates(subset=['prediction_id'], keep='first')
            
            logging.info(f"Found manager feedback for {len(feedback_df)} predictions")
        else:
            # No feedback - create empty dataframe with correct schema
            logging.warning("No manager feedback found! Using all predictions with original risk scores")
            feedback_df = pd.DataFrame(columns=['prediction_id', 'manager_risk', 'created_at'])

        # 3. LEFT JOIN: keep ALL predictions, add feedback where available
        # Join on: model_predictions._id (as string) = risk_feedback.prediction_id
        train_df = pred_df.merge(
            feedback_df[['prediction_id', 'manager_risk', 'created_at']], 
            on='prediction_id', 
            how='left',  # LEFT JOIN - keep all predictions!
            suffixes=('_pred', '_feedback')
        )
        
        logging.info(f"✅ Total predictions: {len(train_df)}")
        
        # 4. Use manager_risk where available, otherwise use predicted_risk
        # Determine which column has the predicted risk
        pred_col = 'risk_score' if 'risk_score' in train_df.columns else 'predicted_risk'
        
        if pred_col not in train_df.columns:
            raise ValueError(f"Neither 'risk_score' nor 'predicted_risk' found in columns: {train_df.columns.tolist()}")
        
        # Create label: manager_risk if available, else predicted_risk
        train_df['needs_maintenance'] = train_df['manager_risk'].fillna(train_df[pred_col])
        
        # Track label source for statistics
        train_df['label_source'] = train_df['manager_risk'].notna().map({True: 'manager', False: 'prediction'})
        
        # Log statistics
        manager_count = (train_df['label_source'] == 'manager').sum()
        prediction_count = (train_df['label_source'] == 'prediction').sum()
        
        logging.info(f"📊 Training labels breakdown:")
        logging.info(f"   - {manager_count} from manager feedback (corrections)")
        logging.info(f"   - {prediction_count} from original predictions (no feedback)")
        logging.info(f"   - Total: {len(train_df)} samples")
        
        # Calculate correction magnitude for records that have feedback
        if manager_count > 0:
            corrected_records = train_df[train_df['label_source'] == 'manager'].copy()
            avg_correction = (corrected_records['manager_risk'] - corrected_records[pred_col]).abs().mean()
            logging.info(f"📈 Average correction magnitude (for corrected records): {avg_correction:.3f}")
        
        # 5. Prepare for training (drop metadata columns)
        drop_cols = ['module', 'repo_name', 'created_at_pred', 'created_at_feedback', 
                     'filename', 'prediction_id', 'manager_risk', 'risk_score', 'predicted_risk',
                     'risk_category', 'user_id', 'model_version', 'model_name', 'source', 'label_source']
        train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors='ignore')
        
        logging.info(f"Columns after dropping metadata: {train_df.columns.tolist()}")
        
        # Verify we have numeric feature columns
        feature_cols = [c for c in train_df.columns if c != 'needs_maintenance']
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found after dropping metadata!")
        
        non_numeric = train_df[feature_cols].select_dtypes(exclude=['int64', 'float64', 'bool']).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns found: {non_numeric}. All features must be numeric!")
        
        # Ensure needs_maintenance is last column
        if 'needs_maintenance' in train_df.columns:
            cols = [c for c in train_df.columns if c != 'needs_maintenance'] + ['needs_maintenance']
            train_df = train_df[cols]

        dataset_path = Config.DATA_DIR / "finetune_dataset.parquet"
        train_df.to_parquet(dataset_path, index=False)
        context['ti'].xcom_push(key='finetune_dataset_path', value=str(dataset_path))
        logging.info(f"Fine-tune dataset prepared: {train_df.shape[0]} rows, {train_df.shape[1]} features")
        logging.info(f"Saved to: {dataset_path}")

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
