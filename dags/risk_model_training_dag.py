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
from src.data.labels import LabelCreator, create_labels_with_sonarqube
from src.models.train import RiskModelTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MONGO_URI = "mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin"

# Configuration: Set to False to skip data collection and use existing labeled data
FETCH_NEW_DATA = os.getenv("FETCH_NEW_DATA", "True").lower() in ("true", "1", "yes")

# Configuration: Filter training data by label source
# Options: "all", "sonarqube", "heuristic", "manager", "sonarqube+heuristic"
LABEL_SOURCE_FILTER = os.getenv("LABEL_SOURCE_FILTER", "all").lower()


def fetch_repo_pr_data(repo_name: str, **context):
    collector = GitHubDataCollector()
    logger.info(f"REPO: {repo_name}")
    df = collector.fetch_pr_data_for_repo(repo_name, since_days=Config.SINCE_DAYS, max_prs=Config.MAX_PRS)

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
    
    # USE SONARQUBE LABELING STRATEGY - STRICT MODE
    # Only uses SonarQube metrics (no heuristic fallback)
    # ⚠️  REQUIREMENT: All repos must be analyzed in SonarQube first!
    labeled_df = create_labels_with_sonarqube(
        df,
        sonarqube_url=Config.SONARQUBE_URL,
        sonarqube_token=Config.SONARQUBE_TOKEN,
        mongo_uri=MONGO_URI,
        mongo_db="risk_model_db",
        feedback_collection="risk_feedback",
        use_manager_feedback=False,  # Disable for initial training
        fallback_to_heuristic=False  # STRICT: SonarQube only (fails if not analyzed)
    )

    labeled_path = Config.DATA_DIR / "labeled_data.parquet"
    labeled_df.to_parquet(labeled_path, index=False)
    context['ti'].xcom_push(key='labeled_data_path', value=str(labeled_path))

    push_to_mongo(
        labeled_df,
        mongo_uri=MONGO_URI,
        mongo_db="risk_model_db",
        mongo_collection="labeled_pr_data"
    )
    
    # Log label source distribution
    label_counts = labeled_df['label_source'].value_counts().to_dict()
    logger.info(f"Labeled data saved and pushed to MongoDB, {len(labeled_df)} rows")
    logger.info(f"Label sources: {label_counts}")




def train_model(**context):
    if FETCH_NEW_DATA:
        # Use data from the pipeline
        labeled_path = context['ti'].xcom_pull(key='labeled_data_path')
        if not labeled_path:
            logger.info("No labeled data, skipping training")
            return
        df = pd.read_parquet(labeled_path)
    else:
        # Load existing labeled data from MongoDB
        logger.info("FETCH_NEW_DATA=False: Loading existing labeled data from MongoDB")
        from pymongo import MongoClient
        
        try:
            client = MongoClient(MONGO_URI)
            db = client["risk_model_db"]
            collection = db["labeled_pr_data"]
            
            # Fetch all labeled data
            data = list(collection.find({}))
            if not data:
                logger.error("No labeled data found in MongoDB. Set FETCH_NEW_DATA=True to collect new data.")
                return
            
            df = pd.DataFrame(data)
            df = df.drop(columns=['_id'], errors='ignore')
            logger.info(f"Loaded {len(df)} samples from MongoDB")
            client.close()
        except Exception as e:
            logger.error(f"Error loading data from MongoDB: {e}")
            return
    
    # Filter by label source
    total_samples = len(df)
    if LABEL_SOURCE_FILTER != "all":
        if "+" in LABEL_SOURCE_FILTER:
            # Multiple sources: e.g., "sonarqube+heuristic"
            allowed_sources = [s.strip() for s in LABEL_SOURCE_FILTER.split("+")]
            df = df[df['label_source'].isin(allowed_sources)]
            logger.info(f"Filtered to label sources: {allowed_sources}")
        else:
            # Single source: e.g., "sonarqube", "heuristic", "manager"
            df = df[df['label_source'] == LABEL_SOURCE_FILTER]
            logger.info(f"Filtered to label source: {LABEL_SOURCE_FILTER}")
        
        filtered_samples = len(df)
        logger.info(f"Training data filtered: {total_samples} → {filtered_samples} samples ({(filtered_samples/total_samples*100):.1f}%)")
        
        if filtered_samples == 0:
            logger.error(f"No samples found with label_source='{LABEL_SOURCE_FILTER}'. Available sources in data:")
            logger.error(f"{df['label_source'].value_counts().to_dict() if 'label_source' in df.columns else 'N/A'}")
            return
        
        # Log distribution by risk category
        if 'risk_category' in df.columns:
            risk_dist = df['risk_category'].value_counts().to_dict()
            logger.info(f"📊 Risk category distribution: {risk_dist}")
    else:
        logger.info(f"✅ Using ALL label sources for robust training")
        
        # Log combined distribution
        if 'label_source' in df.columns:
            source_dist = df['label_source'].value_counts().to_dict()
            logger.info(f"📊 Label source distribution: {source_dist}")
        
        if 'risk_category' in df.columns:
            risk_dist = df['risk_category'].value_counts().to_dict()
            logger.info(f"📊 Risk category distribution: {risk_dist}")
            
            # Log per-source breakdown
            for source in df['label_source'].unique():
                source_df = df[df['label_source'] == source]
                source_risk_dist = source_df['risk_category'].value_counts().to_dict()
                logger.info(f"   {source}: {source_risk_dist}")
    
    # Drop metadata and SonarQube columns
    # SonarQube is used ONLY for labeling (needs_maintenance), not as features
    logger.info("🔧 Using GitHub PR features only (SonarQube used for labeling only)...")
    drop_cols = [
        'module',
        'repo_name', 'created_at', 'filename', 'label_source', 
        'risk_category', 'feedback_count', 'last_feedback_at',
        # SonarQube raw metrics (used for labeling, not as features)
        'sonarqube_coverage', 'sonarqube_bugs', 'sonarqube_complexity',
        'sonarqube_code_smells', 'sonarqube_duplicated_lines_density',
        'sonarqube_cognitive_complexity', 'sonarqube_ncloc',
        'sonarqube_vulnerabilities', 'sonarqube_sqale_index',
        # Old derived columns (no longer used)
        'bug_ratio', 'churn_per_pr', 'author_concentration'
    ]
    
    # Store training metadata before dropping columns
    training_samples = len(df)
    
    df.drop(inplace=True, columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    logger.info(f"🎯 Training on {training_samples} samples with {len(df.columns)-1} features")
    logger.info(f"📋 Features: {list(df.columns)}")
    logger.info(f"⚖️  Class weighting: ENABLED (handling imbalanced data)")
    
    trainer = RiskModelTrainer()
    # Enable class weighting for better handling of imbalanced data
    model, model_name, X_test, y_test = trainer.train(df, use_class_weights=True)

    context['ti'].xcom_push(key='X_test', value=X_test.to_dict(orient='list'))
    context['ti'].xcom_push(key='y_test', value=y_test.tolist())
    context['ti'].xcom_push(key='model_name', value=model_name)
    context['ti'].xcom_push(key='training_samples', value=training_samples)
    context['ti'].xcom_push(key='label_source_filter', value=LABEL_SOURCE_FILTER)
    logger.info("Model training completed")


def evaluate_model(**context):
    model_name = context['ti'].xcom_pull(key='model_name')
    if not model_name:
        logger.info("No trained model to evaluate")
        return

    X_test = pd.DataFrame.from_dict(context['ti'].xcom_pull(key='X_test'))
    y_test = pd.Series(context['ti'].xcom_pull(key='y_test'))
    training_samples = context['ti'].xcom_pull(key='training_samples') or 0
    label_source_filter = context['ti'].xcom_pull(key='label_source_filter') or "all"

    trainer = RiskModelTrainer(model_path=model_name)
    trainer.model = trainer.load_model(model_name)
    metrics = trainer.evaluate(
        X_test, 
        y_test, 
        model_name,
        label_source_filter=label_source_filter,
        training_samples=training_samples
    )
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

    if FETCH_NEW_DATA:
        # Full pipeline: fetch data, engineer features, label, train, evaluate
        logger.info("FETCH_NEW_DATA=True: Running full data collection pipeline")
        
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
    
    else:
        # Skip data collection: train directly from MongoDB data
        logger.info("FETCH_NEW_DATA=False: Skipping data collection, using existing MongoDB data")
        
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

        train_task >> evaluate_task
