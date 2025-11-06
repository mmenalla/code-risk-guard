from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os
import pandas as pd
import logging
from pathlib import Path

from src.utils.config import Config
from src.data.github_client import GitHubDataCollector
from src.data.save_incremental_labeled_data import push_to_mongo
from src.data.feature_engineering import FeatureEngineer
from src.data.labels import LabelCreator, create_labels_with_sonarqube
from src.models.train import RiskModelTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MONGO_URI = "mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin"

# Configuration
FETCH_NEW_DATA = os.getenv("FETCH_NEW_DATA", "True").lower() in ("true", "1", "yes")
LABEL_SOURCE_FILTER = os.getenv("LABEL_SOURCE_FILTER", "sonarqube").lower()


def fetch_commit_data_for_repo(repo_name: str, **context):
    """Fetch commit data from a single local repository"""
    from src.data.git_commit_client import GitCommitCollector
    # Build repo path
    if Config.REPO_BASE_DIR:
        repo_path = os.path.join(Config.REPO_BASE_DIR, repo_name)
    else:
        repo_path = Config.REPO_PATH
    
    # Get branch for this specific repository
    branch = Config.get_branch_for_repo(repo_name)
    
    logger.info(f"🔄 Fetching commits from: {repo_name}")
    logger.info(f"   Path: {repo_path}")
    logger.info(f"   Branch: {branch}, Max commits: {Config.MAX_COMMITS}")
    
    if not Path(repo_path).exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        raise ValueError(f"Repository not found: {repo_path}")
    
    try:
        collector = GitCommitCollector(
            repo_path=repo_path,
            branch=branch  # Use per-repo branch
        )
        raw_df = collector.fetch_commit_data(max_commits=Config.MAX_COMMITS)
        
        if raw_df.empty:
            logger.warning(f"No commit data fetched from {repo_name}")
            return
        
        # Add repo_name for tracking and SonarQube mapping
        raw_df['repo_name'] = repo_name
        
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        raw_path = Config.DATA_DIR / f"raw_commit_{repo_name.replace('/', '_')}.parquet"
        raw_df.to_parquet(raw_path, index=False)
        context['ti'].xcom_push(key=f'raw_commit_{repo_name}', value=str(raw_path))
        
        logger.info(f"✅ Saved commit data for {repo_name}: {len(raw_df)} files")
        logger.info(f"   Total commits: {raw_df['prs'].sum():.0f}")
        
    except Exception as e:
        logger.error(f"Error fetching commits from {repo_name}: {e}")
        raise


def aggregate_commit_data(**context):
    """Aggregate commit data from all repositories"""
    all_paths = []
    
    # Get all repository names
    repo_names = Config.REPO_NAMES if Config.REPO_NAMES else ["single_repo"]
    
    logger.info(f"Aggregating commit data from {len(repo_names)} repositories")
    
    for repo_name in repo_names:
        path = context['ti'].xcom_pull(key=f'raw_commit_{repo_name}')
        if path and Path(path).exists():
            all_paths.append(path)
            logger.info(f"  ✓ Found data for: {repo_name}")
        else:
            logger.warning(f"  ✗ No data for: {repo_name}")
    
    if not all_paths:
        logger.error("No commit data found to aggregate")
        raise ValueError("No commit data available")
    
    df_list = [pd.read_parquet(p) for p in all_paths]
    raw_df = pd.concat(df_list, ignore_index=True)
    
    raw_path = Config.DATA_DIR / "raw_commit_data.parquet"
    raw_df.to_parquet(raw_path, index=False)
    context['ti'].xcom_push(key='raw_data_path', value=str(raw_path))
    
    logger.info(f"✅ Aggregated commit data:")
    logger.info(f"   Total files: {len(raw_df)}")
    logger.info(f"   Total commits: {raw_df['prs'].sum():.0f}")
    logger.info(f"   Repositories: {raw_df['repo_name'].unique().tolist()}")
    
    # Log per-repo statistics
    for repo in raw_df['repo_name'].unique():
        repo_df = raw_df[raw_df['repo_name'] == repo]
        logger.info(f"   - {repo}: {len(repo_df)} files, {repo_df['prs'].sum():.0f} commits")


def feature_engineering(**context):
    """Generate ML features from commit data"""
    raw_path = context['ti'].xcom_pull(key='raw_data_path')
    if not raw_path:
        logger.error("No data to process for feature engineering")
        return

    df = pd.read_parquet(raw_path)
    fe = FeatureEngineer()
    feature_df = fe.transform(df)

    feature_path = Config.DATA_DIR / "features.parquet"
    feature_df.to_parquet(feature_path, index=False)
    context['ti'].xcom_push(key='feature_data_path', value=str(feature_path))
    logger.info(f"✅ Features saved: {feature_df.shape[0]} rows, {feature_df.shape[1]} columns")


def label_data(**context):
    """Label data using SonarQube metrics ONLY (no heuristic fallback)"""
    feature_path = context['ti'].xcom_pull(key='feature_data_path')
    if not feature_path:
        logger.error("No data to label")
        return

    df = pd.read_parquet(feature_path)
    
    logger.info("🏷️  Labeling with SonarQube (STRICT MODE - no fallback)")
    logger.info(f"   SonarQube URL: {Config.SONARQUBE_URL}")
    logger.info(f"   Project Keys: {Config.SONARQUBE_PROJECT_KEYS}")
    
    # Map repo_name to SonarQube project key
    # Assumption: repo_name matches SonarQube project key
    # If different, create a mapping dict here
    
    labeled_df = create_labels_with_sonarqube(
        df,
        sonarqube_url=Config.SONARQUBE_URL,
        sonarqube_token=Config.SONARQUBE_TOKEN,
        mongo_uri=MONGO_URI,
        mongo_db="risk_model_db",
        feedback_collection="risk_feedback",
        use_manager_feedback=False,
        fallback_to_heuristic=False  # STRICT: SonarQube only
    )

    labeled_path = Config.DATA_DIR / "labeled_data.parquet"
    labeled_df.to_parquet(labeled_path, index=False)
    context['ti'].xcom_push(key='labeled_data_path', value=str(labeled_path))

    # Save to MongoDB
    push_to_mongo(
        labeled_df,
        mongo_uri=MONGO_URI,
        mongo_db="risk_model_db",
        mongo_collection="labeled_pr_data"
    )
    
    # Log statistics
    label_counts = labeled_df['label_source'].value_counts().to_dict()
    logger.info(f"✅ Labeled data saved: {len(labeled_df)} rows")
    logger.info(f"   Label sources: {label_counts}")
    
    if 'repo_name' in labeled_df.columns:
        logger.info(f"   Repositories: {labeled_df['repo_name'].unique().tolist()}")
        for repo in labeled_df['repo_name'].unique():
            repo_count = len(labeled_df[labeled_df['repo_name'] == repo])
            logger.info(f"     - {repo}: {repo_count} labeled files")


def train_model(**context):
    """Train XGBoost model on labeled data"""
    if FETCH_NEW_DATA:
        labeled_path = context['ti'].xcom_pull(key='labeled_data_path')
        if not labeled_path:
            logger.error("No labeled data, skipping training")
            return
        df = pd.read_parquet(labeled_path)
    else:
        logger.info("Loading existing labeled data from MongoDB")
        from pymongo import MongoClient
        
        try:
            client = MongoClient(MONGO_URI)
            db = client["risk_model_db"]
            collection = db["labeled_pr_data"]
            
            data = list(collection.find({}))
            if not data:
                logger.error("No labeled data found in MongoDB")
                return
            
            df = pd.DataFrame(data)
            df = df.drop(columns=['_id'], errors='ignore')
            logger.info(f"Loaded {len(df)} samples from MongoDB")
            client.close()
        except Exception as e:
            logger.error(f"Error loading data from MongoDB: {e}")
            return
    
    # Filter by label source (should be "sonarqube" only)
    total_samples = len(df)
    if LABEL_SOURCE_FILTER != "all":
        if "+" in LABEL_SOURCE_FILTER:
            allowed_sources = [s.strip() for s in LABEL_SOURCE_FILTER.split("+")]
            df = df[df['label_source'].isin(allowed_sources)]
        else:
            df = df[df['label_source'] == LABEL_SOURCE_FILTER]
        
        filtered_samples = len(df)
        logger.info(f"Training data filtered: {total_samples} → {filtered_samples} samples")
        
        if filtered_samples == 0:
            logger.error(f"No samples found with label_source='{LABEL_SOURCE_FILTER}'")
            return
    
    # Log distribution
    if 'risk_category' in df.columns:
        risk_dist = df['risk_category'].value_counts().to_dict()
        logger.info(f"📊 Risk distribution: {risk_dist}")
    
    if 'repo_name' in df.columns:
        repo_dist = df['repo_name'].value_counts().to_dict()
        logger.info(f"📊 Repository distribution: {repo_dist}")
    
    # Train model
    logger.info("🚀 Training XGBoost model...")
    model_path = Config.MODELS_DIR / "xgboost_risk_model_multi_repo.pkl"
    trainer = RiskModelTrainer(model_path=str(model_path))
    metrics = trainer.train(df, use_class_weights=False)  # Regression mode
    
    logger.info(f"✅ Model trained successfully!")
    logger.info(f"   MAE: {metrics['mae']:.4f}")
    logger.info(f"   RMSE: {metrics['rmse']:.4f}")
    logger.info(f"   R²: {metrics['r2']:.4f}")


# DAG Definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    # 'retry_delay': timedelta(minutes=5),
}

with DAG(
    'risk_model_training_multi_repo',
    default_args=default_args,
    description='Train risk model on multiple local repositories with SonarQube labels',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['risk-model', 'training', 'multi-repo', 'sonarqube'],
) as dag:

    if FETCH_NEW_DATA and Config.USE_COMMITS:
        # Multi-repo commit collection
        if Config.REPO_NAMES and Config.REPO_BASE_DIR:
            # Dynamically create tasks for each repository
            fetch_tasks = []
            for repo_name in Config.REPO_NAMES:
                task = PythonOperator(
                    task_id=f'fetch_commits_{repo_name.replace("/", "_").replace("-", "_")}',
                    python_callable=fetch_commit_data_for_repo,
                    op_kwargs={'repo_name': repo_name},
                )
                fetch_tasks.append(task)
            
            aggregate_task = PythonOperator(
                task_id='aggregate_commit_data',
                python_callable=aggregate_commit_data,
            )
            
            # Set dependencies: all fetch tasks → aggregate
            for fetch_task in fetch_tasks:
                fetch_task >> aggregate_task
        else:
            raise ValueError("REPO_NAMES and REPO_BASE_DIR must be set for multi-repo training")
        
        feature_task = PythonOperator(
            task_id='feature_engineering',
            python_callable=feature_engineering,
        )
        
        label_task = PythonOperator(
            task_id='label_data',
            python_callable=label_data,
        )
        
        train_task = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
        )
        
        aggregate_task >> feature_task >> label_task >> train_task
    
    else:
        # Train directly from MongoDB (FETCH_NEW_DATA=False)
        train_task = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
        )
