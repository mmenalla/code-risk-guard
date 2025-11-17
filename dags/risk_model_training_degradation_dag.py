"""
Multi-Window Temporal Degradation Model Training DAG

Trains a model to predict code quality degradation using multiple time windows
to create more training samples and capture temporal patterns.

Workflow:
1. Verify multi-window historical SonarQube projects exist
2. Extract Git features and SonarQube metrics for each time window
3. Calculate degradation labels by comparing quality at window boundaries
4. Aggregate labeled data from all repos and windows
5. Train XGBoost regression model
6. Save model and performance metrics
"""

import os
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from src.utils.config import Config
from src.data.git_commit_client import GitCommitCollector
from src.data.sonarqube_client import SonarQubeClient
from src.data.save_incremental_labeled_data import push_to_mongo, log_model_metrics
from src.models.train import RiskModelTrainer

logger = logging.getLogger(__name__)



# Multi-Window Configurations for Task Groups
WINDOW_CONFIGS = {
    'window_2x150d': [0, 150, 300],
    'window_4x50d': [0, 50, 100, 150, 200],
    'window_3x100d': [0, 100, 200, 300]
}
MAX_COMMITS = int(os.getenv("MAX_COMMITS", "10000"))
LABEL_SOURCE_FILTER = "sonarqube_degradation_multi_window"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

def verify_multi_window_projects(**context):
    """Verify that multi-window historical SonarQube projects exist for all repos."""
    logger.info("🔍 Verifying multi-window historical SonarQube projects...")
    
    sonarqube_client = SonarQubeClient(
        Config.SONARQUBE_URL,
        Config.SONARQUBE_TOKEN
    )
    
    all_expected = []
    missing_projects = []
    import sys
    print("Python executable:", sys.executable)
    print("sys.path:", sys.path)
    try:
        import git
        print("GitPython imported successfully!")
    except ImportError as e:
        print("GitPython import failed:", e)
        
    for repo_name, project_key in zip(Config.REPO_NAMES, Config.SONARQUBE_PROJECT_KEYS):
        for group_windows in WINDOW_CONFIGS.values():
            for days_ago in group_windows:
                historical_key = f"{project_key}-{days_ago}d"
                all_expected.append(historical_key)
                if not sonarqube_client.verify_historical_projects_exist([historical_key]):
                    missing_projects.append((repo_name, historical_key, days_ago))
    
    if missing_projects:
        logger.error(f"❌ Missing {len(missing_projects)} historical projects:")
        for repo, key, days in missing_projects[:10]:  # Show first 10
            logger.error(f"   • {repo}: {key} (scan at {days} days ago)")
        
        logger.error("\n💡 Run this to create missing projects:")
        logger.error("   cd /path/to/TDGPTRepos")
        logger.error("   ./scan_multi_window_historical_robust.sh")
        
        raise ValueError(f"Missing {len(missing_projects)} multi-window projects. Run scanner first.")
    
    logger.info(f"✅ All {len(all_expected)} multi-window projects verified")
    total_windows = sum(len(windows) for windows in WINDOW_CONFIGS.values())
    logger.info(f"   {len(Config.REPO_NAMES)} repos × {total_windows} windows (across all groups)")
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='verified_projects', value=all_expected)

def fetch_multi_window_data_task(repo_name: str, window_idx: int, window_size_days: int, **context):
    """Fetch commits and calculate Git features for a specific time window."""
    logger.info(f"🔄 Fetching window {window_idx} data for: {repo_name}")
    
    # Get configuration
    repo_path = f"{Config.REPO_BASE_DIR}/{repo_name}"
    branch = Config.get_branch_for_repo(repo_name)
    logger.info(f"[DEBUG] Repo path: {repo_path}")
    logger.info(f"[DEBUG] Branch: {branch}")
    logger.info(f"[DEBUG] Repo exists: {os.path.exists(repo_path)}")

    reference_date = datetime.now()
    # Get the window boundaries from the config
    # For each window, use windows[window_idx] and windows[window_idx+1] as start and end days ago
    # Use group_name from op_kwargs for explicit window config lookup
    group_name = context.get('group_name')
    if group_name is None:
        logger.error("group_name not provided to fetch_multi_window_data_task")
        return
    windows = WINDOW_CONFIGS.get(group_name)
    if windows is None or window_idx >= len(windows) - 1:
        logger.error(f"No window config found for group {group_name} and index {window_idx}")
        return
    window_start_days = windows[window_idx]
    window_end_days = windows[window_idx + 1]
    window_start = reference_date - timedelta(days=window_end_days)
    window_end = reference_date - timedelta(days=window_start_days)
    logger.info(f"[DEBUG] Window {window_idx} ({group_name}): {window_start.date()} → {window_end.date()}")

    collector = GitCommitCollector(repo_path, branch)
    try:
        features_df = collector.calculate_temporal_features(
            start_date=window_start,
            end_date=window_end,
            max_commits=MAX_COMMITS
        )
        logger.info(f"[DEBUG] Features DataFrame shape: {features_df.shape}")
        if hasattr(collector, 'repo'):
            commits = list(collector.repo.iter_commits(
                branch,
                max_count=MAX_COMMITS,
                since=window_start.strftime('%Y-%m-%d'),
                until=window_end.strftime('%Y-%m-%d')
            ))
            logger.info(f"[DEBUG] Number of commits found: {len(commits)}")
        if features_df.empty:
            logger.warning(f"No features for {repo_name} window {window_idx}")
            context['task_instance'].xcom_push(
                key=f'{repo_name}_window{window_idx}_features',
                value=None
            )
            return
        features_df['repo_name'] = repo_name
        features_df['window_id'] = window_idx
        features_df['window_start'] = str(window_start)
        features_df['window_end'] = str(window_end)
        features_df['window_size_days'] = window_size_days
        logger.info(f"✅ Extracted {len(features_df)} file features from window {window_idx}")
        context['task_instance'].xcom_push(
            key=f'{repo_name}_window{window_idx}_features',
            value=features_df.to_dict('records')
        )
    except Exception as e:
        logger.error(f"❌ Error fetching window {window_idx} for {repo_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def create_multi_window_labels_task(repo_name: str, window_idx: int, window_size_days: int, **context):
    """Create degradation labels by comparing SonarQube metrics at window boundaries."""
    import pandas as pd
    
    logger.info(f"🏷️  Creating labels for {repo_name} window {window_idx}")
    

    # Pull features from XCom using correct fetch task_id
    # Use group_name from op_kwargs for explicit window config lookup
    group_name = context.get('group_name')
    if group_name is None:
        logger.error("group_name not provided to create_multi_window_labels_task")
        return
    windows = WINDOW_CONFIGS.get(group_name)
    if windows is None or window_idx >= len(windows) - 1:
        logger.error(f"No window config found for group {group_name} and index {window_idx}")
        return
    interval_str = f'{windows[window_idx]}_{windows[window_idx + 1]}'
    fetch_task_id = f'fetch_{repo_name}_window{interval_str}d'
    features_data = context['task_instance'].xcom_pull(
        task_ids=f'fetch_{group_name}.{fetch_task_id}',
        key=f'{repo_name}_window{window_idx}_features'
    )
    logger.info(f"[DEBUG] Fetched features_data for {repo_name} window {window_idx}: {features_data}")
    if not features_data:
        logger.warning(f"No features for {repo_name} window {window_idx}, skipping")
        return
    features_df = pd.DataFrame(features_data)
    logger.info(f"[DEBUG] features_df shape: {features_df.shape}")
    logger.info(f"[DEBUG] modules in features_df: {features_df['module'].tolist() if 'module' in features_df.columns else 'No module column'}")
    
    repo_index = Config.REPO_NAMES.index(repo_name)
    base_project_key = Config.SONARQUBE_PROJECT_KEYS[repo_index]
    windows = None
    for group_name, group_windows in WINDOW_CONFIGS.items():
        if window_size_days in group_windows:
            windows = group_windows
            break
    if windows is None:
        logger.error(f"No window config found for window size {window_size_days}")
        return
    if window_idx >= len(windows) - 1:
        logger.error(f"Window index {window_idx} out of range for windows {windows}")
        return
    # List all SonarQube projects and match by base name and window size
    sonarqube_client = SonarQubeClient(
        Config.SONARQUBE_URL,
        Config.SONARQUBE_TOKEN
    )
    all_projects = sonarqube_client.list_all_projects()
    def match_project(base, window):
        for proj in all_projects:
            if base in proj and str(window) in proj:
                return proj
        return None
    window_start_project = match_project(base_project_key, windows[window_idx])
    window_end_project = match_project(base_project_key, windows[window_idx + 1])
    logger.info(f"   Window {window_idx}: [{windows[window_idx]}d → {windows[window_idx + 1]}d ago]")
    logger.info(f"   Label = quality@{window_start_project} - quality@{window_end_project}")
    if not window_start_project or not window_end_project:
        logger.error(f"Could not find SonarQube projects for base '{base_project_key}' and windows {windows[window_idx]}, {windows[window_idx + 1]}")
        return
    
    labeled_samples = []
    
    for _, row in features_df.iterrows():
        module = row['module']
        try:
            metrics = sonarqube_client.get_temporal_metrics(
                current_project_key=window_end_project,
                historical_project_key=window_start_project,
                file_path=module
            )
            if not metrics:
                logger.warning(f"No metrics returned for module '{module}' (projects: {window_end_project} vs {window_start_project})")
            elif 'quality_degradation' not in metrics:
                logger.warning(f"Metrics for module '{module}' missing 'quality_degradation': {metrics}")
            else:
                labeled_row = row.to_dict()
                labeled_row['needs_maintenance'] = metrics['quality_degradation']
                labeled_row['label_source'] = 'sonarqube_degradation_multi_window'
                labeled_row['window_start_project'] = window_start_project
                labeled_row['window_end_project'] = window_end_project
                labeled_row['degradation_days'] = windows[window_idx + 1] - windows[window_idx]
                labeled_samples.append(labeled_row)
        except Exception as e:
            logger.error(f"Exception labeling module '{module}': {e}")
            continue
    
    if not labeled_samples:
        logger.warning(f"No files labeled for {repo_name} window {window_idx}")
        return
    
    labeled_df = pd.DataFrame(labeled_samples)
    logger.info(f"✅ Labeled {len(labeled_df)} files for window {window_idx}")
    
    context['task_instance'].xcom_push(
        key=f'{repo_name}_window{window_idx}_labeled',
        value=labeled_df.to_dict('records')
    )

def aggregate_multi_window_data(**context):
    """Aggregate labeled data from all repos and time windows."""
    import pandas as pd
    
    logger.info("🔄 Aggregating multi-window labeled data...")
    
    all_labeled_data = []
    window_counts = {}
    
    for group_name, windows in WINDOW_CONFIGS.items():
        for repo_name in Config.REPO_NAMES:
            for window_idx in range(len(windows) - 1):
                window_start_days = windows[window_idx]
                window_end_days = windows[window_idx + 1]
                interval_str = f'{window_start_days}_{window_end_days}'
                label_task_id = f'label_{repo_name}_window{interval_str}d'
                labeled_data = context['task_instance'].xcom_pull(
                    task_ids=f'label_{group_name}.{label_task_id}',
                    key=f'{repo_name}_window{window_idx}_labeled'
                )
                logger.info(f"[DEBUG] Aggregation: Pulled from {group_name}.{label_task_id} key {repo_name}_window{window_idx}_labeled: {type(labeled_data)} len={len(labeled_data) if labeled_data else 0}")
                if labeled_data:
                    df = pd.DataFrame(labeled_data)
                    all_labeled_data.append(df)
                    window_counts[f'{repo_name}_window{window_idx}_{group_name}'] = len(df)
                    logger.info(f"  {repo_name} {group_name} window {window_idx}: {len(df)} samples")
    
    if not all_labeled_data:
        logger.error("❌ No labeled data collected from any repository/window")
        # raise ValueError("No labeled data available for training")
    
    combined_df = pd.concat(all_labeled_data, ignore_index=True)
    
    logger.info(f"\n📊 MULTI-WINDOW DATA SUMMARY")
    logger.info(f"   Total samples: {len(combined_df)}")
    logger.info(f"   Unique files: {combined_df['module'].nunique() if 'module' in combined_df.columns else 'N/A'}")
    logger.info(f"   Repos: {combined_df['repo_name'].nunique() if 'repo_name' in combined_df.columns else 'N/A'}")
    logger.info(f"   Windows: {combined_df['window_id'].nunique() if 'window_id' in combined_df.columns else 'N/A'}")
    
    # Log distributions
    if 'window_id' in combined_df.columns:
        window_dist = combined_df['window_id'].value_counts().sort_index().to_dict()
        logger.info(f"📊 Window distribution: {window_dist}")
    
    if 'repo_name' in combined_df.columns:
        repo_dist = combined_df['repo_name'].value_counts().to_dict()
        logger.info(f"📊 Repository distribution: {repo_dist}")
    
    if 'needs_maintenance' in combined_df.columns:
        logger.info(f"📊 Degradation stats:")
        logger.info(f"   Mean: {combined_df['needs_maintenance'].mean():.4f}")
        logger.info(f"   Std: {combined_df['needs_maintenance'].std():.4f}")
        logger.info(f"   Min: {combined_df['needs_maintenance'].min():.4f}")
        logger.info(f"   Max: {combined_df['needs_maintenance'].max():.4f}")
    
    logger.info("\n💾 Saving multi-window labeled data to MongoDB...")
    push_to_mongo(
        combined_df,
        mongo_uri=os.getenv('MONGO_URI', 'mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin'),
        mongo_db=os.getenv('MONGO_DB', 'risk_model_db'),
        mongo_collection="labeled_pr_data_multi_window_degradation"
    )
    
    xcom_df = combined_df.copy()
    for col in xcom_df.columns:
        if pd.api.types.is_datetime64_any_dtype(xcom_df[col]):
            xcom_df[col] = xcom_df[col].astype(str)
    
    context['task_instance'].xcom_push(key='combined_labeled_data', value=xcom_df.to_dict('records'))

def train_degradation_model(**context):
    """Train XGBoost model to predict quality degradation."""
    import pandas as pd
    
    logger.info("🚀 Training degradation prediction model...")
    
    # Pull aggregated data from XCom
    labeled_data = context['task_instance'].xcom_pull(
        task_ids='aggregate_multi_window_data',
        key='combined_labeled_data'
    )
    
    if not labeled_data:
        logger.error("❌ No labeled data available for training")
        raise ValueError("No labeled data found")
    
    df = pd.DataFrame(labeled_data)
    
    logger.info(f"📊 Training with {len(df)} samples")
    
    tune_hyperparameters = os.getenv("TUNE_HYPERPARAMETERS", "false").lower() == "true"
    if tune_hyperparameters:
        logger.info("🔍 Hyperparameter tuning ENABLED (this will take longer)")
    model_path = Config.MODELS_DIR / "xgboost_degradation_model.pkl"
    trainer = RiskModelTrainer(
        model_path=str(model_path),
        tune_hyperparameters=tune_hyperparameters
    )
    
    metrics = trainer.train(df, use_class_weights=False)
    
    logger.info("✅ Model training completed")
    logger.info(f"📊 Performance metrics:")
    logger.info(f"   MAE: {metrics.get('mae', 'N/A')}")
    logger.info(f"   RMSE: {metrics.get('rmse', 'N/A')}")
    logger.info(f"   R²: {metrics.get('r2', 'N/A')}")
    
    log_model_metrics(
        metrics=metrics,
        model_name="xgboost_degradation_model",
        label_source_filter=LABEL_SOURCE_FILTER,
        training_samples=len(df)
    )



with DAG(
    'risk_model_training_degradation',
    default_args=default_args,
    description='Train model to predict quality degradation using multi-window temporal analysis',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'training', 'degradation', 'temporal', 'multi-window']
) as dag:
    verify_task = PythonOperator(
        task_id='verify_multi_window_projects',
        python_callable=verify_multi_window_projects,
        provide_context=True
    )

    # Create TaskGroups for each window config using explicit intervals
    fetch_groups = {}
    label_groups = {}
    for group_name, windows in WINDOW_CONFIGS.items():
        with TaskGroup(f'fetch_{group_name}') as fetch_group:
            for repo_name in Config.REPO_NAMES:
                for window_idx in range(len(windows) - 1):
                    # Only process valid intervals
                    if window_idx + 1 >= len(windows):
                        continue
                    window_start_days = windows[window_idx]
                    window_end_days = windows[window_idx + 1]
                    interval_str = f'{window_start_days}_{window_end_days}'
                    task_id = f'fetch_{repo_name}_window{interval_str}d'
                    PythonOperator(
                        task_id=task_id,
                        python_callable=fetch_multi_window_data_task,
                        op_kwargs={
                            'repo_name': repo_name,
                            'window_idx': window_idx,
                            'window_size_days': window_start_days,
                            'group_name': group_name
                        },
                        provide_context=True
                    )
        fetch_groups[group_name] = fetch_group

        with TaskGroup(f'label_{group_name}') as label_group:
            for repo_name in Config.REPO_NAMES:
                for window_idx in range(len(windows) - 1):
                    # Only process valid intervals
                    if window_idx + 1 >= len(windows):
                        continue
                    window_start_days = windows[window_idx]
                    window_end_days = windows[window_idx + 1]
                    interval_str = f'{window_start_days}_{window_end_days}'
                    task_id = f'label_{repo_name}_window{interval_str}d'
                    PythonOperator(
                        task_id=task_id,
                        python_callable=create_multi_window_labels_task,
                        op_kwargs={
                            'repo_name': repo_name,
                            'window_idx': window_idx,
                            'window_size_days': window_start_days,
                            'group_name': group_name
                        },
                        provide_context=True
                    )
        label_groups[group_name] = label_group

    aggregate_task = PythonOperator(
        task_id='aggregate_multi_window_data',
        python_callable=aggregate_multi_window_data,
        provide_context=True
    )

    train_task = PythonOperator(
        task_id='train_degradation_model',
        python_callable=train_degradation_model,
        provide_context=True
    )

    # Set dependencies for all groups
    for group_name in WINDOW_CONFIGS.keys():
        verify_task >> fetch_groups[group_name] >> label_groups[group_name] >> aggregate_task
    aggregate_task >> train_task
