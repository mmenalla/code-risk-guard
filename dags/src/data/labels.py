import pandas as pd
import os
import logging
from pymongo import MongoClient

from src.data.feature_engineering import FeatureEngineer
from src.data.github_client import GitHubDataCollector

logger = logging.getLogger(__name__)


class LabelCreator:
    def __init__(
        self, 
        mongo_uri: str = None,
        mongo_db: str = None,
        feedback_collection: str = "risk_feedback",
        use_manager_feedback: bool = True,
        fallback_to_heuristic: bool = True
    ):
        """
        Hybrid labeling strategy: Use manager feedback as ground truth, fall back to heuristics.
        
        Parameters:
        - mongo_uri: MongoDB connection string (defaults to env var MONGO_URI)
        - mongo_db: MongoDB database name (defaults to env var MONGO_DB)
        - feedback_collection: Collection name for manager feedback
        - use_manager_feedback: Whether to fetch and use manager feedback
        - fallback_to_heuristic: Use heuristic scoring for modules without feedback
        """
        self.mongo_uri = mongo_uri or os.getenv('MONGO_URI')
        self.mongo_db = mongo_db or os.getenv('MONGO_DB', 'risk_model_db')
        self.feedback_collection = feedback_collection
        self.use_manager_feedback = use_manager_feedback
        self.fallback_to_heuristic = fallback_to_heuristic

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels using hybrid strategy:
        1. Fetch manager feedback from MongoDB
        2. Use manager risk scores where available
        3. Fall back to heuristic scoring for modules without feedback
        """
        df = df.copy()
        df['prs'] = df['prs'].replace(0, 1)
        
        # Try to fetch manager feedback
        if self.use_manager_feedback and self.mongo_uri:
            feedback_df = self._fetch_feedback_labels(df['module'].tolist())
            
            if not feedback_df.empty:
                logger.info(f"Found manager feedback for {len(feedback_df)} modules")
                
                # Merge manager feedback
                df = df.merge(
                    feedback_df[['module', 'manager_risk', 'feedback_count']], 
                    on='module', 
                    how='left'
                )
                
                # Use manager risk where available
                df['needs_maintenance'] = df['manager_risk']
                
                # Track which labels are from managers vs heuristics
                df['label_source'] = 'manager'
                
                # For modules without feedback, use heuristic if enabled
                if self.fallback_to_heuristic:
                    mask = df['needs_maintenance'].isna()
                    heuristic_count = mask.sum()
                    
                    if heuristic_count > 0:
                        logger.info(f"Using heuristic labels for {heuristic_count} modules without feedback")
                        df.loc[mask, 'needs_maintenance'] = self._compute_heuristic_score(df[mask])
                        df.loc[mask, 'label_source'] = 'heuristic'
                
                # Clean up temporary columns
                df = df.drop(columns=['manager_risk'], errors='ignore')
            else:
                logger.info("No manager feedback found, using heuristic labels for all modules")
                df['needs_maintenance'] = self._compute_heuristic_score(df)
                df['label_source'] = 'heuristic'
                df['feedback_count'] = 0
        else:
            # No feedback collection configured, use heuristic
            logger.info("Manager feedback disabled, using heuristic labels")
            df['needs_maintenance'] = self._compute_heuristic_score(df)
            df['label_source'] = 'heuristic'
            df['feedback_count'] = 0
        
        # Create risk categories
        df["risk_category"] = pd.cut(
            df["needs_maintenance"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["no-risk", "low-risk", "medium-risk", "high-risk"],
            include_lowest=True
        )

        return df
    
    def _fetch_feedback_labels(self, modules: list) -> pd.DataFrame:
        """
        Fetch latest manager risk scores from MongoDB risk_feedback collection.
        
        Returns DataFrame with columns: module, manager_risk, feedback_count
        """
        if not self.mongo_uri:
            logger.warning("MongoDB URI not configured, cannot fetch manager feedback")
            return pd.DataFrame()
        
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            
            # Debug: Check if risk_feedback collection exists and has data
            total_feedback_count = db[self.feedback_collection].count_documents({})
            logger.info(f"Total documents in {self.feedback_collection}: {total_feedback_count}")
            
            if total_feedback_count == 0:
                logger.warning(f"No feedback found in {self.feedback_collection} collection")
                return pd.DataFrame()
            
            # Debug: Show sample feedback document structure
            sample_feedback = db[self.feedback_collection].find_one({})
            if sample_feedback:
                logger.info(f"Sample feedback document keys: {list(sample_feedback.keys())}")
                logger.info(f"Sample module field: {sample_feedback.get('module', 'NOT FOUND')}")
            
            # Debug: Log how many modules we're looking for
            logger.info(f"Looking for feedback for {len(modules)} modules")
            logger.info(f"Sample modules from features: {modules[:3] if len(modules) > 0 else 'none'}")
            
            # Get latest feedback for each module with count of total feedbacks
            pipeline = [
                {'$match': {'module': {'$in': modules}}},
                {'$sort': {'created_at': -1}},
                {'$group': {
                    '_id': '$module',
                    'manager_risk': {'$first': '$manager_risk'},
                    'created_at': {'$first': '$created_at'},
                    'feedback_count': {'$sum': 1}
                }}
            ]
            
            results = list(db[self.feedback_collection].aggregate(pipeline))
            
            if not results:
                logger.warning(f"No matching feedback found for {len(modules)} modules")
                # Debug: Show what modules exist in feedback
                all_feedback_modules = db[self.feedback_collection].distinct('module')
                logger.info(f"Modules in feedback collection: {all_feedback_modules[:5]}")
                logger.info(f"Modules we're searching for: {modules[:5]}")
                return pd.DataFrame()
            
            feedback_df = pd.DataFrame([
                {
                    'module': r['_id'], 
                    'manager_risk': r['manager_risk'],
                    'feedback_count': r['feedback_count'],
                    'last_feedback_at': r['created_at']
                }
                for r in results
            ])
            
            logger.info(f"✅ Fetched feedback for {len(feedback_df)} modules (avg {feedback_df['feedback_count'].mean():.1f} updates per module)")
            
            return feedback_df
            
        except Exception as e:
            logger.error(f"❌ Error fetching manager feedback: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        finally:
            try:
                client.close()
            except:
                pass
    
    def _compute_heuristic_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Improved heuristic scoring with better weighting.
        
        Strategy:
        - Bug ratio weighted higher (0.7) - bugs are strongest signal
        - Churn weighted lower (0.3) - context-dependent
        - Uses enhanced features if available
        """
        # Normalize bug_ratio to 0-1 (already mostly in that range)
        bug_score = df['bug_ratio'].clip(0, 1)
        
        # Normalize churn using max in dataset (or 95th percentile to avoid outliers)
        churn_max = df['churn_per_pr'].quantile(0.95) if len(df) > 10 else df['churn_per_pr'].max()
        churn_max = max(churn_max, 1)  # Avoid division by zero
        churn_score = (df['churn_per_pr'] / churn_max).clip(0, 1)
        
        # Base score: weighted combination (bugs weighted higher)
        base_score = 0.7 * bug_score + 0.3 * churn_score
        
        # # Boost score if enhanced features indicate additional risk
        # if 'author_concentration' in df.columns:
        #     # Single author = knowledge risk (boost by up to 10%)
        #     ownership_boost = df['author_concentration'].clip(0, 1) * 0.1
        #     base_score = (base_score + ownership_boost).clip(0, 1)
        
        # if 'bug_density' in df.columns:
        #     # High bug density = additional risk signal (boost by up to 10%)
        #     density_max = df['bug_density'].quantile(0.95) if len(df) > 10 else df['bug_density'].max()
        #     if density_max > 0:
        #         density_boost = (df['bug_density'] / density_max).clip(0, 1) * 0.1
        #         base_score = (base_score + density_boost).clip(0, 1)
        
        return base_score


def create_labels_with_sonarqube(
    df: pd.DataFrame,
    sonarqube_url: str = None,
    sonarqube_token: str = None,
    mongo_uri: str = None,
    mongo_db: str = None,
    feedback_collection: str = "risk_feedback",
    use_manager_feedback: bool = True,
    fallback_to_heuristic: bool = True
) -> pd.DataFrame:
    """
    Create labels using SonarQube code quality metrics with fallback strategy.
    
    Priority order:
    1. Manager feedback (if available and use_manager_feedback=True)
    2. SonarQube metrics (if available)
    3. Heuristic scores (fallback)
    
    Parameters:
    - df: DataFrame with features (must have 'module' and 'repo_name' columns)
    - sonarqube_url: SonarQube server URL (defaults to config)
    - sonarqube_token: SonarQube auth token (defaults to config)
    - mongo_uri: MongoDB connection for manager feedback
    - mongo_db: MongoDB database name
    - feedback_collection: Collection name for manager feedback
    - use_manager_feedback: Whether to use manager feedback as highest priority
    - fallback_to_heuristic: Use heuristic scoring when SonarQube unavailable
    
    Returns:
    - DataFrame with 'needs_maintenance', 'label_source', and 'risk_category' columns
    """
    from src.utils.config import Config
    from src.data.sonarqube_client import SonarQubeClient
    
    df = df.copy()
    df['prs'] = df['prs'].replace(0, 1)
    
    # Initialize tracking columns
    df['needs_maintenance'] = None
    df['label_source'] = None
    df['feedback_count'] = 0
    
    # Priority 1: Manager Feedback
    if use_manager_feedback and mongo_uri:
        labeler = LabelCreator(
            mongo_uri=mongo_uri,
            mongo_db=mongo_db,
            feedback_collection=feedback_collection,
            use_manager_feedback=True,
            fallback_to_heuristic=False
        )
        feedback_df = labeler._fetch_feedback_labels(df['module'].tolist())
        
        if not feedback_df.empty:
            logger.info(f"Found manager feedback for {len(feedback_df)} modules")
            df = df.merge(
                feedback_df[['module', 'manager_risk', 'feedback_count']], 
                on='module', 
                how='left'
            )
            # Use manager risk where available
            mask = df['manager_risk'].notna()
            df.loc[mask, 'needs_maintenance'] = df.loc[mask, 'manager_risk']
            df.loc[mask, 'label_source'] = 'manager'
            df = df.drop(columns=['manager_risk'], errors='ignore')
    
    # Priority 2: SonarQube Metrics
    sonarqube_url = sonarqube_url or Config.SONARQUBE_URL
    sonarqube_token = sonarqube_token or Config.SONARQUBE_TOKEN
    
    if sonarqube_url and sonarqube_token and Config.SONARQUBE_PROJECT_KEYS:
        try:
            logger.info(f"Initializing SonarQube client at {sonarqube_url}")
            sonarqube_client = SonarQubeClient(sonarqube_url, sonarqube_token)
            
            # Create mapping from repo_name to project_key
            # Use REPO_NAMES for local repos, fallback to GITHUB_REPOS for GitHub-based analysis
            repo_list = Config.REPO_NAMES if Config.REPO_NAMES else Config.GITHUB_REPOS
            repo_to_project = {}
            if len(repo_list) == len(Config.SONARQUBE_PROJECT_KEYS):
                repo_to_project = dict(zip(repo_list, Config.SONARQUBE_PROJECT_KEYS))
                logger.info(f"Created repo-to-project mapping: {repo_to_project}")
            else:
                logger.warning(f"Mismatch: {len(repo_list)} repos vs {len(Config.SONARQUBE_PROJECT_KEYS)} SonarQube projects")
            
            sonarqube_count = 0
            for repo_name in df['repo_name'].unique():
                if repo_name not in repo_to_project:
                    logger.warning(f"No SonarQube project key for repo: {repo_name}")
                    continue
                
                project_key = repo_to_project[repo_name]
                repo_mask = df['repo_name'] == repo_name
                
                # Get files for this repo that don't already have labels
                unlabeled_mask = repo_mask & df['needs_maintenance'].isna()
                repo_files = df[unlabeled_mask]['module'].tolist()
                
                logger.info(f"Fetching SonarQube metrics for {len(repo_files)} files in {repo_name} (project: {project_key})")
                
                for file_path in repo_files:
                    file_mask = (df['repo_name'] == repo_name) & (df['module'] == file_path)
                    
                    # Fetch SonarQube metrics for file
                    metrics = sonarqube_client.get_file_measures(project_key, file_path)
                    
                    if metrics:
                        # Calculate maintainability score
                        score = sonarqube_client.calculate_maintainability_score(metrics)
                        
                        df.loc[file_mask, 'needs_maintenance'] = score
                        df.loc[file_mask, 'label_source'] = 'sonarqube'
                        sonarqube_count += 1
                        
                        # Store raw metrics (optional, for analysis)
                        for metric_key, metric_value in metrics.items():
                            col_name = f'sonarqube_{metric_key}'
                            if col_name not in df.columns:
                                df[col_name] = None
                            df.loc[file_mask, col_name] = metric_value
            
            logger.info(f"✅ Applied SonarQube labels to {sonarqube_count} files")
            
        except Exception as e:
            logger.error(f"❌ Error fetching SonarQube metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning("SonarQube not configured (missing URL, token, or project keys)")
    
    # Priority 3: Heuristic Fallback
    if fallback_to_heuristic:
        mask = df['needs_maintenance'].isna()
        heuristic_count = mask.sum()
        
        if heuristic_count > 0:
            logger.info(f"Using heuristic labels for {heuristic_count} modules without SonarQube/feedback data")
            labeler = LabelCreator(use_manager_feedback=False, fallback_to_heuristic=True)
            df.loc[mask, 'needs_maintenance'] = labeler._compute_heuristic_score(df[mask])
            df.loc[mask, 'label_source'] = 'heuristic'
    
    # Priority 4: Heuristic for any remaining NaN values (even when fallback_to_heuristic=False)
    # This handles edge cases like non-Python files that SonarQube doesn't analyze
    remaining_na = df['needs_maintenance'].isna().sum()
    if remaining_na > 0:
        logger.info(f"Using heuristic for {remaining_na} files without SonarQube metrics (non-Python files)")
        labeler = LabelCreator(use_manager_feedback=False, fallback_to_heuristic=True)
        mask = df['needs_maintenance'].isna()
        df.loc[mask, 'needs_maintenance'] = labeler._compute_heuristic_score(df[mask])
        df.loc[mask, 'label_source'] = 'heuristic'

    # Create risk categories with adjusted thresholds for better class balance
    # Thresholds optimized for SonarQube data distribution (85 high-risk samples vs 7 before)
    df["risk_category"] = pd.cut(
        df["needs_maintenance"],
        bins=[0, 0.22, 0.47, 0.65, 1.0],
        labels=["no-risk", "low-risk", "medium-risk", "high-risk"],
        include_lowest=True
    )
    
    # Log label source distribution
    label_counts = df['label_source'].value_counts()
    logger.info(f"📊 Label sources: {label_counts.to_dict()}")
    
    return df
