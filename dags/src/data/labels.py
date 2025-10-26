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
        
        # Boost score if enhanced features indicate additional risk
        if 'author_concentration' in df.columns:
            # Single author = knowledge risk (boost by up to 10%)
            ownership_boost = df['author_concentration'].clip(0, 1) * 0.1
            base_score = (base_score + ownership_boost).clip(0, 1)
        
        if 'bug_density' in df.columns:
            # High bug density = additional risk signal (boost by up to 10%)
            density_max = df['bug_density'].quantile(0.95) if len(df) > 10 else df['bug_density'].max()
            if density_max > 0:
                density_boost = (df['bug_density'] / density_max).clip(0, 1) * 0.1
                base_score = (base_score + density_boost).clip(0, 1)
        
        return base_score
