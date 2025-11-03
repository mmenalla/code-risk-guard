import logging

import pandas as pd
import joblib

from src.data.feature_engineering import FeatureEngineer
from src.data.github_client import GitHubDataCollector
from src.utils.config import Config


class RiskPredictor:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.MODELS_DIR / "xgboost_risk_model.json"
        self.model = joblib.load(self.model_path)
        self.fe = FeatureEngineer()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict maintenance risk for modules.
        Input df: raw module stats from GitHub
        Returns df with predicted risk score (0-1) and optional risk category.
        """
        df_features = self.fe.transform(df)
        logging.info(f"Features before cleanup: {len(df_features.columns)} columns")
        
        # Store metadata for later (keep module, repo_name, filename)
        metadata_cols = ['module', 'repo_name', 'filename', 'created_at']
        metadata = df_features[[col for col in metadata_cols if col in df_features.columns]].copy()
        
        # Drop metadata columns (same as training pipeline)
        drop_cols = [
            'module', 'repo_name', 'created_at', 'filename', 'label_source', 
            'risk_category', 'feedback_count', 'last_feedback_at',
            # Drop features not used in training v20+
            'bug_ratio', 'churn_per_pr', 'author_concentration'
        ]
        
        X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns], errors='ignore')
        
        # Drop any remaining non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
        if len(non_numeric_cols) > 0:
            logging.warning(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols, errors='ignore')
        
        # Ensure column order matches training
        # Get expected feature names from the model
        expected_features = self.model.get_booster().feature_names
        if expected_features:
            logging.info(f"Model expects features in order: {expected_features}")
            # Reorder columns to match model's expectation
            X = X[expected_features]
        
        logging.info(f"Features for prediction: {X.columns.tolist()}")
        logging.info(f"Using {len(X.columns)} features for inference")

        # Predict continuous risk score
        risk_score = self.model.predict(X)
        
        # Restore metadata
        result = metadata.copy()
        result['risk_score'] = risk_score

        # Use updated thresholds matching training (v15+)
        result['risk_category'] = pd.cut(
            risk_score,
            bins=[0, 0.22, 0.47, 0.65, 1.0],
            labels=['no-risk', 'low-risk', 'medium-risk', 'high-risk'],
            include_lowest=True
        )

        return result
