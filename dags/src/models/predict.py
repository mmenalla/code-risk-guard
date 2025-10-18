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
        Returns df with predicted probability and risk label.
        """
        df_features = self.fe.transform(df)
        logging.warning(f"Features for prediction: {df_features.columns.tolist()}")
        feature_cols = [c for c in df_features.columns if c != 'module']
        X = df_features[feature_cols]
        non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
        if len(non_numeric_cols) > 0:
            print(f"Dropping non-numeric columns for inference: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols, errors='ignore')
            
        # Predict probability of maintenance need
        prob = self.model.predict_proba(X)[:, 1]
        df_features['risk_score'] = prob
        df_features['needs_maintenance'] = (prob >= 0.5).astype(int)

        return df_features
