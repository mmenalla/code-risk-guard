import sys
from typing import Optional, List
from src.data.sonarqube_client import SonarQubeClient
from src.utils.config import Config
from src.data.save_incremental_labeled_data import load_labeled_data_from_mongo
def run_training_for_windows(project_keys: List[str], window_size_days: int, num_windows: int, trainer_kwargs: Optional[dict] = None):
    """
    For each project and window, load data and train model.
    """
    for project_key in project_keys:
        for window_index in range(num_windows):
            # Load labeled data from MongoDB for this project and window
            df = load_labeled_data_from_mongo(
                project_key=project_key,
                window_size_days=window_size_days,
                window_index=window_index
            )
            if df is None or df.empty:
                logging.warning(f"No data for project={project_key}, window_size_days={window_size_days}, window_index={window_index}")
                continue
            trainer = RiskModelTrainer(**(trainer_kwargs or {}))
            logging.info(f"=== Training for project={project_key}, window_size_days={window_size_days}, window_index={window_index} ===")
            metrics = trainer.train(df)
            logging.info(f"Metrics for {project_key} window {window_size_days}d idx {window_index}: {metrics}")



    # Load all SonarQube project keys from config
    project_keys = Config.SONARQUBE_PROJECT_KEYS

    # Two 150-day windows
    run_training_for_windows(project_keys, window_size_days=150, num_windows=2)
    # Four 50-day windows
    run_training_for_windows(project_keys, window_size_days=50, num_windows=4)

def main():
    # Load all SonarQube project keys from config
    project_keys = Config.SONARQUBE_PROJECT_KEYS

    # Two 150-day windows
    run_training_for_windows(project_keys, window_size_days=150, num_windows=2)
    # Four 50-day windows
    run_training_for_windows(project_keys, window_size_days=50, num_windows=4)

if __name__ == "__main__":
    main()
from pathlib import Path
import os
import re
import joblib
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.config import Config
from src.data.save_incremental_labeled_data import log_model_metrics


class RiskModelTrainer:
    def __init__(self, model_path: str = None, tune_hyperparameters: bool = False):
        # Ensure model_path is a Path object
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = Config.MODELS_DIR / "xgboost_risk_model.pkl"
        self.model = None
        self.tune_hyperparameters = tune_hyperparameters
        self.logger = logging.getLogger(__name__)

    def load_existing_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Loaded existing model from {self.model_path} for fine-tuning")
        else:
            self.logger.info("No existing model found. Training from scratch.")
    
    def tune_model_hyperparameters(self, X_train, y_train):
        self.logger.info("🔍 Starting hyperparameter tuning...")
        
        # Define parameter search space
        param_distributions = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [5, 6, 7, 8, 9],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'min_child_weight': [1, 2, 3, 4],
            'subsample': [0.7, 0.8, 0.85, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.85, 0.9],
            'gamma': [0, 0.05, 0.1, 0.2],
            'reg_alpha': [0, 0.05, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0],
        }
        
        # Base model for tuning
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42,
            tree_method='hist'
        )
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=20,
            scoring='r2',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        self.logger.info(f"✅ Best R² from tuning: {random_search.best_score_:.4f}")
        self.logger.info(f"✅ Best hyperparameters:")
        for param, value in random_search.best_params_.items():
            self.logger.info(f"   {param:20s}: {value}")
        
        return random_search.best_estimator_

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.
        Note: lines_per_author, churn_per_commit, bug_ratio, commits_per_day 
        are already calculated in temporal_git_collector.py
        """
        df = df.copy()
        
        # 1. net_lines - Code growth
        if 'lines_added' in df.columns and 'lines_deleted' in df.columns:
            df['net_lines'] = df['lines_added'] - df['lines_deleted']
        
            # 0. window_size_days - Time window for aggregation (must be present)
            if 'window_size_days' not in df.columns:
                raise ValueError("window_size_days column must be present in training data for multi-window modeling.")
            # 1. net_lines - Code growth
            df['code_stability'] = df['churn'] / (df['lines_added'] + 1)
        
            # 2. code_stability - Churn relative to additions
        if 'churn_per_commit' in df.columns:
            df['is_high_churn_commit'] = (df['churn_per_commit'] > 100).astype(int)
            # 3. is_high_churn_commit - Binary flag for large changes
        # 4. bug_commit_rate - Proportion of bug commits (ratio is 1 if all commits are bug commits)
        if 'bug_commits' in df.columns and 'commits' in df.columns:
            # 4. bug_commit_rate - Proportion of bug commits (ratio is 1 if all commits are bug commits)
        
        # 5. commits_squared - Non-linear commit activity
            # 5. commits_squared - Non-linear commit activity
            df['commits_squared'] = df['commits'] ** 2
        
            # 6. author_concentration - Bus factor
        if 'authors' in df.columns:
            df['author_concentration'] = 1.0 / (df['authors'] + 1)
            # 7. lines_per_commit - Average code change size
        # 7. lines_per_commit - Average code change size
        if 'lines_added' in df.columns and 'commits' in df.columns:
            # 8. churn_rate - Churn velocity
        
        # 8. churn_rate - Churn velocity
            # 9. modification_ratio - Deletion relative to addition
            df['churn_rate'] = df['churn'] / (df['days_active'] + 1)
        
            # 10. churn_per_author - Code change per developer
        if 'lines_added' in df.columns and 'lines_deleted' in df.columns:
            df['modification_ratio'] = df['lines_deleted'] / (df['lines_added'] + 1)
            # 11. deletion_rate - Code removal rate
        # 10. churn_per_author - Code change per developer
        if 'churn' in df.columns and 'authors' in df.columns:
            # 12. commit_density - Commit frequency
        
        # 11. deletion_rate - Code removal rate
            self.logger.info(f"✨ Feature engineering complete. Added engineered features (including window_size_days)")
        if 'lines_deleted' in df.columns and 'lines_added' in df.columns:
            df['deletion_rate'] = df['lines_deleted'] / (df['lines_added'] + df['lines_deleted'] + 1)
        
        # 12. commit_density - Commit frequency
        if 'commits' in df.columns and 'days_active' in df.columns:
            df['commit_density'] = df['commits'] / (df['days_active'] + 1)
        
        self.logger.info(f"✨ Feature engineering complete. Added engineered features")
        return df

    def train(self, df: pd.DataFrame, feature_cols: list = None, use_class_weights: bool = False):
        df = df.copy()
        self.logger.info("🔧 Applying feature engineering...")
        df = self.engineer_features(df)

        # Use only the features specified for inference
        allowed_features = [
            'net_lines',
            'modification_ratio',
            'lines_per_commit',
            'lines_deleted',
            'days_active',
            'commits_squared',
            'commits_per_day',
            'commits',
            'commit_density',
            'code_stability',
            'bug_ratio',
            'bug_commits',
            'bug_commit_rate',
            'author_concentration',
            'window_size_in_days',
        ]
        feature_cols = allowed_features
        self.logger.info(f"Training with {len(feature_cols)} features (inference-matched)")
        self.logger.info(f"Feature sample: {feature_cols}")

        # Drop missing features if any
        X = df[[f for f in feature_cols if f in df.columns]].copy()
        missing = set(feature_cols) - set(X.columns)
        if missing:
            self.logger.warning(f"Missing features for training: {missing}. Filling with 0.")
            for feat in missing:
                X[feat] = 0
        X = X[feature_cols]
        y = df['needs_maintenance']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Determine window size for naming
        window_size = None
        if 'window_size_days' in df.columns:
            unique_windows = df['window_size_days'].unique()
            if len(unique_windows) == 1:
                window_size = unique_windows[0]
            else:
                window_size = 'multiwindow'
        else:
            window_size = 'unknownwindow'

        # Use window size in log and model name
        self.logger.info(f"🪟 Training task for window size: {window_size}")

        if self.tune_hyperparameters:
            self.model = self.tune_model_hyperparameters(X_train, y_train)
        else:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=300,
                max_depth=7,
                learning_rate=0.03,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                colsample_bylevel=0.85,
                gamma=0.05,
                reg_alpha=0.05,
                reg_lambda=1.5,
                random_state=42,
                tree_method='hist',
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

        self.logger.info(f"📊 Training completed with {self.model.n_estimators} trees")
        if hasattr(self.model, 'best_iteration'):
            self.logger.info(f"   Best iteration: {self.model.best_iteration}")

        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.logger.info(f"🎯 Top 5 most important features:")
            for idx, row in feature_importance.head(5).iterrows():
                self.logger.info(f"   {row['feature']:25s}: {row['importance']:.4f}")

        # Save model with window size in name
        model_name = self.save_model(suffix=f"_{window_size}")
        # Save training prediction statistics for calibration
        train_preds = self.model.predict(X_train)
        stats = {
            "mean": float(np.mean(train_preds)),
            "std": float(np.std(train_preds)),
            "min": float(np.min(train_preds)),
            "max": float(np.max(train_preds))
        }
        import json
        metadata_path = Path(model_name).parent / (Path(model_name).stem + "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"Saved training prediction statistics to {metadata_path}")
        metrics = self.evaluate(X_test, y_test, model_name, label_source_filter="sonarqube", training_samples=len(df))
        return metrics

    def evaluate(self, X_test, y_test, model_name: str, label_source_filter: str = "sonarqube", training_samples: int = 0):
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        self.logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }
        log_model_metrics(
            metrics, 
            model_name=model_name.split("/")[-1].rsplit(".", 1)[0],
            label_source_filter=label_source_filter,
            training_samples=training_samples
        )
        return metrics

    def save_model(self, suffix: str = ""):
        """Versioned model saving with optional suffix (e.g., window size)"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        base_name = self.model_path.stem + suffix
        ext = self.model_path.suffix
        existing = list(self.model_path.parent.glob(f"{base_name}_v*{ext}"))

        pattern = re.compile(rf"{base_name}_v(\d+){ext}")
        version_numbers = [int(m.group(1)) for f in existing if (m := pattern.match(f.name))]
        next_version = max(version_numbers, default=0) + 1

        new_model_path = self.model_path.parent / f"{base_name}_v{next_version}{ext}"
        joblib.dump(self.model, new_model_path)
        logging.info(f"Model saved to {new_model_path}")
        return str(new_model_path)

    def load_model(self, model_path: str):
        """Load a specific trained model for evaluation"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = joblib.load(model_path)
        self.logger.info(f"Loaded model from {model_path}")
        return model
