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
        
        # 2. code_stability - Churn relative to additions
        if 'lines_added' in df.columns and 'churn' in df.columns:
            df['code_stability'] = df['churn'] / (df['lines_added'] + 1)
        
        # 3. is_high_churn_commit - Binary flag for large changes
        if 'churn_per_commit' in df.columns:
            df['is_high_churn_commit'] = (df['churn_per_commit'] > 100).astype(int)
        
        # 4. bug_commit_rate - Proportion of bug commits
        if 'bug_commits' in df.columns and 'commits' in df.columns:
            df['bug_commit_rate'] = df['bug_commits'] / (df['commits'] + 1)
        
        # 5. commits_squared - Non-linear commit activity
        if 'commits' in df.columns:
            df['commits_squared'] = df['commits'] ** 2
        
        # 6. author_concentration - Bus factor
        if 'authors' in df.columns:
            df['author_concentration'] = 1.0 / (df['authors'] + 1)
        
        # 7. lines_per_commit - Average code change size
        if 'lines_added' in df.columns and 'commits' in df.columns:
            df['lines_per_commit'] = df['lines_added'] / (df['commits'] + 1)
        
        # 8. churn_rate - Churn velocity
        if 'churn' in df.columns and 'days_active' in df.columns:
            df['churn_rate'] = df['churn'] / (df['days_active'] + 1)
        
        # 9. modification_ratio - Deletion relative to addition
        if 'lines_added' in df.columns and 'lines_deleted' in df.columns:
            df['modification_ratio'] = df['lines_deleted'] / (df['lines_added'] + 1)
        
        # 10. churn_per_author - Code change per developer
        if 'churn' in df.columns and 'authors' in df.columns:
            df['churn_per_author'] = df['churn'] / (df['authors'] + 1)
        
        # 11. deletion_rate - Code removal rate
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
        
        exclude_cols = [
            'module', 'needs_maintenance', 'repo_name', 'created_at', '_id', 
            'risk_category', 'filename', 'last_modified', 'label_source',
            'window_id', 'window_start', 'window_end', 'window_size_days',
            'current_sonar_project', 'future_sonar_project',
        ]
        
        sonarqube_prefixes = [
            'sonarqube_', 'current_', 'historical_', 'quality_degradation',
            'complexity_delta', 'code_smells_delta', 'bugs_delta',
            'vulnerabilities_delta', 'technical_debt_delta',
        ]
        
        if feature_cols is None:
            feature_cols = []
            for c in df.columns:
                if c in exclude_cols:
                    continue
                
                is_sonarqube_col = any(c.startswith(prefix) or c == prefix for prefix in sonarqube_prefixes)
                if is_sonarqube_col:
                    continue
                
                if pd.api.types.is_numeric_dtype(df[c]):
                    feature_cols.append(c)
        
        self.logger.info(f"Training with {len(feature_cols)} features")
        self.logger.info(f"Feature sample: {feature_cols[:10]}...")

        X = df[feature_cols]
        y = df['needs_maintenance']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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

        model_name = self.save_model()
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

    def save_model(self):
        """Versioned model saving"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        base_name = self.model_path.stem
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
