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
        """Load an existing model if available for fine-tuning"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Loaded existing model from {self.model_path} for fine-tuning")
        else:
            self.logger.info("No existing model found. Training from scratch.")
    
    def tune_model_hyperparameters(self, X_train, y_train):
        """
        Use RandomizedSearchCV to find optimal hyperparameters.
        This is computationally expensive but can significantly improve R².
        """
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
        
        # Randomized search with cross-validation
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=20,  # Try 20 random combinations
            scoring='r2',
            cv=3,  # 3-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        random_search.fit(X_train, y_train)
        
        self.logger.info(f"✅ Best R² from tuning: {random_search.best_score_:.4f}")
        self.logger.info(f"✅ Best hyperparameters:")
        for param, value in random_search.best_params_.items():
            self.logger.info(f"   {param:20s}: {value}")
        
        return random_search.best_estimator_

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to improve model performance.
        
        Feature Engineering Techniques:
        1. Interaction terms (multiplicative features)
        2. Temporal decay features
        3. Ratio features
        4. Polynomial features for key metrics
        5. Developer collaboration metrics
        6. Complexity growth indicators
        7. Code review quality proxies
        """
        df = df.copy()
        
        # 1. INTERACTION TERMS - capture combined effects
        if 'churn' in df.columns and 'commits' in df.columns:
            df['churn_per_commit'] = df['churn'] / (df['commits'] + 1)
            
        if 'lines_added' in df.columns and 'commits' in df.columns:
            df['lines_per_commit'] = df['lines_added'] / (df['commits'] + 1)
            
        if 'lines_added' in df.columns and 'lines_deleted' in df.columns:
            df['net_lines'] = df['lines_added'] - df['lines_deleted']
            df['modification_ratio'] = df['lines_deleted'] / (df['lines_added'] + 1)
            
        if 'bug_commits' in df.columns and 'commits' in df.columns:
            df['bug_commit_rate'] = df['bug_commits'] / (df['commits'] + 1)
            
        # 2. TEMPORAL DECAY FEATURES - older activity matters less
        if 'file_age_days' in df.columns:
            df['file_age_months'] = df['file_age_days'] / 30.0
            df['file_age_log'] = np.log1p(df['file_age_days'])
            
        if 'avg_commit_interval' in df.columns:
            df['commit_frequency'] = 1.0 / (df['avg_commit_interval'] + 1)
            
        # 3. ACTIVITY INTENSITY FEATURES
        if 'churn' in df.columns and 'authors' in df.columns:
            df['churn_per_author'] = df['churn'] / (df['authors'] + 1)
            
        if 'commits' in df.columns and 'authors' in df.columns:
            df['commits_per_author'] = df['commits'] / (df['authors'] + 1)
            
        # 4. COMPLEXITY PROXIES
        if 'lines_added' in df.columns and 'churn' in df.columns:
            df['code_stability'] = df['churn'] / (df['lines_added'] + 1)
            
        # 5. POLYNOMIAL FEATURES for key metrics (capturing non-linear relationships)
        if 'churn' in df.columns:
            df['churn_squared'] = df['churn'] ** 2
            df['churn_log'] = np.log1p(df['churn'])
            
        if 'commits' in df.columns:
            df['commits_squared'] = df['commits'] ** 2
            df['commits_log'] = np.log1p(df['commits'])
        
        # 7. DEVELOPER COLLABORATION FEATURES (NEW - High Impact)
        if 'authors' in df.columns and 'commits' in df.columns:
            # Primary author dominance (bus factor indicator)
            # Higher values = risk (knowledge concentrated in few people)
            df['author_concentration'] = 1.0 / (df['authors'] + 1)
            
            # Team size indicator
            df['is_single_author'] = (df['authors'] == 1).astype(int)
            df['is_small_team'] = (df['authors'] <= 2).astype(int)
            
        # 8. COMPLEXITY GROWTH RATE (NEW - High Impact)
        if 'churn' in df.columns and 'days_active' in df.columns:
            # Rate of change - fast churning code is risky
            df['churn_rate'] = df['churn'] / (df['days_active'] + 1)
            
        if 'commits' in df.columns and 'days_active' in df.columns:
            # Commit density - too many commits in short time = rushed work
            df['commit_density'] = df['commits'] / (df['days_active'] + 1)
            
        # 9. CODE REVIEW QUALITY PROXIES (NEW - Medium Impact)
        if 'bug_commits' in df.columns and 'feature_commits' in df.columns:
            # High bug-to-feature ratio indicates poor initial quality
            df['bug_to_feature_ratio'] = df['bug_commits'] / (df['feature_commits'] + 1)
            
        if 'refactor_commits' in df.columns and 'commits' in df.columns:
            # Refactoring rate (healthy code gets refactored)
            df['refactor_rate'] = df['refactor_commits'] / (df['commits'] + 1)
            
        if 'refactor_commits' in df.columns and 'feature_commits' in df.columns:
            # Balance between refactoring and features
            df['refactor_to_feature_ratio'] = df['refactor_commits'] / (df['feature_commits'] + 1)
            
        # 10. ACTIVITY PATTERNS (NEW - Medium Impact)
        if 'lines_added' in df.columns and 'authors' in df.columns:
            # Code per developer (large values = complex changes by few people)
            df['lines_per_author'] = df['lines_added'] / (df['authors'] + 1)
            
        if 'bug_commits' in df.columns and 'authors' in df.columns:
            # Bugs per developer
            df['bugs_per_author'] = df['bug_commits'] / (df['authors'] + 1)
            
        # 11. STABILITY INDICATORS (NEW - Medium Impact)
        if 'lines_deleted' in df.columns and 'lines_added' in df.columns:
            # Deletion rate (high = code being removed/rewritten often)
            df['deletion_rate'] = df['lines_deleted'] / (df['lines_added'] + df['lines_deleted'] + 1)
            
        if 'churn' in df.columns and 'commits' in df.columns:
            # Churn intensity (large changes per commit = risky)
            churn_per_commit = df['churn'] / (df['commits'] + 1)
            df['is_high_churn_commit'] = (churn_per_commit > 100).astype(int)
        
        # 12. INTERACTION TERMS - ADVANCED (NEW - Low to Medium Impact)
        if 'bug_ratio' in df.columns and 'commits' in df.columns:
            # Bug intensity
            df['bug_intensity'] = df['bug_ratio'] * df['commits']
            
        if 'commits' in df.columns and 'days_active' in df.columns:
            # Development pace (commits per day)
            df['development_pace'] = df['commits'] / (df['days_active'] + 1)
            # Logarithmic pace (for better distribution)
            df['development_pace_log'] = np.log1p(df['development_pace'])
        
        # Count new features (compare before/after)
        original_features = ['lines_added', 'churn', 'commits', 'authors', 'bug_commits', 
                            'avg_commit_interval', 'file_age_days', 'window_id', 'lines_deleted',
                            'days_active', 'feature_commits', 'refactor_commits', 'bug_ratio']
        new_feature_count = len([col for col in df.columns if col not in original_features and 
                                not col in ['module', 'needs_maintenance', 'repo_name', 'created_at', '_id']])
        
        self.logger.info(f"✨ Feature engineering complete. Added {new_feature_count} new features")
        
        return df

    def train(self, df: pd.DataFrame, feature_cols: list = None, use_class_weights: bool = False):
        """
        Train regression model on the full incremental dataset
        
        Parameters:
        - df: Training dataframe
        - feature_cols: List of feature columns to use
        - use_class_weights: Deprecated parameter (kept for backward compatibility, ignored)
        """
        df = df.copy()
        
        # Apply feature engineering
        self.logger.info("🔧 Applying feature engineering...")
        df = self.engineer_features(df)
        
        # Exclude non-numeric and metadata columns from training
        exclude_cols = [
            'module', 'needs_maintenance', 'repo_name', 'created_at', '_id', 
            'risk_category', 'filename', 'last_modified', 'label_source',
            # Multi-window metadata (not features)
            'window_id', 'window_start', 'window_end', 'window_size_days',
            'current_sonar_project', 'future_sonar_project',
        ]
        
        # Exclude all SonarQube-derived columns (used for labeling, not features)
        sonarqube_prefixes = [
            'sonarqube_',           # Legacy SonarQube columns
            'current_',             # Current SonarQube metrics (temporal)
            'historical_',          # Historical SonarQube metrics (temporal)
            'quality_degradation',  # The calculated degradation (this is the label!)
            'complexity_delta',     # Delta metrics derived from SonarQube
            'code_smells_delta',
            'bugs_delta',
            'vulnerabilities_delta',
            'technical_debt_delta',
        ]
        
        if feature_cols is None:
            # Auto-select numeric features only
            feature_cols = []
            for c in df.columns:
                if c in exclude_cols:
                    continue
                
                # Exclude SonarQube-derived columns
                is_sonarqube_col = any(c.startswith(prefix) or c == prefix for prefix in sonarqube_prefixes)
                if is_sonarqube_col:
                    self.logger.debug(f"Excluding SonarQube-derived column: {c}")
                    continue
                
                # Only include numeric types
                if pd.api.types.is_numeric_dtype(df[c]):
                    feature_cols.append(c)
                else:
                    self.logger.debug(f"Excluding non-numeric column: {c} (dtype: {df[c].dtype})")
        
        self.logger.info(f"Training with {len(feature_cols)} features")
        self.logger.info(f"Feature sample: {feature_cols[:10]}...")

        X = df[feature_cols]
        y = df['needs_maintenance']

        # Simple train/test split for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Option 1: Hyperparameter tuning (if enabled)
        if self.tune_hyperparameters:
            self.model = self.tune_model_hyperparameters(X_train, y_train)
        else:
            # Option 2: Use optimized default hyperparameters
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=300,        # More trees for better learning
                max_depth=7,             # Deeper trees to capture complex interactions
                learning_rate=0.03,      # Lower learning rate with more estimators
                min_child_weight=2,      # Allow slightly smaller leaf nodes
                subsample=0.85,          # Use 85% of samples per tree
                colsample_bytree=0.85,   # Use 85% of features per tree
                colsample_bylevel=0.85,  # Feature sampling at each tree level
                gamma=0.05,              # Less aggressive pruning
                reg_alpha=0.05,          # L1 regularization (feature selection)
                reg_lambda=1.5,          # L2 regularization (weight smoothing)
                random_state=42,
                tree_method='hist',      # Faster training with histogram-based algorithm
            )
            
            # Train with early stopping to prevent overfitting
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        
        # Log training info
        self.logger.info(f"📊 Training completed with {self.model.n_estimators} trees")
        if hasattr(self.model, 'best_iteration'):
            self.logger.info(f"   Best iteration: {self.model.best_iteration}")
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.logger.info(f"🎯 Top 5 most important features:")
            for idx, row in feature_importance.head(5).iterrows():
                self.logger.info(f"   {row['feature']:25s}: {row['importance']:.4f}")

        # Save model and get path
        model_name = self.save_model()
        
        # Evaluate and return metrics
        metrics = self.evaluate(X_test, y_test, model_name, label_source_filter="sonarqube", training_samples=len(df))
        return metrics


    # def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    #     """Evaluate the trained model"""
    #     y_pred = self.model.predict(X_test)
    #
    #     acc = accuracy_score(y_test, y_pred)
    #     cm = confusion_matrix(y_test, y_pred)
    #     labels = np.unique(np.concatenate([y_test, y_pred]))
    #
    #     logging.info(f"Accuracy: {acc:.4f}")
    #     logging.info(f"Confusion Matrix:\n{cm}")
    #
    #     if cm.shape == (2, 2):
    #         tn, fp, fn, tp = cm.ravel()
    #         self.logger.info(
    #             f"Confusion Matrix Summary → "
    #             f"True Negatives (TN): {tn}, False Positives (FP): {fp}, "
    #             f"False Negatives (FN): {fn}, True Positives (TP): {tp}"
    #         )
    #
    #         self.logger.info(
    #             f"Model detected {tp + tn} correct predictions and "
    #             f"{fp + fn} incorrect ones. "
    #             f"Precision: {tp / (tp + fp + 1e-9):.3f}, "
    #             f"Recall: {tp / (tp + fn + 1e-9):.3f}, "
    #             f"F1: {2 * tp / (2 * tp + fp + fn + 1e-9):.3f}"
    #         )
    #
    #         metrics = {
    #             "accuracy": acc,
    #             "confusion_matrix": cm.tolist(),
    #             "labels": labels.tolist(),
    #             "tn": tn,
    #             "fp": fp,
    #             "fn": fn,
    #             "tp": tp,
    #         }
    #
    #         # Log metrics to MongoDB
    #         log_model_metrics(metrics, model_name=model_name.split("/")[-1].rsplit(".", 1)[0])
    #
    #         return metrics
    #     else:
    #         self.logger.info(
    #             f"⚠️ Evaluation skipped detailed breakdown — only one class ({labels[0]}) "
    #             f"present in test or predicted data."
    #         )
    #         return None

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
