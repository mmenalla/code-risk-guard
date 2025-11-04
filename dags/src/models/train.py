import os
import re
import joblib
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.config import Config
from src.data.save_incremental_labeled_data import log_model_metrics


class RiskModelTrainer:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.MODELS_DIR / "xgboost_risk_model.pkl"
        self.model = None
        self.logger = logging.getLogger(__name__)

    def load_existing_model(self):
        """Load an existing model if available for fine-tuning"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Loaded existing model from {self.model_path} for fine-tuning")
        else:
            self.logger.info("No existing model found. Training from scratch.")

    def train(self, df: pd.DataFrame, feature_cols: list = None, use_class_weights: bool = False):
        """
        Train regression model on the full incremental dataset
        
        Parameters:
        - df: Training dataframe
        - feature_cols: List of feature columns to use
        - use_class_weights: Deprecated parameter (kept for backward compatibility, ignored)
        """
        df = df.copy()
        feature_cols = feature_cols or [
            c for c in df.columns
            if c not in ['module', 'needs_maintenance', 'repo_name', 'created_at', '_id', 'risk_category', 'filename']
        ]

        X = df[feature_cols]
        y = df['needs_maintenance']

        # Simple train/test split for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


        # Enhanced hyperparameters for regression
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            n_estimators=200,        # Increased from 100 (more trees = better learning)
            max_depth=6,             # Increased from 4 (capture more complex patterns)
            learning_rate=0.05,      # Decreased from 0.1 (more conservative, better generalization)
            min_child_weight=3,      # Regularization to prevent overfitting
            subsample=0.8,           # Use 80% of samples per tree (reduces overfitting)
            colsample_bytree=0.8,    # Use 80% of features per tree (feature diversity)
            gamma=0.1,               # Minimum loss reduction for split (regularization)
            reg_alpha=0.1,           # L1 regularization
            reg_lambda=1.0,          # L2 regularization
            random_state=42
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

        model_name = self.save_model()
        return self.model, model_name, X_test, y_test


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

    def evaluate(self, X_test, y_test, model_name: str, label_source_filter: str = "all", training_samples: int = 0):
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.logger.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

        metrics = {
            "mae": mae,
            "mse": mse,
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
