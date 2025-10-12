import os
import re
import joblib
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from src.utils.config import Config


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

    def train_or_finetune(self, df: pd.DataFrame, feature_cols: list = None):
        """Fine-tune existing model if available, otherwise train new"""
        df = df.copy()
        feature_cols = feature_cols or [c for c in df.columns if c not in ['module', 'needs_maintenance']]
        X = df[feature_cols]
        y = df['needs_maintenance']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.load_existing_model()

        if self.model is None:
            # Train from scratch
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train, y_train)

        else:
            # Fine-tune from existing Booster
            self.logger.info("Fine-tuning existing model...")

            booster = self.model.get_booster()
            dtrain = xgb.DMatrix(X_train, label=y_train)

            params = self.model.get_xgb_params()

            # Continue training for a few boosting rounds
            updated_booster = xgb.train(
                params=params,
                dtrain=dtrain,
                xgb_model=booster,
                num_boost_round=20,
            )

            # Replace the model’s internal booster
            self.model._Booster = updated_booster

        model_name = self.save_model()
        return self.model, model_name, X_test, y_test

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate the trained model"""
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        labels = np.unique(np.concatenate([y_test, y_pred]))

        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            self.logger.info(
                f"Confusion Matrix Summary → "
                f"True Negatives (TN): {tn}, False Positives (FP): {fp}, "
                f"False Negatives (FN): {fn}, True Positives (TP): {tp}"
            )

            self.logger.info(
                f"Model detected {tp + tn} correct predictions and "
                f"{fp + fn} incorrect ones. "
                f"Precision: {tp / (tp + fp + 1e-9):.3f}, "
                f"Recall: {tp / (tp + fn + 1e-9):.3f}, "
                f"F1: {2 * tp / (2 * tp + fp + fn + 1e-9):.3f}"
            )
        else:
            self.logger.info(
                f"⚠️ Evaluation skipped detailed breakdown — only one class ({labels[0]}) "
                f"present in test or predicted data."
            )

        return {
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "labels": labels.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

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
