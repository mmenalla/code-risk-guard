#!/usr/bin/env python3
"""
Standalone Risk Prediction Script
==================================

This script performs risk prediction on a local git repository.

1. Fetches commits from a local git repository
2. Generates features from the commit data
3. Predicts risk scores using a trained XGBoost model
4. Saves predictions to a CSV file

Usage:
    python predictmeifyoucan.py --repo-path /path/to/repo --models-dir /path/to/models

Requirements:
    - GitPython
    - pandas
    - xgboost
    - scikit-learn
    - joblib
"""

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Set
import pandas as pd
import numpy as np
import joblib
from git import Repo


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GitCommitCollector:
    """Collects commit data from local git repository (matches degradation model approach)."""
    
    def __init__(self, repo_path: str, branch: str = "main", window_size_days: int = 150):
        self.repo_path = Path(repo_path)
        self.branch = branch
        self.window_size_days = window_size_days
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        try:
            self.repo = Repo(self.repo_path)
            logger.info(f"Initialized git repository: {self.repo_path}")
        except Exception as e:
            raise ValueError(f"Invalid git repository: {e}")
        
        if branch not in self.repo.heads:
            available_branches = [h.name for h in self.repo.heads]
            raise ValueError(f"Branch '{branch}' not found. Available: {available_branches}")
        
        logger.info(f"Using branch: {branch}")
        logger.info(f"Window size: {window_size_days} days")
    
    def _is_source_file(self, filepath: str) -> bool:
        """Check if file is a source code file."""
        source_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.scala',
            '.r', '.m', '.jsx', '.tsx', '.vue', '.sol'
        }
        ext = Path(filepath).suffix.lower()
        return ext in source_extensions
    
    def fetch_commit_data(self, max_commits: int = 10000) -> pd.DataFrame:
        """Fetch commit data and aggregate by file (matches temporal_git_collector approach)."""
        logger.info(f"Fetching commits from {self.repo_path} (branch: {self.branch})")
        logger.info(f"Max commits: {max_commits}")
        logger.info(f"Time window: last {self.window_size_days} days")
        
        since_date = datetime.now() - timedelta(days=self.window_size_days)
        commits = list(self.repo.iter_commits(
            self.branch, 
            max_count=max_commits,
            since=since_date
        ))
        logger.info(f"Found {len(commits)} commits in time window")
        
        file_stats: Dict[str, Dict] = {}
        
        for commit in commits:
            commit_date = commit.committed_datetime
            if commit_date.tzinfo is not None:
                commit_date = commit_date.astimezone(timezone.utc).replace(tzinfo=None)
            
            author = commit.author.email
            message = commit.message.lower()
            is_bug_fix = any(kw in message for kw in ['fix', 'bug', 'patch', 'hotfix', 'bugfix'])
            is_feature = any(kw in message for kw in ['feat', 'feature', 'add', 'implement'])
            is_refactor = any(kw in message for kw in ['refactor', 'clean', 'improve'])
            
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
                
                for diff in diffs:
                    filepath = diff.b_path or diff.a_path
                    if not filepath or not self._is_source_file(filepath):
                        continue
                    
                    if filepath not in file_stats:
                        file_stats[filepath] = {
                            'lines_added': 0,
                            'lines_deleted': 0,
                            'commits': 0,
                            'authors': set(),
                            'bug_commits': 0,
                            'feature_commits': 0,
                            'refactor_commits': 0,
                            'first_commit': commit_date,
                            'last_commit': commit_date
                        }
                    
                    stats = file_stats[filepath]
                    
                    if diff.diff:
                        diff_text = diff.diff.decode('utf-8', errors='ignore')
                        lines_added = sum(1 for line in diff_text.split('\n') 
                                        if line.startswith('+') and not line.startswith('+++'))
                        lines_deleted = sum(1 for line in diff_text.split('\n') 
                                          if line.startswith('-') and not line.startswith('---'))
                        stats['lines_added'] += lines_added
                        stats['lines_deleted'] += lines_deleted
                    
                    stats['commits'] += 1
                    stats['authors'].add(author)
                    if is_bug_fix:
                        stats['bug_commits'] += 1
                    if is_feature:
                        stats['feature_commits'] += 1
                    if is_refactor:
                        stats['refactor_commits'] += 1
                    
                    stats['first_commit'] = min(stats['first_commit'], commit_date)
                    stats['last_commit'] = max(stats['last_commit'], commit_date)
        
        if not file_stats:
            logger.warning("No source files found in commits")
            return pd.DataFrame()
        
        records = []
        repo_name = self.repo_path.name
        
        for filepath, stats in file_stats.items():
            days_active = max((stats['last_commit'] - stats['first_commit']).days, 1)
            num_authors = len(stats['authors'])
            num_commits = stats['commits']
            
            # Calculate base features (matching git_commit_client.py exactly)
            # Base features: 13 total
            # commits, authors, lines_added, lines_deleted, churn
            # bug_commits, refactor_commits, feature_commits
            # lines_per_author, churn_per_commit, bug_ratio, days_active, commits_per_day
            records.append({
                'module': filepath,
                'commits': num_commits,
                'authors': num_authors,
                'lines_added': stats['lines_added'],
                'lines_deleted': stats['lines_deleted'],
                'churn': stats['lines_added'] + stats['lines_deleted'],
                'bug_commits': stats['bug_commits'],
                'refactor_commits': stats['refactor_commits'],
                'feature_commits': stats['feature_commits'],
                'lines_per_author': stats['lines_added'] / num_authors if num_authors > 0 else 0,
                'churn_per_commit': (stats['lines_added'] + stats['lines_deleted']) / num_commits if num_commits > 0 else 0,
                'bug_ratio': stats['bug_commits'] / num_commits if num_commits > 0 else 0,
                'days_active': days_active,
                'commits_per_day': num_commits / days_active
            })
        
        df = pd.DataFrame(records)
        logger.info(f"Fetched data for {len(df)} files from {len(commits)} commits")
        
        return df


class FeatureEngineer:
    """Transforms raw commit data into ML features (matches train.py - 12 engineered features)."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the 12 engineered features used by the model.
        
        Note: Base features (lines_per_author, churn_per_commit, bug_ratio, commits_per_day)
        are already calculated in fetch_commit_data() to match training exactly.
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
        
        # 13. degradation_days - Same as days_active (temporal window for degradation)
        # This feature was added to match the model's expected features
        if 'days_active' in df.columns:
            df['degradation_days'] = df['days_active']
        
        logger.info(f"✨ Feature engineering complete. Total features: {len(df.columns)}")
        return df


class StandaloneRiskPredictor:
    """Standalone risk predictor that doesn't rely on Config or external services."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"✅ Model loaded successfully")
        
        self.feature_engineer = FeatureEngineer()
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating features for {len(df)} file records...")
        df_features = self.feature_engineer.transform(df)
        
        metadata = df_features[['module']].copy()
        
        # Only drop metadata columns, keep all numeric features (base + engineered)
        drop_cols = [
            'module', 'repo_name', 'created_at', 'filename', 'label_source', 
            'risk_category', 'feedback_count', 'last_feedback_at', 'last_modified',
            'file_age_days', 'avg_commit_interval'  # Not used in model
        ]
        X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns], errors='ignore')
        
        expected_features = self.model.get_booster().feature_names
        if expected_features:
            logger.info(f"Model expects {len(expected_features)} features")
            logger.info(f"We have {len(X.columns)} features")
            
            # Check for missing features
            missing_features = set(expected_features) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                logger.info("Creating missing features with default values...")
                for feat in missing_features:
                    X[feat] = 0
            
            # Check for extra features
            extra_features = set(X.columns) - set(expected_features)
            if extra_features:
                logger.info(f"Extra features (will be dropped): {extra_features}")
            
            X = X[expected_features]
        
        logger.info(f"Using {len(X.columns)} features for prediction")
        
        logger.info(f"Running inference...")
        raw_predictions = self.model.predict(X)
        
        # Apply prediction calibration
        # Training data statistics: mean=-0.011, std=0.082, range=[-0.531, 0.566]
        # Raw predictions are often out of this range due to feature distribution mismatch
        degradation_score = self._calibrate_predictions(raw_predictions)
        
        result = metadata.copy()
        result['degradation_score'] = degradation_score
        result['raw_prediction'] = raw_predictions  # Keep raw for debugging
        
        # Categorize based on degradation score
        # Training data range: -0.53 to +0.57, avg: -0.01, stdDev: 0.082
        # Bins based on actual training distribution:
        # < 0: improved, 0-0.1: stable, 0.1-0.2: degraded, > 0.2: severely degraded
        result['risk_category'] = pd.cut(
            degradation_score,
            bins=[-float('inf'), 0, 0.1, 0.2, float('inf')],
            labels=['improved', 'stable', 'degraded', 'severely-degraded'],
            include_lowest=True
        )
        
        logger.info(f"✅ Predictions complete")
        logger.info(f"   Raw predictions - Mean: {raw_predictions.mean():.3f}, Range: [{raw_predictions.min():.3f}, {raw_predictions.max():.3f}]")
        logger.info(f"   Calibrated predictions - Mean: {degradation_score.mean():.3f}, Range: [{degradation_score.min():.3f}, {degradation_score.max():.3f}]")
        logger.info(f"   Std dev: {degradation_score.std():.3f}")
        
        return result
    
    def _calibrate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions to match training data distribution.
        
        Training data statistics (from MongoDB):
        - Mean: -0.011
        - Std: 0.082
        - Range: [-0.531, 0.566]
        
        Strategy: Linear scaling + shift to map raw predictions to expected range
        """
        import numpy as np
        
        # Training data statistics
        TRAIN_MEAN = -0.011
        TRAIN_STD = 0.082
        TRAIN_MIN = -0.531
        TRAIN_MAX = 0.566
        
        # Calculate raw prediction statistics
        raw_mean = predictions.mean()
        raw_std = predictions.std()
        
        # Method 1: Z-score normalization then scale to training distribution
        # This preserves relative differences while matching the target distribution
        if raw_std > 0:
            # Standardize (z-score)
            z_scores = (predictions - raw_mean) / raw_std
            
            # Scale to training distribution
            calibrated = z_scores * TRAIN_STD + TRAIN_MEAN
            
            # Clip to training data range (with small buffer for unseen cases)
            calibrated = np.clip(calibrated, TRAIN_MIN - 0.1, TRAIN_MAX + 0.1)
        else:
            # All predictions the same - just shift to training mean
            calibrated = np.full_like(predictions, TRAIN_MEAN)
        
        logger.info(f"   📊 Calibration: Shifted mean from {raw_mean:.3f} to {calibrated.mean():.3f}")
        
        return calibrated


def fetch_commits(repo_path: str, branch: str = "main", max_commits: int = 10000, window_size_days: int = 150) -> pd.DataFrame:
    logger.info(f"🔄 Fetching commits from repository: {repo_path}")
    logger.info(f"   Branch: {branch}")
    logger.info(f"   Max commits: {max_commits}")
    logger.info(f"   Window size: {window_size_days} days")
    
    collector = GitCommitCollector(repo_path=repo_path, branch=branch, window_size_days=window_size_days)
    df = collector.fetch_commit_data(max_commits=max_commits)
    
    if df.empty:
        logger.warning("⚠️  No commits fetched from repository")
        return df
    
    logger.info(f"✅ Fetched data for {len(df)} files from {df['commits'].sum():.0f} commits")
    logger.info(f"   Files: {len(df)}, Total commits analyzed: {df['commits'].sum():.0f}")
    
    return df


def save_predictions(predictions_df: pd.DataFrame, output_path: str):
    timestamp = int(datetime.now().timestamp())
    
    output_path = Path(output_path)
    if output_path.is_dir() or not output_path.suffix:
        output_dir = output_path
    else:
        output_dir = output_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamped_filename = f"predictions_{timestamp}.csv"
    final_output_path = output_dir / timestamped_filename
    
    # Include raw_prediction if available for debugging
    cols_to_save = ['module', 'degradation_score', 'risk_category']
    if 'raw_prediction' in predictions_df.columns:
        cols_to_save.append('raw_prediction')
    
    output_df = predictions_df[cols_to_save].copy()
    output_df = output_df.sort_values('degradation_score', ascending=False)
    
    output_df.to_csv(final_output_path, index=False)
    logger.info(f"✅ Predictions saved to {final_output_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DEGRADATION PREDICTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files analyzed: {len(output_df)}")
    logger.info(f"\nDegradation Distribution:")
    risk_counts = output_df['risk_category'].value_counts().sort_index()
    for category, count in risk_counts.items():
        pct = count / len(output_df) * 100
        logger.info(f"  {category:20s}: {count:5d} ({pct:5.1f}%)")
    
    logger.info(f"\nTop 10 Most Degraded Files:")
    logger.info(f"{'-'*60}")
    for idx, row in output_df.head(10).iterrows():
        logger.info(f"  {row['degradation_score']:+.3f} - {row['risk_category']:20s} - {row['module']}")
    
    logger.info(f"\nTop 10 Most Improved Files:")
    logger.info(f"{'-'*60}")
    for idx, row in output_df.tail(10).iloc[::-1].iterrows():
        logger.info(f"  {row['degradation_score']:+.3f} - {row['risk_category']:20s} - {row['module']}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone risk prediction for git repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with v10 degradation model
  python predictmeifyoucan.py --repo-path /path/to/repo --model-path ../dags/src/models/artifacts/xgboost_degradation_model_v10.pkl
  
  # Specify branch and time window
  python predictmeifyoucan.py --repo-path /path/to/repo --model-path ../dags/src/models/artifacts/xgboost_degradation_model_v10.pkl --branch develop --window-size-days 180
  
  # Custom output directory (filename will be predictions_<timestamp>.csv)
  python predictmeifyoucan.py --repo-path /path/to/repo --model-path ../dags/src/models/artifacts/xgboost_degradation_model_v10.pkl --output results/
        """
    )
    
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to local git repository to analyze"
    )
    
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model file (.pkl)"
    )
    
    parser.add_argument(
        "--branch",
        default="main",
        help="Git branch to analyze (default: main)"
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=10000,
        help="Maximum number of commits to analyze (default: 10000)"
    )
    parser.add_argument(
        "--window-size-days",
        type=int,
        default=150,
        help="Time window in days for commit analysis (default: 150)"
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for CSV file (default: current directory). Filename will be predictions_<timestamp>.csv"
    )
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"STANDALONE RISK PREDICTION")
    logger.info(f"{'='*60}\n")
    
    try:
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            sys.exit(1)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP 1: FETCH COMMIT DATA")
        logger.info(f"{'='*60}")
        commits_df = fetch_commits(
            repo_path=args.repo_path,
            branch=args.branch,
            max_commits=args.max_commits,
            window_size_days=args.window_size_days
        )
        
        if commits_df.empty:
            logger.error("No commit data fetched. Exiting.")
            sys.exit(1)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP 2: PREDICT RISK SCORES")
        logger.info(f"{'='*60}")
        predictor = StandaloneRiskPredictor(model_path=str(model_path))
        predictions_df = predictor.predict(commits_df)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP 3: SAVE RESULTS")
        logger.info(f"{'='*60}")
        save_predictions(predictions_df, args.output)
        
        logger.info(f"✅ Risk prediction complete!")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
