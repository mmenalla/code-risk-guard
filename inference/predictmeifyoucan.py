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
import joblib
from git import Repo


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GitCommitCollector:
    """Collects commit data from local git repository."""
    
    def __init__(self, repo_path: str, branch: str = "main"):
        self.repo_path = Path(repo_path)
        self.branch = branch
        
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
    
    def _is_source_file(self, filepath: str) -> bool:
        """Check if file is a source code file."""
        source_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.scala',
            '.r', '.m', '.jsx', '.tsx', '.vue', '.sol'
        }
        ext = Path(filepath).suffix.lower()
        return ext in source_extensions
    
    def fetch_commit_data(self, max_commits: int = 300) -> pd.DataFrame:
        """Fetch commit data and aggregate by file."""
        logger.info(f"Fetching commits from {self.repo_path} (branch: {self.branch})")
        logger.info(f"Max commits: {max_commits}")
        
        commits = list(self.repo.iter_commits(self.branch, max_count=max_commits))
        logger.info(f"Found {len(commits)} commits")
        
        file_stats: Dict[str, Dict] = {}
        
        for commit in commits:
            commit_date = commit.committed_datetime
            if commit_date.tzinfo is not None:
                commit_date = commit_date.astimezone(timezone.utc).replace(tzinfo=None)
            
            author = commit.author.email
            message = commit.message.lower()
            is_bug_fix = any(kw in message for kw in ['fix', 'bug', 'patch', 'hotfix', 'bugfix'])
            
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
                            'lines_removed': 0,
                            'commits': 0,
                            'authors': set(),
                            'bug_commits': 0,
                            'first_commit': commit_date,
                            'last_commit': commit_date
                        }
                    
                    stats = file_stats[filepath]
                    
                    if diff.diff:
                        diff_text = diff.diff.decode('utf-8', errors='ignore')
                        lines_added = sum(1 for line in diff_text.split('\n') 
                                        if line.startswith('+') and not line.startswith('+++'))
                        lines_removed = sum(1 for line in diff_text.split('\n') 
                                          if line.startswith('-') and not line.startswith('---'))
                        stats['lines_added'] += lines_added
                        stats['lines_removed'] += lines_removed
                    
                    stats['commits'] += 1
                    stats['authors'].add(author)
                    if is_bug_fix:
                        stats['bug_commits'] += 1
                    
                    stats['first_commit'] = min(stats['first_commit'], commit_date)
                    stats['last_commit'] = max(stats['last_commit'], commit_date)
        
        if not file_stats:
            logger.warning("No source files found in commits")
            return pd.DataFrame()
        
        records = []
        repo_name = self.repo_path.name
        
        for filepath, stats in file_stats.items():
            records.append({
                'module': filepath,
                'filename': filepath,
                'repo_name': repo_name,
                'lines_added': stats['lines_added'],
                'lines_removed': stats['lines_removed'],
                'prs': stats['commits'],
                'unique_authors': len(stats['authors']),
                'bug_prs': stats['bug_commits'],
                'churn': stats['lines_added'] + stats['lines_removed'],
                'created_at': stats['first_commit'],
                'last_modified': stats['last_commit']
            })
        
        df = pd.DataFrame(records)
        logger.info(f"Fetched data for {len(df)} files from {len(commits)} commits")
        
        return df


class FeatureEngineer:
    """Transforms raw commit data into ML features."""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ML features from commit data."""
        df = df.copy()
        
        df['prs'] = df['prs'].replace(0, 1)
        df['unique_authors'] = df['unique_authors'].replace(0, 1)
        
        total_lines = df['lines_added'] + df['lines_removed']
        
        df['lines_per_pr'] = total_lines / df['prs']
        df['lines_per_author'] = total_lines / df['unique_authors']
        df['add_del_ratio'] = df['lines_added'] / df['lines_removed'].replace(0, 1)
        df['deletion_ratio'] = df['lines_removed'] / total_lines.replace(0, 1)
        df['bug_density'] = df['bug_prs'] / total_lines.replace(0, 1)
        df['collaboration_complexity'] = df['unique_authors'] * (df['churn'] / df['prs'])
        
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
        
        drop_cols = [
            'module', 'repo_name', 'created_at', 'filename', 'label_source', 
            'risk_category', 'feedback_count', 'last_feedback_at'
        ]
        X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns], errors='ignore')
        
        expected_features = self.model.get_booster().feature_names
        if expected_features:
            X = X[expected_features]
        
        logger.info(f"Using {len(X.columns)} features for prediction")
        
        logger.info(f"Running inference...")
        risk_score = self.model.predict(X)
        
        result = metadata.copy()
        result['risk_score'] = risk_score
        
        result['risk_category'] = pd.cut(
            risk_score,
            bins=[0, 0.22, 0.47, 0.65, 1.0],
            labels=['no-risk', 'low-risk', 'medium-risk', 'high-risk'],
            include_lowest=True
        )
        
        logger.info(f"✅ Predictions complete")
        logger.info(f"   Mean risk score: {risk_score.mean():.3f}")
        logger.info(f"   Std dev: {risk_score.std():.3f}")
        logger.info(f"   Min: {risk_score.min():.3f}, Max: {risk_score.max():.3f}")
        
        return result


def fetch_commits(repo_path: str, branch: str = "main", max_commits: int = 300) -> pd.DataFrame:
    logger.info(f"🔄 Fetching commits from repository: {repo_path}")
    logger.info(f"   Branch: {branch}")
    logger.info(f"   Max commits: {max_commits}")
    
    collector = GitCommitCollector(repo_path=repo_path, branch=branch)
    df = collector.fetch_commit_data(max_commits=max_commits)
    
    if df.empty:
        logger.warning("⚠️  No commits fetched from repository")
        return df
    
    logger.info(f"✅ Fetched data for {len(df)} files from {df['prs'].sum():.0f} file-commit pairs")
    logger.info(f"   Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
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
    
    output_df = predictions_df[['module', 'risk_score', 'risk_category']].copy()
    output_df = output_df.sort_values('risk_score', ascending=False)
    
    output_df.to_csv(final_output_path, index=False)
    logger.info(f"✅ Predictions saved to {final_output_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PREDICTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files analyzed: {len(output_df)}")
    logger.info(f"\nRisk Distribution:")
    risk_counts = output_df['risk_category'].value_counts().sort_index()
    for category, count in risk_counts.items():
        pct = count / len(output_df) * 100
        logger.info(f"  {category:15s}: {count:5d} ({pct:5.1f}%)")
    
    logger.info(f"\nTop 10 Highest Risk Files:")
    logger.info(f"{'-'*60}")
    for idx, row in output_df.head(10).iterrows():
        logger.info(f"  {row['risk_score']:.3f} - {row['risk_category']:12s} - {row['module']}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone risk prediction for git repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python predictmeifyoucan.py --repo-path /path/to/repo --model-path ../dags/src/models/artifacts/xgboost_risk_model_v23.pkl
  
  # Specify branch and max commits
  python predictmeifyoucan.py --repo-path /path/to/repo --model-path ../dags/src/models/artifacts/xgboost_risk_model_v23.pkl --branch develop --max-commits 500
  
  # Custom output directory (filename will be predictions_<timestamp>.csv)
  python predictmeifyoucan.py --repo-path /path/to/repo --model-path ../dags/src/models/artifacts/xgboost_risk_model_v23.pkl --output results/
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
        default=300,
        help="Maximum number of commits to analyze (default: 300)"
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
            max_commits=args.max_commits
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
