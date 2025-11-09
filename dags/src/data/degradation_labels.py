"""
Degradation Label Creator for Temporal Risk Prediction

Creates labels based on quality degradation over time:
label = sonarqube_score_today - sonarqube_score_100_days_ago

Positive values = Quality degraded (increased maintenance risk)
Negative values = Quality improved (decreased maintenance risk)
Zero = No change

This approach enables the model to predict FUTURE quality degradation
based on recent Git activity patterns, which is more actionable than
predicting absolute quality.
"""

import pandas as pd
import logging
from typing import Dict, Optional, List
from datetime import datetime

from src.data.sonarqube_client import SonarQubeClient
from src.utils.config import Config

logger = logging.getLogger(__name__)


class DegradationLabelCreator:
    """
    Creates degradation labels by comparing current vs historical SonarQube metrics.
    
    Label interpretation:
    - degradation > 0.2: High risk (quality significantly degraded)
    - 0.1 < degradation <= 0.2: Medium risk (quality moderately degraded)
    - 0 < degradation <= 0.1: Low risk (quality slightly degraded)
    - degradation <= 0: No risk (quality stable or improved)
    """
    
    def __init__(
        self,
        sonarqube_url: str = None,
        sonarqube_token: str = None
    ):
        """
        Initialize degradation label creator.
        
        Args:
            sonarqube_url: SonarQube server URL (defaults to Config)
            sonarqube_token: SonarQube auth token (defaults to Config)
        """
        self.sonarqube_url = sonarqube_url or Config.SONARQUBE_URL
        self.sonarqube_token = sonarqube_token or Config.SONARQUBE_TOKEN
        
        if not self.sonarqube_url or not self.sonarqube_token:
            raise ValueError("SonarQube URL and token must be configured")
        
        self.temporal_client = SonarQubeClient(
            self.sonarqube_url,
            self.sonarqube_token
        )
    
    def create_degradation_labels(
        self,
        df: pd.DataFrame,
        repo_name: str,
        current_project_key: str,
        historical_project_key: str = None
    ) -> pd.DataFrame:
        """
        Create degradation labels for a DataFrame of files.
        
        Args:
            df: DataFrame with 'module' column (file paths) and Git features
            repo_name: Repository name
            current_project_key: SonarQube project key for current state
            historical_project_key: SonarQube project key for historical state
                                    (defaults to f"{current_project_key}-historical")
        
        Returns:
            DataFrame with degradation labels and temporal SonarQube metrics
        """
        df = df.copy()
        
        if historical_project_key is None:
            historical_project_key = f"{current_project_key}-historical"
        
        logger.info(f"Creating degradation labels for {len(df)} files in {repo_name}")
        logger.info(f"  Current project: {current_project_key}")
        logger.info(f"  Historical project: {historical_project_key}")
        
        # Get list of files
        file_paths = df['module'].tolist()
        
        # Fetch temporal metrics for all files
        temporal_metrics = self.temporal_client.get_temporal_metrics_batch(
            current_project_key,
            historical_project_key,
            file_paths
        )
        
        # Add temporal metrics to DataFrame
        labeled_count = 0
        for idx, row in df.iterrows():
            file_path = row['module']
            
            if file_path in temporal_metrics:
                metrics = temporal_metrics[file_path]
                
                # Add all temporal metrics as columns
                for metric_name, metric_value in metrics.items():
                    df.at[idx, metric_name] = metric_value
                
                # Set the main label
                df.at[idx, 'needs_maintenance'] = metrics['quality_degradation']
                df.at[idx, 'label_source'] = 'sonarqube_degradation'
                
                labeled_count += 1
        
        # Create risk categories based on degradation
        df['risk_category'] = pd.cut(
            df['needs_maintenance'],
            bins=[-float('inf'), 0, 0.1, 0.2, float('inf')],
            labels=['no-risk', 'low-risk', 'medium-risk', 'high-risk']
        )
        
        # Log statistics
        logger.info(f"✅ Successfully labeled {labeled_count}/{len(df)} files with degradation metrics")
        
        if labeled_count > 0:
            degradation_stats = df['needs_maintenance'].describe()
            logger.info(f"📊 Degradation statistics:")
            logger.info(f"   Mean: {degradation_stats['mean']:.3f}")
            logger.info(f"   Std: {degradation_stats['std']:.3f}")
            logger.info(f"   Min: {degradation_stats['min']:.3f}")
            logger.info(f"   Max: {degradation_stats['max']:.3f}")
            
            risk_dist = df['risk_category'].value_counts().to_dict()
            logger.info(f"📊 Risk distribution: {risk_dist}")
        
        return df
    
    def verify_historical_projects(
        self,
        project_keys: List[str]
    ) -> Dict[str, bool]:
        """
        Verify that historical SonarQube projects exist.
        
        Args:
            project_keys: List of current project keys (e.g., ["numpy", "flask"])
            
        Returns:
            Dictionary mapping project_key -> exists (bool)
        """
        logger.info(f"Verifying historical projects for {len(project_keys)} repositories")
        
        existing, missing = self.temporal_client.verify_historical_projects_exist(project_keys)
        
        result = {}
        for key in project_keys:
            result[key] = key in existing
        
        if missing:
            logger.warning(f"❌ Missing historical projects: {missing}")
            logger.warning("Run TDGPTRepos/scan_historical_sonarqube.sh to create them")
        
        return result
    
    @staticmethod
    def interpret_degradation_score(score: float) -> str:
        """
        Human-readable interpretation of degradation score.
        
        Args:
            score: Degradation score (current - historical)
            
        Returns:
            Interpretation string
        """
        if score > 0.2:
            return "HIGH RISK: Quality significantly degraded"
        elif score > 0.1:
            return "MEDIUM RISK: Quality moderately degraded"
        elif score > 0:
            return "LOW RISK: Quality slightly degraded"
        elif score > -0.1:
            return "STABLE: Quality unchanged"
        else:
            return "IMPROVED: Quality got better"
    
    @staticmethod
    def create_degradation_report(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Create a report of files with highest degradation.
        
        Args:
            df: DataFrame with degradation labels
            top_n: Number of top degraded files to return
            
        Returns:
            DataFrame with top degraded files and their metrics
        """
        if 'needs_maintenance' not in df.columns:
            logger.error("DataFrame missing 'needs_maintenance' column")
            return pd.DataFrame()
        
        # Filter to only files with degradation data
        degraded_df = df[df['label_source'] == 'sonarqube_degradation'].copy()
        
        if degraded_df.empty:
            logger.warning("No files with degradation labels found")
            return pd.DataFrame()
        
        # Sort by degradation (highest first)
        degraded_df = degraded_df.sort_values('needs_maintenance', ascending=False)
        
        # Select relevant columns for report
        report_cols = [
            'module',
            'repo_name',
            'needs_maintenance',
            'risk_category',
            'current_maintainability_score',
            'historical_maintainability_score',
            'complexity_delta',
            'code_smells_delta',
            'bugs_delta',
            'commits',
            'authors',
            'churn'
        ]
        
        available_cols = [col for col in report_cols if col in degraded_df.columns]
        report_df = degraded_df[available_cols].head(top_n)
        
        logger.info(f"📊 Top {top_n} degraded files:")
        for idx, row in report_df.iterrows():
            logger.info(f"   {row['module']}: degradation={row['needs_maintenance']:.3f} ({row['risk_category']})")
        
        return report_df
