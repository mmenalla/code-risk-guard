"""
Temporal SonarQube Client for Degradation Analysis

This module fetches SonarQube metrics from two time points:
1. Current state (today's metrics)
2. Historical state (100 days ago metrics from -historical projects)

Used to calculate quality degradation: score_today - score_100_days_ago
"""

import logging
from typing import Dict, Optional, List, Tuple
from src.data.sonarqube_client import SonarQubeClient

logger = logging.getLogger(__name__)


class TemporalSonarQubeClient:
    """
    Fetches SonarQube metrics from current and historical project states
    to enable temporal degradation analysis.
    """
    
    def __init__(self, sonarqube_url: str, sonarqube_token: str):
        """
        Initialize temporal SonarQube client
        
        Args:
            sonarqube_url: SonarQube server URL
            sonarqube_token: Authentication token
        """
        self.client = SonarQubeClient(sonarqube_url, sonarqube_token)
        self.sonarqube_url = sonarqube_url
        self.sonarqube_token = sonarqube_token
    
    def get_temporal_metrics(
        self, 
        current_project_key: str, 
        historical_project_key: str, 
        file_path: str
    ) -> Optional[Dict[str, float]]:
        """
        Get metrics from both current and historical states for a file.
        
        Args:
            current_project_key: SonarQube project key for current state (e.g., "numpy")
            historical_project_key: SonarQube project key for historical state (e.g., "numpy-historical")
            file_path: Relative path to the file
            
        Returns:
            Dictionary with current metrics, historical metrics, and degradation scores
            None if data is unavailable for either time point
        """
        # Fetch current metrics
        current_metrics = self.client.get_file_measures(current_project_key, file_path)
        if not current_metrics:
            logger.debug(f"No current metrics for {file_path} in {current_project_key}")
            return None
        
        # Fetch historical metrics
        historical_metrics = self.client.get_file_measures(historical_project_key, file_path)
        if not historical_metrics:
            logger.debug(f"No historical metrics for {file_path} in {historical_project_key}")
            return None
        
        # Calculate maintainability scores
        current_score = self.client.calculate_maintainability_score(current_metrics)
        historical_score = self.client.calculate_maintainability_score(historical_metrics)
        
        # Calculate degradation (positive = got worse, negative = got better)
        degradation = current_score - historical_score
        
        return {
            # Current state
            'current_complexity': current_metrics.get('complexity', 0),
            'current_code_smells': current_metrics.get('code_smells', 0),
            'current_bugs': current_metrics.get('bugs', 0),
            'current_vulnerabilities': current_metrics.get('vulnerabilities', 0),
            'current_technical_debt': current_metrics.get('sqale_index', 0),
            'current_maintainability_score': current_score,
            
            # Historical state
            'historical_complexity': historical_metrics.get('complexity', 0),
            'historical_code_smells': historical_metrics.get('code_smells', 0),
            'historical_bugs': historical_metrics.get('bugs', 0),
            'historical_vulnerabilities': historical_metrics.get('vulnerabilities', 0),
            'historical_technical_debt': historical_metrics.get('sqale_index', 0),
            'historical_maintainability_score': historical_score,
            
            # Degradation metrics (delta)
            'complexity_delta': current_metrics.get('complexity', 0) - historical_metrics.get('complexity', 0),
            'code_smells_delta': current_metrics.get('code_smells', 0) - historical_metrics.get('code_smells', 0),
            'bugs_delta': current_metrics.get('bugs', 0) - historical_metrics.get('bugs', 0),
            'vulnerabilities_delta': current_metrics.get('vulnerabilities', 0) - historical_metrics.get('vulnerabilities', 0),
            'technical_debt_delta': current_metrics.get('sqale_index', 0) - historical_metrics.get('sqale_index', 0),
            'quality_degradation': degradation,  # This is the main label
        }
    
    def get_temporal_metrics_batch(
        self,
        current_project_key: str,
        historical_project_key: str,
        file_paths: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get temporal metrics for multiple files in batch.
        
        Args:
            current_project_key: SonarQube project key for current state
            historical_project_key: SonarQube project key for historical state
            file_paths: List of file paths to analyze
            
        Returns:
            Dictionary mapping file_path -> temporal_metrics
        """
        results = {}
        
        logger.info(f"Fetching temporal metrics for {len(file_paths)} files")
        logger.info(f"  Current project: {current_project_key}")
        logger.info(f"  Historical project: {historical_project_key}")
        
        success_count = 0
        for file_path in file_paths:
            metrics = self.get_temporal_metrics(
                current_project_key, 
                historical_project_key, 
                file_path
            )
            if metrics:
                results[file_path] = metrics
                success_count += 1
        
        logger.info(f"✅ Successfully fetched temporal metrics for {success_count}/{len(file_paths)} files")
        
        return results
    
    def verify_historical_projects_exist(self, project_keys: List[str]) -> Tuple[List[str], List[str]]:
        """
        Verify that historical projects exist for the given current projects.
        
        Args:
            project_keys: List of current project keys (e.g., ["numpy", "flask"])
            
        Returns:
            Tuple of (existing_projects, missing_projects)
        """
        existing = []
        missing = []
        
        for project_key in project_keys:
            historical_key = f"{project_key}-historical"
            
            # Check if historical project exists
            try:
                response = self.client.session.get(
                    f"{self.sonarqube_url}/api/projects/search",
                    params={"projects": historical_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('components'):
                        existing.append(project_key)
                        logger.info(f"✓ Historical project exists: {historical_key}")
                    else:
                        missing.append(project_key)
                        logger.warning(f"✗ Historical project missing: {historical_key}")
                else:
                    missing.append(project_key)
                    logger.warning(f"✗ Failed to check: {historical_key}")
            except Exception as e:
                logger.error(f"Error checking {historical_key}: {e}")
                missing.append(project_key)
        
        return existing, missing
