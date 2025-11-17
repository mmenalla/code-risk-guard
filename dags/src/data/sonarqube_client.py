"""
SonarQube API Client for fetching code quality metrics
"""
import logging
import requests
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SonarQubeClient:
    @staticmethod
    def debug_list_projects_and_scores():
        """
        Helper to print all SonarQube project keys and their maintainability scores.
        Set SONARQUBE_URL and SONARQUBE_TOKEN before running.
        """
        import os
        base_url = os.getenv('SONARQUBE_URL', 'http://localhost:9000')
        token = 'squ_ef497f4dedc5a94afe35c963445e836e2304fcde'
        client = SonarQubeClient(base_url, token)
        print("Listing all SonarQube project keys:")
        project_keys = client.list_all_projects()
        for key in project_keys:
            print(key)
        print("\nProject maintainability scores:")
        for key in project_keys:
            measures = client.get_project_measures(key)
            score = client.calculate_maintainability_score(measures) if measures else None
            print(f"{key}: {score}")
    def list_all_projects(self) -> List[str]:
        """
        List all existing SonarQube project keys.
        Returns:
            List of project keys (str)
        """
        endpoint = f"{self.base_url}/api/projects/search"
        params = {"ps": 500}  # Page size
        all_projects = []
        page = 1
        while True:
            params["p"] = page
            try:
                response = self.session.get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                projects = data.get("components", [])
                for proj in projects:
                    key = proj.get("key")
                    if key:
                        all_projects.append(key)
                paging = data.get("paging", {})
                if page >= paging.get("total", 1):
                    break
                page += 1
            except Exception as e:
                logger.error(f"Error listing SonarQube projects: {e}")
                break
        logger.info(f"Found {len(all_projects)} SonarQube projects.")
        return all_projects
    """
    Client for fetching code quality metrics from SonarQube API
    """
    
    def __init__(self, base_url: str, token: str):
        """
        Initialize SonarQube client
        
        Parameters:
        - base_url: SonarQube server URL (e.g., http://localhost:9000)
        - token: Authentication token
        """
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        self.session.auth = (token, '')  # Token auth (username=token, password=empty)
        self.session.headers.update({'Accept': 'application/json'})
    
    def get_project_measures(self, project_key: str, metrics: List[str] = None) -> Dict:
        """
        Get project-level quality metrics
        
        Parameters:
        - project_key: SonarQube project key
        - metrics: List of metric keys (default: common quality metrics)
        
        Returns:
        - Dictionary of metric_key -> value
        """
        if metrics is None:
            # Default comprehensive metrics
            metrics = [
                'ncloc',                    # Lines of code
                'complexity',               # Cyclomatic complexity
                'cognitive_complexity',     # Cognitive complexity
                'code_smells',              # Code smell count
                'sqale_index',              # Technical debt (minutes)
                'sqale_rating',             # Maintainability rating (A-E)
                'sqale_debt_ratio',         # Technical debt ratio
                'bugs',                     # Bug count
                'vulnerabilities',          # Security vulnerabilities
                'security_hotspots',        # Security hotspot count
                'duplicated_lines_density', # % duplicated lines
                'coverage',                 # Test coverage %
                'reliability_rating',       # Reliability rating (A-E)
                'security_rating',          # Security rating (A-E)
            ]
        
        endpoint = f"{self.base_url}/api/measures/component"
        params = {
            'component': project_key,
            'metricKeys': ','.join(metrics)
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse measures
            measures = {}
            for measure in data.get('component', {}).get('measures', []):
                metric_key = measure['metric']
                value = measure.get('value', measure.get('periods', [{}])[0].get('value', 0))
                measures[metric_key] = value
            
            logger.info(f"Fetched {len(measures)} metrics for project {project_key}")
            return measures
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching project measures: {e}")
            return {}
    
    def get_file_measures(self, project_key: str, file_path: str) -> Dict:
        """
        Get file-level quality metrics
        
        Parameters:
        - project_key: SonarQube project key
        - file_path: Relative file path in project
        
        Returns:
        - Dictionary of metric_key -> value for specific file
        """
        # SonarQube component key format: projectKey:filePath
        component_key = f"{project_key}:{file_path}"
        
        metrics = [
            'ncloc',
            'complexity',
            'cognitive_complexity',
            'code_smells',
            'sqale_index',              # Technical debt
            'bugs',
            'vulnerabilities',
            'duplicated_lines_density',
            'coverage',
        ]
        
        endpoint = f"{self.base_url}/api/measures/component"
        params = {
            'component': component_key,
            'metricKeys': ','.join(metrics)
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            measures = {}
            for measure in data.get('component', {}).get('measures', []):
                metric_key = measure['metric']
                value = measure.get('value', 0)
                measures[metric_key] = self._parse_metric_value(value)
            
            return measures
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"No metrics for {file_path} in {project_key}: {e}")
            return {}
    
    def get_all_files_measures(self, project_key: str) -> pd.DataFrame:
        """
        Get metrics for all files in a project
        
        Returns:
        - DataFrame with columns: [file_path, metric1, metric2, ...]
        """
        # Get all project files
        files = self._get_project_files(project_key)
        
        file_metrics = []
        for file_info in files:
            file_path = file_info['path']
            component_key = file_info['key']
            
            # Get file measures
            measures = self._get_component_measures(component_key)
            measures['file_path'] = file_path
            measures['component_key'] = component_key
            file_metrics.append(measures)
        
        df = pd.DataFrame(file_metrics)
        logger.info(f"Fetched metrics for {len(df)} files in {project_key}")
        return df
    
    def _get_project_files(self, project_key: str) -> List[Dict]:
        """Get all files in a project"""
        endpoint = f"{self.base_url}/api/components/tree"
        params = {
            'component': project_key,
            'qualifiers': 'FIL',  # Files only
            'ps': 500  # Page size
        }
        
        all_files = []
        page = 1
        
        while True:
            params['p'] = page
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            files = data.get('components', [])
            all_files.extend(files)
            
            # Check if more pages
            paging = data.get('paging', {})
            if page >= paging.get('pageIndex', 1) and len(files) < paging.get('pageSize', 500):
                break
            
            page += 1
        
        return all_files
    
    def _get_component_measures(self, component_key: str) -> Dict:
        """Get measures for a specific component"""
        metrics = [
            'ncloc', 'complexity', 'cognitive_complexity',
            'code_smells', 'sqale_index', 'bugs', 'vulnerabilities',
            'duplicated_lines_density', 'coverage'
        ]
        
        endpoint = f"{self.base_url}/api/measures/component"
        params = {
            'component': component_key,
            'metricKeys': ','.join(metrics)
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            measures = {}
            for measure in data.get('component', {}).get('measures', []):
                metric_key = measure['metric']
                value = measure.get('value', 0)
                measures[metric_key] = self._parse_metric_value(value)
            
            return measures
        except:
            return {}
    
    def _parse_metric_value(self, value):
        """Parse metric value to appropriate type"""
        try:
            # Try float first
            return float(value)
        except (ValueError, TypeError):
            # Return as-is if not numeric (e.g., ratings like 'A', 'B')
            return value
    
    def calculate_maintainability_score(self, metrics: Dict) -> float:
        """
        Calculate normalized maintainability score (0-1) from SonarQube metrics
        
        Higher score = higher maintenance need (worse quality)
        
        Formula combines multiple quality dimensions:
        - Technical debt ratio
        - Code smells density
        - Complexity
        - Bug/vulnerability presence
        - Test coverage
        
        Adjusted caps to produce better distribution across risk categories.
        """
        # Extract metrics (with defaults), handling NaN values
        import math
        
        def safe_float(value, default=0):
            """Convert to float, handling None and NaN"""
            try:
                val = float(value) if value is not None else default
                return default if math.isnan(val) else val
            except (ValueError, TypeError):
                return default
        
        sqale_debt_ratio = safe_float(metrics.get('sqale_debt_ratio', 0))  # %
        code_smells = int(safe_float(metrics.get('code_smells', 0)))
        ncloc = max(int(safe_float(metrics.get('ncloc', 1))), 1)  # Avoid division by zero
        complexity = int(safe_float(metrics.get('complexity', 0)))
        cognitive_complexity = int(safe_float(metrics.get('cognitive_complexity', complexity)))  # Use cognitive if available
        bugs = int(safe_float(metrics.get('bugs', 0)))
        vulnerabilities = int(safe_float(metrics.get('vulnerabilities', 0)))
        coverage = safe_float(metrics.get('coverage', 0))  # Default 0% (pessimistic)
        duplicated_lines_density = safe_float(metrics.get('duplicated_lines_density', 0))
        
        # Normalize metrics to 0-1 scale with MORE SENSITIVE caps
        
        # 1. Technical debt ratio (0-100% -> 0-1, cap at 5% instead of 10%)
        # Most well-maintained files have <2% debt ratio
        debt_score = min(sqale_debt_ratio / 5, 1.0)
        
        # 2. Code smells per 100 lines (cap at 1.5 smells/100 LOC instead of 2)
        # Good code has <0.5 smell per 100 lines
        smells_per_100_loc = (code_smells / ncloc) * 100
        smells_score = min(smells_per_100_loc / 1.5, 1.0)
        
        # 3. Cognitive complexity per 100 lines (cap at 20/100 LOC instead of 30)
        # Cognitive complexity is more meaningful than cyclomatic complexity
        complexity_per_100_loc = (cognitive_complexity / ncloc) * 100
        complexity_score = min(complexity_per_100_loc / 20, 1.0)
        
        # 4. Issues (bugs + vulnerabilities) per 100 lines (cap at 0.5 issue/100 LOC instead of 1)
        # Any bugs/vulnerabilities are serious
        issues = bugs + vulnerabilities
        issues_per_100_loc = (issues / ncloc) * 100
        issues_score = min(issues_per_100_loc / 0.5, 1.0)
        
        # 5. Coverage deficit (0-100% -> 1-0, with non-linear scaling)
        # Emphasize low coverage more (0% = 1.0, 50% = 0.5, 80% = 0.1, 100% = 0.0)
        if coverage >= 80:
            coverage_deficit_score = (100 - coverage) / 100  # Linear for high coverage
        else:
            coverage_deficit_score = 0.2 + 0.8 * (80 - coverage) / 80  # Steeper penalty for low coverage
        
        # 6. Duplication penalty (cap at 5% duplication instead of 10%)
        duplication_score = min(duplicated_lines_density / 5, 1.0)
        
        # Weighted combination (weights sum to 1.0)
        # Adjusted weights to emphasize code quality issues more
        needs_maintenance = (
            0.20 * debt_score +             # Technical debt
            0.30 * smells_score +            # Code smells (highest weight - most actionable)
            0.25 * complexity_score +        # Cognitive complexity (second highest)
            0.15 * issues_score +            # Bugs/vulnerabilities
            0.05 * coverage_deficit_score +  # Test coverage
            0.05 * duplication_score         # Code duplication
        )
        
        return min(needs_maintenance, 1.0)  # Ensure 0-1 range
    
    def get_temporal_metrics(
        self, 
        current_project_key: str, 
        historical_project_key: str, 
        file_path: str
    ) -> Optional[Dict]:
        """
        Get metrics from both current and historical states for a file.
        
        This method enables temporal degradation analysis by comparing
        metrics from two time points (e.g., current vs 100 days ago).
        
        Args:
            current_project_key: SonarQube project key for current state (e.g., "numpy")
            historical_project_key: SonarQube project key for historical state (e.g., "numpy-historical")
            file_path: Relative path to the file
            
        Returns:
            Dictionary with current metrics, historical metrics, and degradation scores
            None if data is unavailable for either time point
        """
        # Fetch current metrics
        current_metrics = self.get_file_measures(current_project_key, file_path)
        if not current_metrics:
            logger.debug(f"No current metrics for {file_path} in {current_project_key}")
            return None
        
        # Fetch historical metrics
        historical_metrics = self.get_file_measures(historical_project_key, file_path)
        if not historical_metrics:
            logger.debug(f"No historical metrics for {file_path} in {historical_project_key}")
            return None
        
        # Calculate maintainability scores
        current_score = self.calculate_maintainability_score(current_metrics)
        historical_score = self.calculate_maintainability_score(historical_metrics)
        
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
    ) -> Dict[str, Dict]:
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
    
    def verify_historical_projects_exist(self, project_keys: List[str]) -> tuple:
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
                response = self.session.get(
                    f"{self.base_url}/api/projects/search",
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
