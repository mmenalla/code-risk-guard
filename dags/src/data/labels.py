import pandas as pd

from src.data.feature_engineering import FeatureEngineer
from src.data.github_client import GitHubDataCollector


class LabelCreator:
    def __init__(self, bug_threshold: float = 0.2, churn_threshold: float = 50):
        """
        Parameters:
        - bug_threshold: fraction of PRs that are bug fixes to consider module high-risk
        - churn_threshold: average churn per PR above which module is considered high-risk
        """
        self.bug_threshold = bug_threshold
        self.churn_threshold = churn_threshold

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'needs_maintenance' column to the DataFrame.

        Assumes df has columns:
        ['module', 'prs', 'bug_prs', 'churn', ...]
        """
        df = df.copy()
        # Avoid division by zero
        df['prs'] = df['prs'].replace(0, 1)

        # Define label
        df['needs_maintenance'] = ((df['bug_ratio'] >= self.bug_threshold) |
                                   (df['churn_per_pr'] >= self.churn_threshold)).astype(int)

        return df
