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
        df = df.copy()
        df['prs'] = df['prs'].replace(0, 1)

        # Label
        # df['needs_maintenance'] = ((df['bug_ratio'] >= self.bug_threshold) |
        #                            (df['churn_per_pr'] >= self.churn_threshold)).astype(int)
        # Normalize bug_ratio and churn_per_pr to 0–1 range, then average
        df["bug_score"] = df["bug_ratio"].clip(0, 1)
        df["churn_score"] = (df["churn_per_pr"] / df["churn_per_pr"].max()).clip(0, 1)

        # Combine the two metrics to form a continuous "maintenance need" score
        df["needs_maintenance"] = 0.5 * df["bug_score"] + 0.5 * df["churn_score"]
        df["risk_category"] = pd.cut(
            df["needs_maintenance"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["no-risk", "low-risk", "medium-risk", "high-risk"]
        )

        return df.drop(columns=["bug_score", "churn_score"])
