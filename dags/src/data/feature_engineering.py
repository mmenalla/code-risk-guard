import pandas as pd

from src.data.github_client import GitHubDataCollector


class FeatureEngineer:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw GitHub module stats into ML-ready features.

        Input df columns:
        ['module', 'lines_added', 'lines_removed', 'prs', 'unique_authors', 'bug_prs', 'churn']

        - bug_ratio: fraction of PRs that are bug-fixes → captures historical maintenance risk.

        - churn_per_pr: average lines changed per PR → high churn can indicate instability.

        - lines_per_pr: total lines changed per PR → another measure of change magnitude.

        - lines_per_author: workload per contributor → many changes by few authors can signal fragile code
        """
        df = df.copy()

        # Avoid division by zero
        df['prs'] = df['prs'].replace(0, 1)
        df['unique_authors'] = df['unique_authors'].replace(0, 1)

        # Derived features
        df['bug_ratio'] = df['bug_prs'] / df['prs']
        df['churn_per_pr'] = df['churn'] / df['prs']
        df['lines_per_pr'] = (df['lines_added'] + df['lines_removed']) / df['prs']
        df['lines_per_author'] = (df['lines_added'] + df['lines_removed']) / df['unique_authors']

        # log-transform skewed features
        # log1p compresses large values while keeping zeros → helps models like XGBoost learn patterns better
        # for col in ['lines_added', 'lines_removed', 'churn', 'lines_per_pr', 'lines_per_author']:
        #     df[f'log_{col}'] = np.log1p(df[col])

        feature_cols = [c for c in df.columns if c != 'module']
        return df[['module'] + feature_cols]
