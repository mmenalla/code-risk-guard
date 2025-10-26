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

        Current features:
        - bug_ratio: fraction of PRs that are bug-fixes → captures historical maintenance risk.
        - churn_per_pr: average lines changed per PR → high churn can indicate instability.
        - lines_per_pr: total lines changed per PR → another measure of change magnitude.
        - lines_per_author: workload per contributor → many changes by few authors can signal fragile code
        
        New enhanced features:
        - author_concentration: 1/unique_authors → single author = knowledge risk
        - add_del_ratio: additions/deletions → high ratio = growth, low = refactoring
        - deletion_ratio: deletions/total_lines → high = cleanup/refactoring (less risky)
        - bug_density: bugs per lines of code → bugs normalized by code volume
        - collaboration_complexity: authors × churn → coordination risk
        - prs_per_day: PR frequency → constant changes = unstable (if days_tracked available)
        """
        df = df.copy()

        df['prs'] = df['prs'].replace(0, 1)
        df['unique_authors'] = df['unique_authors'].replace(0, 1)

        # === EXISTING FEATURES ===
        df['bug_ratio'] = df['bug_prs'] / df['prs']
        df['churn_per_pr'] = df['churn'] / df['prs']
        df['lines_per_pr'] = (df['lines_added'] + df['lines_removed']) / df['prs']
        df['lines_per_author'] = (df['lines_added'] + df['lines_removed']) / df['unique_authors']

        # === NEW FEATURES ===
        
        # 1. Ownership concentration (single author = risky)
        # Higher value = fewer authors = more concentrated knowledge = higher risk
        df['author_concentration'] = 1.0 / df['unique_authors']
        
        # 2. Change imbalance (lots of deletions = refactoring, less risky)
        # High ratio = mostly additions (new code, potentially riskier)
        # Low ratio = mostly deletions (cleanup, potentially less risky)
        df['add_del_ratio'] = df['lines_added'] / df['lines_removed'].replace(0, 1)
        df['deletion_ratio'] = df['lines_removed'] / (df['lines_added'] + df['lines_removed']).replace(0, 1)
        
        # 3. Bug density (bugs per lines of code)
        # Normalizes bug count by code volume
        total_lines = df['lines_added'] + df['lines_removed']
        df['bug_density'] = df['bug_prs'] / total_lines.replace(0, 1)
        
        # 4. Collaborative complexity (many authors + high churn = coordination risk)
        # High value = many people making big changes = potential coordination issues
        df['collaboration_complexity'] = df['unique_authors'] * df['churn_per_pr']
        
        # 5. Change frequency (constant changes = unstable)
        # Only calculate if days_tracked is available
        if 'days_tracked' in df.columns:
            df['prs_per_day'] = df['prs'] / df['days_tracked'].replace(0, 1)
        
        # 6. Activity recency (old files with recent changes = risky)
        # Only calculate if temporal data is available
        if 'file_age_days' in df.columns and 'last_modified_days' in df.columns:
            # High ratio = old file that was recently touched = potential risk
            df['age_to_activity_ratio'] = df['file_age_days'] / df['last_modified_days'].replace(0, 1)

        feature_cols = [c for c in df.columns if c != 'module']
        return df[['module'] + feature_cols]
