# Standalone Risk Prediction

Standalone script that analyzes git repositories and predicts file-level maintenance risk using XGBoost degradation model (v10).

## Quick Start

```bash
# Install dependencies
pip install GitPython pandas xgboost scikit-learn joblib

# Run prediction with v10 model (25 features: 13 base + 12 engineered)
python predictmeifyoucan.py \
    --repo-path /path/to/your/repo \
    --model-path ../dags/src/models/artifacts/xgboost_degradation_model_v10.pkl
```

## Usage

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--repo-path` | ✓ | - | Path to local git repository |
| `--model-path` | ✓ | - | Path to trained model (.pkl) |
| `--branch` | | `main` | Branch to analyze |
| `--max-commits` | | `10000` | Number of commits to analyze |
| `--window-size-days` | | `150` | Time window in days (matches training) |
| `--output` | | `.` | Output directory |

### Examples

```bash
# Basic usage with v10 degradation model
python predictmeifyoucan.py \
    --repo-path /Users/megi/Documents/Other/LLM/readlike-me \
    --model-path ../dags/src/models/artifacts/xgboost_degradation_model_v10.pkl

# Analyze specific branch with custom time window
python predictmeifyoucan.py \
    --repo-path /path/to/repo \
    --model-path ../dags/src/models/artifacts/xgboost_degradation_model_v10.pkl \
    --branch develop \
    --window-size-days 180 \
    --output results/
```

Or use the provided example:
```bash
./example.sh
```

## Output

Generates `predictions_<timestamp>.csv` sorted by risk score (highest first):

```csv
module,risk_score,risk_category
src/core/engine.py,0.856,high-risk
src/utils/helpers.py,0.623,medium-risk
src/api/routes.py,0.412,low-risk
```

### Risk Categories

| Category | Score Range |
|----------|-------------|
| `no-risk` | 0.00 - 0.22 |
| `low-risk` | 0.22 - 0.47 |
| `medium-risk` | 0.47 - 0.65 |
| `high-risk` | 0.65 - 1.00 |

## How It Works

1. **Extract Commits**: Analyze git history within time window (default: 150 days) for file-level changes
2. **Calculate Base Features (13)**: commits, authors, lines_added, lines_deleted, churn, bug_commits, refactor_commits, feature_commits, lines_per_author, churn_per_commit, bug_ratio, days_active, commits_per_day
3. **Engineer Features (12)**: net_lines, code_stability, is_high_churn_commit, bug_commit_rate, commits_squared, author_concentration, lines_per_commit, churn_rate, modification_ratio, churn_per_author, deletion_rate, commit_density
4. **Predict Risk**: XGBoost degradation model scores each file (0.0-1.0) using all 25 features and assigns category

## Features (Model v10)

The model uses **25 features** (13 base + 12 engineered):

### Base Features (13)
Direct measurements from Git history:
- `commits`, `authors`, `lines_added`, `lines_deleted`, `churn`
- `bug_commits`, `refactor_commits`, `feature_commits`
- `lines_per_author`, `churn_per_commit`, `bug_ratio`
- `days_active`, `commits_per_day`

### Engineered Features (12)
Derived metrics for predictive power:
1. **Code Growth**: `net_lines` (lines added - deleted)
2. **Stability**: `code_stability` (churn relative to additions)
3. **Change Intensity**: `is_high_churn_commit`, `lines_per_commit`
4. **Bug Patterns**: `bug_commit_rate`
5. **Complexity**: `commits_squared`, `commit_density`
6. **Team Dynamics**: `author_concentration`, `churn_per_author`
7. **Modification Patterns**: `modification_ratio`, `deletion_rate`, `churn_rate`

