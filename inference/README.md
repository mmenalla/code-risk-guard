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

....

## How It Works

1. **Extract Commits**: Analyze git history within time window (default: 150 days) for file-level changes
2. **Calculate Base Features (13)**:...
3. **Engineer Features (12)**: ..
4. **Predict Risk**: XGBoost degradation model scores each file ....

## Features (Model v10)




### Selected Features (based on v14 importance)

- `days_active`: Number of days between first and last commit
- `net_lines`: $net\_lines = lines\_added - lines\_deleted$
- `bug_ratio`: $bug\_ratio = \frac{bug\_commits}{commits}$
- `commits_per_day`: $commits\_per\_day = \frac{commits}{days\_active}$
- `commits`: Number of commits affecting the file
- `commits_squared`: $commits\_squared = commits^2$
- `code_stability`: $code\_stability = \frac{churn}{lines\_added + 1}$
- `modification_ratio`: $modification\_ratio = \frac{lines\_deleted}{lines\_added + 1}$
- `commit_density`: $commit\_density = \frac{commits}{days\_active + 1}$
- `bug_commit_rate`: $bug\_commit\_rate = \frac{bug\_commits}{commits + 1}$
- `bug_commits`: Number of bug-fix commits
- `lines_per_commit`: $lines\_per\_commit = \frac{lines\_added}{commits + 1}$
- `lines_deleted`: Total lines deleted
- `author_concentration`: $author\_concentration = \frac{1}{authors + 1}$

