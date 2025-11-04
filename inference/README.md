# Standalone Risk Prediction

Standalone script that analyzes git repositories and predicts file-level maintenance risk using XGBoost.

## Quick Start

```bash
# Install dependencies
pip install GitPython pandas xgboost scikit-learn joblib

# Run prediction
python predictmeifyoucan.py \
    --repo-path /path/to/your/repo \
    --model-path ../dags/src/models/artifacts/xgboost_risk_model_v23.pkl
```

## Usage

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--repo-path` | ✓ | - | Path to local git repository |
| `--model-path` | ✓ | - | Path to trained model (.pkl) |
| `--branch` | | `main` | Branch to analyze |
| `--max-commits` | | `300` | Number of commits to analyze |
| `--output` | | `.` | Output directory |

### Examples

```bash
# Basic usage
python predictmeifyoucan.py \
    --repo-path /Users/megi/Documents/Other/LLM/readlike-me \
    --model-path ../dags/src/models/artifacts/xgboost_risk_model_v23.pkl

# Analyze specific branch with more commits
python predictmeifyoucan.py \
    --repo-path /path/to/repo \
    --model-path ../dags/src/models/artifacts/xgboost_risk_model_v23.pkl \
    --branch develop \
    --max-commits 1000 \
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

1. **Extract Commits**: Analyze git history for file-level changes (lines added/removed, authors, bug fixes)
2. **Generate Features**: Calculate metrics (churn, author concentration, bug density, etc.)
3. **Predict Risk**: XGBoost model scores each file (0.0-1.0) and assigns category

