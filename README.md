# MaintSight 🛡️

**AI-powered code quality degradation prediction using temporal Git analysis and SonarQube metrics.**

Predict which files will degrade in quality over time based on recent commit patterns, helping teams proactively address technical debt before it becomes critical.

---

## 🎯 Overview

MaintSight is a machine learning platform that predicts **future code quality degradation** by analyzing Git commit patterns and SonarQube quality metrics across multiple time windows. Unlike traditional static analysis tools that only tell you what's wrong *now*, MaintSight predicts what *will* go wrong based on historical patterns.

### Key Innovation: Multi-Window Temporal Analysis

The model analyzes code changes across multiple time windows (e.g., 0-150 days ago, 150-300 days ago) to:
- **Learn degradation patterns**: How past activity led to quality decline
- **Generate more training data**: Multiple samples per file from different time periods
- **Capture temporal dynamics**: Recent vs historical activity patterns
- **Predict future risk**: Which files will degrade based on current activity

**Performance:** R² = 0.36 with 8,099 training samples across 12 repositories using 25 features (13 base + 12 engineered)

---

## 🚀 Features

### Core Capabilities
- ✅ **Multi-Window Temporal Feature Extraction** - Analyzes Git history across sliding time windows
- ✅ **Quality Degradation Prediction** - Predicts future maintainability decline (not just current state)
- ✅ **SonarQube Integration** - Uses historical SonarQube scans for ground truth labels
- ✅ **Hyperparameter Optimization** - RandomizedSearchCV for optimal model performance
- ✅ **Feature Engineering** - 25 features: 13 base features from Git analysis + 12 engineered features
- ✅ **Production-Ready** - No training/serving skew (uses only Git features available at inference time)

### Data Collection Methods
- **Git Commit Analysis** (Recommended) - Direct Git repository analysis, no API limits
- **GitHub PR Analysis** (Legacy) - Pull request-based feature extraction
- **Local Repository Support** - Works with any Git repository

### ML Pipeline
- **Training DAGs**: 4 Airflow DAGs for different training approaches
- **Inference DAG**: Real-time predictions on new code
- **Fine-tuning DAG**: Incremental learning from new data
- **MongoDB Storage**: Labeled data, metrics, and predictions

### Integrations
- 🔧 **SonarQube** - Code quality metrics and historical analysis
- 🤖 **LLM/GPT-4** - Intelligent ticket generation (optional)
- 📊 **JIRA** - Automated technical debt ticket creation (optional)
- 📈 **Streamlit Dashboard** - Manual review and approval workflow (optional)

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          MaintSight                              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │  Airflow│          │ MongoDB │          │SonarQube│
   │   DAGs  │          │ Storage │          │ Scanner │
   └────┬────┘          └─────────┘          └────┬────┘
        │                                          │
   ┌────▼────────────────────────────────────────▼────┐
   │          Data Collection & Labeling               │
   │  • Git commit feature extraction (25 features)    │
   │  • Multi-window temporal sampling                 │
   │  • SonarQube quality degradation labels           │
   └───────────────────────┬───────────────────────────┘
                           │
   ┌────────────────────────▼──────────────────────────┐
   │          XGBoost Training Pipeline                 │
   │  • Feature engineering (simplified model)          │
   │  • Hyperparameter tuning (RandomizedSearchCV)      │
   │  • Cross-validation (3-fold CV)                    │
   │  • Model versioning & metrics logging              │
   └───────────────────────┬───────────────────────────┘
                           │
   ┌────────────────────────▼──────────────────────────┐
   │          Production Inference                      │
   │  • Real-time predictions on new commits            │
   │  • Risk scoring (0.0 - 1.0)                        │
   │  • Optional: JIRA ticket generation                │
   └───────────────────────────────────────────────────┘
```

---

## 🧠 Model Features (25 Total)

The model uses **25 carefully selected features** - a 36% reduction from the original 39 features, focusing only on high-impact predictors.

### Base Features from Git Analysis (13)
Extracted directly from Git commit history:

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `commits` | Number of commits touching the file | Count of commits |
| `authors` | Number of unique contributors | Unique author count |
| `lines_added` | Total lines of code added | Sum of additions |
| `lines_deleted` | Total lines of code removed | Sum of deletions |
| `churn` | Total code churn | lines_added + lines_deleted |
| `bug_commits` | Bug fix commits | Keywords: 'fix', 'bug', 'patch', 'hotfix' |
| `refactor_commits` | Refactoring commits | Keywords: 'refactor', 'clean', 'improve' |
| `feature_commits` | New feature commits | Keywords: 'feat', 'feature', 'add', 'new' |
| `lines_per_author` | Average contribution size | lines_added / authors |
| `churn_per_commit` | Average change size | churn / commits |
| `bug_ratio` | Proportion of bug fixes | bug_commits / commits |
| `days_active` | Development lifespan | Days between first and last commit |
| `commits_per_day` | Commit frequency | commits / days_active |

### Engineered Features (12)
Derived features that capture complex patterns:

| Feature | Description | Formula | Interpretation |
|---------|-------------|---------|----------------|
| `net_lines` | Net code growth | lines_added - lines_deleted | Positive = growth, negative = shrinkage |
| `code_stability` | Churn relative to size | churn / lines_added | Higher = more unstable |
| `is_high_churn_commit` | Large change flag | 1 if churn_per_commit > 100 else 0 | Binary indicator of big changes |
| `bug_commit_rate` | Bug fix intensity | bug_commits / commits | Higher = more bugs |
| `commits_squared` | Non-linear activity | commits² | Captures heavy activity patterns |
| `author_concentration` | Knowledge concentration | 1 / authors | Higher = fewer owners (bus factor) |
| `lines_per_commit` | Avg change magnitude | lines_added / commits | Large commits = risky |
| `churn_rate` | Change velocity | churn / days_active | High velocity = unstable |
| `modification_ratio` | Deletion intensity | lines_deleted / lines_added | High = lots of rework |
| `churn_per_author` | Individual impact | churn / authors | Per-developer change rate |
| `deletion_rate` | Code removal rate | lines_deleted / (lines_added + lines_deleted) | 0-1 scale |
| `commit_density` | Activity concentration | commits / days_active | Burst vs steady work |

### Feature Categories

📊 **Activity Metrics (6)**: commits, commits_squared, commits_per_day, commit_density, days_active, is_high_churn_commit  
🔄 **Code Change (8)**: lines_added, lines_deleted, churn, net_lines, churn_per_commit, churn_rate, lines_per_commit, modification_ratio  
🐛 **Quality Indicators (5)**: bug_commits, bug_ratio, bug_commit_rate, refactor_commits, feature_commits  
👥 **Collaboration (4)**: authors, lines_per_author, author_concentration, churn_per_author  
🛡️ **Stability (2)**: code_stability, deletion_rate  

**Key Insight:** No duplication - each feature is calculated exactly once in a single location, eliminating technical debt in the codebase itself!

---

## 🏗️ Project Structure

```
MaintSight/
├── dags/                           # Airflow DAG definitions
│   ├── risk_model_training_degradation_dag.py  # ⭐ Main training DAG (NEW)
│   ├── risk_model_training_dag.py              # Legacy single-repo training
│   ├── risk_model_training_dag_multi_repo.py   # Legacy multi-repo training
│   ├── risk_model_inference_dag.py             # Production inference
│   ├── risk_model_finetune_dag.py              # Model fine-tuning
│   └── src/
│       ├── data/                               # Data collection & processing
│       │   ├── git_commit_client.py            # Git feature extraction (consolidated)
│       │   ├── sonarqube_client.py             # SonarQube API client (consolidated)
│       │   ├── save_incremental_labeled_data.py# MongoDB persistence
│       │   ├── github_client.py                # GitHub API (legacy)
│       │   ├── feature_engineering.py          # Legacy feature engineering
│       │   └── labels.py                       # Legacy labeling
│       ├── models/
│       │   ├── train.py                        # ⭐ Model training with feature engineering
│       │   ├── predict.py                      # Inference pipeline
│       │   └── explain.py                      # Model explainability
│       ├── llm/
│       │   ├── ticket_generator.py             # LLM-powered ticket generation
│       │   └── prompts.py                      # GPT-4 prompts
│       ├── jira/
│       │   └── jira_client.py                  # JIRA integration
│       ├── dashboard/
│       │   └── app.py                          # Streamlit dashboard
│       └── utils/
│           └── config.py                       # Configuration management
├── inference/
│   └── predictmeifyoucan.py                    # Standalone inference script
├── docker-compose.yml                          # Container orchestration
├── Dockerfile                                  # Airflow container image
├── requirements.txt                            # Python dependencies
├── approval_app.py                             # Streamlit approval interface
└── .env.example                                # Environment configuration template
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **Docker** & **Docker Compose** (required)
- **Python 3.10+** (for local development)
- **Git repositories** cloned locally
- **SonarQube** instance (Docker container or SonarCloud)

### 1. Clone Repository

```bash
git clone https://github.com/mmenalla/code-risk-guard.git
cd code-risk-guard
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example dags/src/.env

# Edit configuration (see Configuration section below)
nano dags/src/.env
```

**Required Configuration:**
```bash
# Repository paths (Docker mounted volume)
REPO_BASE_DIR=/repos/TDGPTRepos
REPO_NAMES=["scikit-learn", "pandas", "transformers", "kubernetes", "numpy"...] 

# SonarQube
SONARQUBE_URL=http://sonarqube:9000
SONARQUBE_TOKEN=sqp_your_token_here
SONARQUBE_PROJECT_KEYS=["sklearn", "pandas", "transformers", "kubernetes", "numpy"...]

# Model training
TUNE_HYPERPARAMETERS=true
WINDOW_SIZE_DAYS=150
MAX_COMMITS=10000
```

### 3. Start Services

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f airflow-scheduler
```

**Services Started:**
- **Airflow Webserver**: http://localhost:8080 (admin/admin)
- **SonarQube**: http://localhost:9000 (admin/admin)
- **MongoDB**: localhost:27017
- **PostgreSQL**: localhost:5432

### 4. Prepare SonarQube Data

```bash
# Navigate to your repositories directory
cd /path/to/TDGPTRepos

# Run historical SonarQube scans (creates multi-window projects)
./scan_multi_window_historical_robust.sh
```

This creates SonarQube projects for each time window:
- `sklearn` (current)
- `sklearn-150d` (150 days ago)
- `sklearn-300d` (300 days ago)

### 5. Trigger Training

1. Open Airflow UI: http://localhost:8080
2. Find DAG: `risk_model_training_degradation`
3. Enable the DAG (toggle switch)
4. Click "Trigger DAG" (play button)

**Training Steps:**
1. ✅ Verify SonarQube projects exist
2. ✅ Extract Git features from multiple time windows
3. ✅ Create degradation labels (compare SonarQube at window boundaries)
4. ✅ Aggregate data from all repositories and windows
5. ✅ Train XGBoost model with hyperparameter tuning
6. ✅ Save model and log metrics to MongoDB

**Expected Duration:** ~10-20 minutes (with hyperparameter tuning)

---

## 📖 Usage

### Training a New Model

**Recommended: Multi-Window Degradation Model**
```python
# Airflow DAG: risk_model_training_degradation
# Features: Temporal Git patterns (15 optimized features)
# Labels: Quality degradation from SonarQube comparisons
# Approach: Multiple time windows per file
```

**Configuration:**
- `TUNE_HYPERPARAMETERS=true` - Better performance (~15 min)
- `TUNE_HYPERPARAMETERS=false` - Faster training (~3 min)
- `WINDOW_SIZE_DAYS=150` - Size of each time window
- `MAX_COMMITS=10000` - Max commits per window

### Making Predictions

...

### Monitoring Model Performance

...

### Generating Technical Debt Tickets

```python
# Requires: OPENAI_API_KEY, JIRA_API_TOKEN

# Airflow DAG: risk_model_inference_dag
# Set: GENERATE_TICKETS=True
#      RISK_SCORE_THRESHOLD=0.2

# Automatically creates JIRA tickets for high-risk files
```

---

## ⚙️ Configuration

### Environment Variables

See `.env.example` for full documentation. Key variables:

**Repositories & Paths**
```bash
REPO_BASE_DIR=/repos/TDGPTRepos          # Base directory for repos
REPO_NAMES=["sklearn", "pandas", ...]     # Repository names (JSON array)
BRANCH_NAME=main                          # Default branch
REPO_BRANCHES={"sklearn": "main"}         # Per-repo branches (JSON object)
```

**SonarQube**
```bash
SONARQUBE_URL=http://sonarqube:9000
SONARQUBE_TOKEN=sqp_...
SONARQUBE_PROJECT_KEYS=["sklearn", ...]
```

**Model Training**
```bash
TUNE_HYPERPARAMETERS=true    # Enable hyperparameter optimization
WINDOW_SIZE_DAYS=150         # Multi-window time period
MAX_COMMITS=10000            # Max commits per window
SINCE_DAYS=150               # Historical lookback period
```

**Optional Features**
```bash
GITHUB_TOKEN=ghp_...         # For GitHub PR analysis (legacy)
OPENAI_API_KEY=sk-...        # For LLM ticket generation
JIRA_API_TOKEN=...           # For JIRA integration
GENERATE_TICKETS=False       # Enable automatic ticket creation
RISK_SCORE_THRESHOLD=0.5     # Threshold for high-risk files
```

**MongoDB**
```bash
MONGO_URI=mongodb://admin:admin@mongo:27017/risk_model_db?authSource=admin
```

### Feature Set (25 Features)

The model uses 13 base features directly from Git analysis and 12 engineered features:

#### Base Features (from GitCommitCollector)

| Feature | Description |
|---------|-------------|
| `commits` | Total number of commits |
| `authors` | Number of unique contributors |
| `lines_added` | Total lines added |
| `lines_deleted` | Total lines deleted |
| `churn` | Total code churn (lines added + deleted) |
| `bug_commits` | Number of bug-fixing commits |
| `refactor_commits` | Number of refactoring commits |
| `feature_commits` | Number of feature commits |
| `lines_per_author` | Lines added per unique author |
| `churn_per_commit` | Average churn per commit |
| `bug_ratio` | Proportion of bug-fixing commits |
| `days_active` | Days between first and last commit |
| `commits_per_day` | Commit frequency |

#### Engineered Features (from FeatureEngineer)

| Feature | Description |
|---------|-------------|
| `net_lines` | Lines added - lines deleted (code growth) |
| `code_stability` | Churn relative to lines added |
| `is_high_churn_commit` | Binary flag for large changes (>100 lines/commit) |
| `bug_commit_rate` | Bug commits / total commits |
| `commits_squared` | Non-linear commit activity |
| `author_concentration` | Knowledge concentration (bus factor) |
| `lines_per_commit` | Average lines changed per commit |
| `churn_rate` | Churn velocity (churn / days active) |
| `modification_ratio` | Deletion rate relative to additions |
| `churn_per_author` | Code change per developer |
| `deletion_rate` | Code removal rate |
| `commit_density` | Commits per day active |


### Label Source: SonarQube Degradation

**Label Calculation:**
```python
label = quality_at_window_end - quality_at_window_start
```

**Example:**
- Window 0 [0-150d]: `label = quality@today - quality@150d`
- Window 1 [150-300d]: `label = quality@150d - quality@300d`

**Positive label** = Quality degraded (maintenance needed)  
**Negative label** = Quality improved (no maintenance needed)

----
### Adding New Repositories

1. Clone repository to `REPO_BASE_DIR`
2. Add to `.env`:
   ```bash
   REPO_NAMES=["sklearn", "pandas", "new-repo"]
   SONARQUBE_PROJECT_KEYS=["sklearn", "pandas", "new-repo"]
   ```
3. Run SonarQube scans:
   ```bash
   cd /path/to/TDGPTRepos/new-repo
   sonar-scanner -Dsonar.projectKey=new-repo ...
   ```
4. Restart Airflow: `docker-compose restart airflow-scheduler`
5. Retrigger training DAG


## 🙏 Acknowledgments

- **Repositories analyzed**: scikit-learn, pandas, transformers, kubernetes, numpy, and more
- **SonarQube** for code quality metrics
- **XGBoost** for gradient boosting framework
- **Apache Airflow** for workflow orchestration

---

**Built with ❤️ for proactive technical debt management**
