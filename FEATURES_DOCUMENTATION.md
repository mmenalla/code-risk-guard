# MaintSight Features Documentation

Complete reference for all features used in the risk prediction model.

---

## 📊 Feature Categories

1. **Raw GitHub Metrics** - Collected directly from GitHub API
2. **Basic Calculated Features** - Simple transformations of raw metrics
3. **Calculated Features** - Complex derived features with risk signals
4. **Target Variable** - What we're predicting

---

## 1️⃣ Raw GitHub Metrics (Collected)

These metrics are collected directly from GitHub pull request history using the GitHub API.

### `module` (string)
- **Source**: GitHub file path
- **Example**: `src/auth/jwt_handler.py`
- **Usage**: Identifier for the code file (not used as feature in training)
- **Collection Logic**: Extracted from PR file changes

---

### `lines_added` (integer)
- **Source**: GitHub PR diff stats
- **What it measures**: Total lines of code added to this file across all PRs
- **Example**: `1250` (1,250 lines added in 150 days)
- **Why it matters**: High additions = active development or new functionality
- **Collection Logic**: Sum of `additions` field from GitHub PR API for each file

---

### `lines_removed` (integer)
- **Source**: GitHub PR diff stats
- **What it measures**: Total lines of code deleted from this file across all PRs
- **Example**: `450` (450 lines removed in 150 days)
- **Why it matters**: High deletions = refactoring or cleanup activity
- **Collection Logic**: Sum of `deletions` field from GitHub PR API for each file

---

### `prs` (integer)
- **Source**: GitHub PR count
- **What it measures**: Number of pull requests that touched this file
- **Example**: `25` (file was modified in 25 different PRs)
- **Why it matters**: 
  - High value = frequently changed file (potentially unstable)
  - Could indicate hotspot or core functionality
- **Collection Logic**: Count distinct PRs containing changes to this file
- **Special Handling**: Replaced with `1` if `0` to avoid division by zero

---

### `unique_authors` (integer)
- **Source**: GitHub PR author metadata
- **What it measures**: Number of different developers who modified this file
- **Example**: `7` (7 different developers touched this file)
- **Why it matters**:
  - Low value = knowledge silo (single point of failure)
  - High value = collaborative but potentially complex coordination
- **Collection Logic**: Count distinct PR authors for changes to this file
- **Special Handling**: Replaced with `1` if `0` to avoid division by zero

---

### `bug_prs` (integer)
- **Source**: GitHub PR commit messages (heuristic detection)
- **What it measures**: Number of PRs that were bug fixes
- **Example**: `8` (8 out of 25 PRs were bug fixes)
- **Why it matters**: High bug count = historically problematic code
- **Collection Logic**: Count PRs where commit messages contain bug-related keywords
- **Detection Keywords**:
  ```python
  ['fix', 'bug', 'patch', 'hotfix', 'bugfix', 'issue', 
   'defect', 'error', 'crash', 'broken', 'repair']
  ```
- **Note**: This is a heuristic approximation (relies on commit message conventions)

---

### `churn` (integer)
- **Source**: Calculated from `lines_added` + `lines_removed`
- **What it measures**: Total lines changed (added + deleted)
- **Example**: `1700` (1,250 added + 450 removed)
- **Why it matters**: High churn = high activity (could be good or bad depending on context)
- **Collection Logic**: `churn = lines_added + lines_removed`

---

### `repo_name` (string)
- **Source**: GitHub repository identifier
- **Example**: `scikit-learn/scikit-learn`
- **Usage**: Track which repository the file belongs to (not used in training)
- **Collection Logic**: Repository name from GitHub API

---

### `created_at` (datetime)
- **Source**: Timestamp when data was collected
- **Example**: `2025-10-27T15:30:00Z`
- **Usage**: Track data freshness (not used in training)
- **Collection Logic**: Current timestamp when PR data is fetched

---

## 2️⃣ Basic Calculated Features

These features are simple transformations of the raw GitHub metrics.

### `bug_ratio` (float, 0-1)
- **Formula**: `bug_prs / prs`
- **Example**: `8 / 25 = 0.32` (32% of PRs were bug fixes)
- **What it measures**: Proportion of changes that were bug fixes
- **Why it's important**: 
  - **Strongest signal** of maintenance burden
  - High ratio = file has frequent bugs
  - Used as primary signal in heuristic labeling (70% weight)
- **Interpretation**:
  - `< 0.1`: Low bug rate (healthy)
  - `0.1 - 0.3`: Moderate (normal)
  - `> 0.3`: High bug rate (problematic)

---

### `churn_per_pr` (float)
- **Formula**: `churn / prs`
- **Example**: `1700 / 25 = 68` (average 68 lines changed per PR)
- **What it measures**: Average magnitude of change per PR
- **Why it's important**:
  - High value = large changes (risky refactors or feature adds)
  - Low value = small incremental changes (safer)
  - Context-dependent (high churn in auth module = risky, in docs = fine)
- **Used in heuristic labeling**: 30% weight (secondary to bug_ratio)
- **Interpretation**:
  - `< 50`: Small changes (safer)
  - `50 - 200`: Medium changes (normal)
  - `> 200`: Large changes (potentially risky)

---

### `lines_per_pr` (float)
- **Formula**: `(lines_added + lines_removed) / prs`
- **Example**: `(1250 + 450) / 25 = 68` (same as churn_per_pr in this case)
- **What it measures**: Average total lines modified per PR
- **Why it's important**: Similar to churn_per_pr but explicit about additions vs deletions
- **Difference from churn_per_pr**: None in current implementation (they're identical)
- **Note**: Could be deprecated as it's redundant with churn_per_pr

---

### `lines_per_author` (float)
- **Formula**: `(lines_added + lines_removed) / unique_authors`
- **Example**: `1700 / 7 = 243` (each author changed ~243 lines on average)
- **What it measures**: Workload distribution across developers
- **Why it's important**:
  - High value = few developers doing lots of work (knowledge concentration)
  - Low value = work distributed across many devs (better knowledge sharing)
- **Interpretation**:
  - `< 100`: Well-distributed work (good)
  - `100 - 500`: Moderate concentration (normal)
  - `> 500`: High concentration (knowledge risk)

---

## 3️⃣ Calculated Features

These features are more sophisticated transformations that capture specific risk signals.

### `author_concentration` (float, 0-1)
- **Formula**: `1 / unique_authors`
- **Example**: `1 / 7 = 0.143`
- **What it measures**: Knowledge silo risk
- **Why it's important**:
  - High value (close to 1) = single author (high bus factor risk)
  - Low value (close to 0) = many authors (better knowledge distribution)
- **Risk Signal**: Higher = more risky
- **Use Case**: Identifies files where only one person knows the code
- **Interpretation**:
  - `1.0`: Single author (critical risk)
  - `0.5`: Two authors (moderate risk)
  - `0.2`: Five authors (low risk)
  - `< 0.1`: Many authors (minimal risk)

---

### `add_del_ratio` (float)
- **Formula**: `lines_added / lines_removed` (with 1 as minimum denominator)
- **Example**: `1250 / 450 = 2.78`
- **What it measures**: Growth vs refactoring balance
- **Why it's important**:
  - High ratio (> 2) = mostly additions (new code, potentially less tested)
  - Low ratio (< 0.5) = mostly deletions (cleanup, potentially safer)
  - Ratio ≈ 1 = balanced (rewriting)
- **Risk Signal**: Very high or very low can indicate different risks
- **Interpretation**:
  - `> 5`: Heavy growth (new untested code)
  - `2 - 5`: Moderate growth (normal development)
  - `0.5 - 2`: Balanced (refactoring + features)
  - `< 0.5`: Heavy deletion (cleanup/removal)

---

### `deletion_ratio` (float, 0-1)
- **Formula**: `lines_removed / (lines_added + lines_removed)`
- **Example**: `450 / 1700 = 0.265` (26.5% of changes were deletions)
- **What it measures**: Proportion of changes that are deletions
- **Why it's important**:
  - High ratio = cleanup/refactoring (generally safer)
  - Low ratio = mostly additions (new code, potentially riskier)
- **Risk Signal**: Higher = potentially less risky (code cleanup)
- **Interpretation**:
  - `> 0.6`: Heavy refactoring (cleanup)
  - `0.3 - 0.6`: Balanced changes
  - `< 0.3`: Mostly additions (new code)

---

### `bug_density` (float)
- **Formula**: `bug_prs / (lines_added + lines_removed)` (normalized per 1000 lines if scaled)
- **Example**: `8 / 1700 = 0.0047` (0.47 bugs per 100 lines)
- **What it measures**: Bug rate normalized by code volume
- **Why it's important**:
  - Separates "buggy code" from "frequently changed code"
  - A file with 10 bugs in 100 lines is worse than 10 bugs in 10,000 lines
- **Risk Signal**: Higher = more bugs per unit of code
- **Used in heuristic labeling**: Boosts risk score by up to 10%
- **Interpretation**:
  - `< 0.001`: Very low bug density (healthy)
  - `0.001 - 0.01`: Normal bug density
  - `> 0.01`: High bug density (problematic)

---

### `collaboration_complexity` (float)
- **Formula**: `unique_authors × churn_per_pr`
- **Example**: `7 × 68 = 476`
- **What it measures**: Coordination risk from many people making big changes
- **Why it's important**:
  - High value = many developers + large changes = coordination overhead
  - Captures risk from complex team dynamics
  - Files with high collaboration complexity may have merge conflicts, communication overhead
- **Risk Signal**: Higher = more coordination risk
- **Interpretation**:
  - `< 100`: Simple collaboration
  - `100 - 500`: Moderate complexity
  - `> 500`: High coordination risk

---

### `prs_per_day` (float) *[Optional - if temporal data available]*
- **Formula**: `prs / days_tracked`
- **Example**: `25 / 150 = 0.167` (one PR every 6 days)
- **What it measures**: Change frequency
- **Why it's important**:
  - High value = constant changes (potentially unstable)
  - Low value = stable file
- **Risk Signal**: Higher = more frequent changes = higher instability risk
- **Note**: Only calculated if `days_tracked` column exists in data
- **Interpretation**:
  - `> 1.0`: Multiple PRs per day (very active)
  - `0.5 - 1.0`: Frequent changes (active)
  - `0.1 - 0.5`: Moderate activity
  - `< 0.1`: Rarely changed (stable)

---

### `age_to_activity_ratio` (float) *[Optional - if temporal data available]*
- **Formula**: `file_age_days / last_modified_days`
- **Example**: `730 / 10 = 73` (2-year-old file modified 10 days ago)
- **What it measures**: Old files with recent activity
- **Why it's important**:
  - High ratio = old file recently touched (risky - legacy code changes)
  - Low ratio = recent file or not recently changed (safer)
- **Risk Signal**: Higher = old code being modified (potentially fragile)
- **Note**: Only calculated if temporal columns exist
- **Interpretation**:
  - `> 100`: Old file, recently modified (risky)
  - `10 - 100`: Moderate
  - `< 10`: Recent file or stable

---

## 4️⃣ Target Variable

### `needs_maintenance` (float, 0-1)
- **Source**: Hybrid labeling strategy
- **What it measures**: Risk score that file needs maintenance
- **How it's created**:
  
  **Method 1: Heuristic (Initial Training)**
  ```python
  needs_maintenance = 0.7 × bug_ratio + 0.3 × normalized_churn
  # Then boosted by:
  # + up to 10% from author_concentration
  # + up to 10% from bug_density
  ```
  
  **Method 2: Manager Feedback (Continuous Learning)**
  - Manager reviews predictions in dashboard
  - Adjusts risk score: `predicted_risk: 0.85 → manager_risk: 0.40`
  - Manager score becomes ground truth for that file
  
- **Range**: 0.0 (no risk) to 1.0 (critical risk)
- **Usage**: Target variable for XGBoost training

---

## 📊 Feature Summary Table

| Feature | Type | Source | Risk Signal | Weight in Heuristic |
|---------|------|--------|-------------|---------------------|
| **Raw Metrics** ||||
| `module` | string | GitHub | Identifier | - |
| `lines_added` | int | GitHub | Neutral | - |
| `lines_removed` | int | GitHub | Neutral | - |
| `prs` | int | GitHub | Higher = more changes | - |
| `unique_authors` | int | GitHub | Lower = knowledge risk | - |
| `bug_prs` | int | GitHub (heuristic) | Higher = more bugs | - |
| `churn` | int | Calculated | Higher = more activity | - |
| `repo_name` | string | GitHub | Metadata | - |
| `created_at` | datetime | System | Metadata | - |
| **Basic Features** ||||
| `bug_ratio` | float | Calculated | Higher = more risky | 70% |
| `churn_per_pr` | float | Calculated | Higher = bigger changes | 30% |
| `lines_per_pr` | float | Calculated | Higher = bigger changes | - |
| `lines_per_author` | float | Calculated | Higher = concentrated work | - |
| **Advanced Features** ||||
| `author_concentration` | float | Calculated | Higher = knowledge silo | +10% boost |
| `add_del_ratio` | float | Calculated | Extreme = risky | - |
| `deletion_ratio` | float | Calculated | Higher = safer | - |
| `bug_density` | float | Calculated | Higher = more risky | +10% boost |
| `collaboration_complexity` | float | Calculated | Higher = coordination risk | - |
| `prs_per_day` | float | Calculated (optional) | Higher = unstable | - |
| `age_to_activity_ratio` | float | Calculated (optional) | Higher = legacy risk | - |
| **Target** ||||
| `needs_maintenance` | float | Hybrid | 0=safe, 1=risky | (what we predict) |

---

## 🎯 Feature Importance in Heuristic Labeling

When no manager feedback is available, risk scores are calculated as:

```python
# Base score (0-1 range)
base_score = 0.7 × bug_ratio + 0.3 × normalized_churn_per_pr

# Boost from single authorship
if author_concentration > 0.5:  # 2 or fewer authors
    base_score += author_concentration × 0.1  # up to +10%

# Boost from bug density
if bug_density in top 95th percentile:
    base_score += (bug_density / max_density) × 0.1  # up to +10%

# Final clipping
needs_maintenance = min(base_score, 1.0)
```

**Weights Rationale**:
- **70% bug_ratio**: Bugs are the strongest signal of maintenance needs
- **30% churn**: High activity can indicate instability but is context-dependent
- **+10% each**: Author concentration and bug density are secondary boosters

---

## 🔬 Feature Engineering Philosophy

### What Makes a Good Feature?

1. **Interpretable**: Can explain to manager why file is risky
2. **Actionable**: Points to specific problems (bugs, knowledge silos, etc.)
3. **Stable**: Doesn't fluctuate wildly with minor changes
4. **Signal-to-Noise**: Correlates with actual maintenance needs

### Why These Features?

| Feature Category | Maintenance Signal |
|------------------|-------------------|
| **Bug metrics** | Direct evidence of past problems |
| **Churn metrics** | Instability, frequent changes |
| **Ownership** | Knowledge silos, bus factor |
| **Collaboration** | Coordination complexity |
| **Change pattern** | Growth vs cleanup, activity recency |

---

## 📈 Future Feature Ideas

### Potentially High-Value (Not Yet Implemented)

1 **Test Coverage** (if CI/CD integration available)
   - `test_coverage`: % of code covered by tests
   - `tests_per_line`: Test density
   - **Source**: Coverage reports from pytest/coverage.py

2 **Deployment Metrics** (if CI/CD integration available)
   - `deployment_frequency`: How often deployed
   - `failed_deployments`: Deployment failure rate
   - **Source**: CI/CD pipeline logs

3 **Review Metrics** (from GitHub)
   - `avg_review_time`: Time to get PR approved
   - `review_comments`: Number of review comments
   - `review_rejections`: PRs that needed changes
   - **Source**: GitHub PR review API

4 **Developer Experience**
   - `avg_author_experience`: Average tenure of contributors
   - `junior_author_ratio`: % of changes by junior devs
   - **Source**: GitHub user data + company records

---

### XGBoost Feature Importance
```python
# After training, check which features matter most
import matplotlib.pyplot as plt

# Get feature importance
importance = model.get_booster().get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Plot
plt.barh([f[0] for f in sorted_importance], [f[1] for f in sorted_importance])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()
```

