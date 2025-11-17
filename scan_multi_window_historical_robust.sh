#!/bin/bash

#############################################################################
# Multi-Window Historical SonarQube Scanner (Robust Version)
#
# This version is more robust and handles edge cases better:
# - Doesn't use set -e (continues even if one scan fails)
# - Cleans working directory before checkout
# - Uses force checkout to return to original branch
# - Better error handling
#
# Usage:
#   ./scan_multi_window_historical_robust.sh
#############################################################################

# Configuration
WINDOWS=(50 100 150 200)  # Days ago to scan
SONARQUBE_URL="http://localhost:9000"
SONARQUBE_TOKEN="squ_12f832ac884d1d84540827faeb5f51e6f78280aa"

# Repository configurations: "name:source_directory"
REPOS=(
    "TechDebtGPT:."
    "backend-api:."
    "charts:."
    "tech-debt-api:."
    "tech-debt-services:."
    "scikit-learn:sklearn"
    "pandas:pandas"
    "transformers:src/transformers"
    "numpy:numpy"
    "flask:src/flask"
    "react:packages/react/src"
    "express:lib"
    "kubernetes:pkg"
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🕰️  MULTI-WINDOW HISTORICAL SONARQUBE SCANNER (ROBUST)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Scanning ${#REPOS[@]} repos at ${#WINDOWS[@]} time points"
echo "Time windows: ${WINDOWS[@]} days ago"
echo ""

SUCCESS_COUNT=0
FAILURE_COUNT=0
TOTAL_SCANS=$((${#REPOS[@]} * ${#WINDOWS[@]}))

for REPO_CONFIG in "${REPOS[@]}"; do
    IFS=':' read -r REPO_NAME SOURCE_DIR <<< "$REPO_CONFIG"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📁 Repository: $REPO_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ ! -d "$REPO_NAME" ]; then
        echo "   ❌ Directory not found: $REPO_NAME"
        ((FAILURE_COUNT += ${#WINDOWS[@]}))
        continue
    fi
    
    cd "$REPO_NAME"
    
    # Store current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ -z "$CURRENT_BRANCH" ] || [ "$CURRENT_BRANCH" = "HEAD" ]; then
        # Detached HEAD, get main/master branch
        if git show-ref --verify --quiet refs/heads/main; then
            CURRENT_BRANCH="main"
        elif git show-ref --verify --quiet refs/heads/master; then
            CURRENT_BRANCH="master"
        elif git show-ref --verify --quiet refs/heads/dev; then
            CURRENT_BRANCH="dev"
        else
            echo "   ❌ Cannot determine main branch"
            cd ..
            ((FAILURE_COUNT += ${#WINDOWS[@]}))
            continue
        fi
    fi
    
    echo "   Current branch: $CURRENT_BRANCH"
    
    for DAYS_AGO in "${WINDOWS[@]}"; do
        echo ""
        echo "   ⏰ Scanning at $DAYS_AGO days ago..."
        
        # Clean working directory before checkout
        echo "      Cleaning working directory..."
        git reset --hard HEAD >/dev/null 2>&1
        git clean -fd >/dev/null 2>&1
        
        # Make sure we're on the main branch
        git checkout -f "$CURRENT_BRANCH" >/dev/null 2>&1
        
        # Calculate target date
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            TARGET_DATE=$(date -v-${DAYS_AGO}d +%Y-%m-%d)
        else
            # Linux
            TARGET_DATE=$(date -d "$DAYS_AGO days ago" +%Y-%m-%d)
        fi
        
        echo "      Target date: $TARGET_DATE"
        
        # Find commit closest to target date
        HISTORICAL_COMMIT=$(git rev-list -n 1 --before="$TARGET_DATE 23:59" "$CURRENT_BRANCH" 2>/dev/null)
        
        if [ -z "$HISTORICAL_COMMIT" ]; then
            echo "      ❌ No commits found before $TARGET_DATE"
            ((FAILURE_COUNT++))
            continue
        fi
        
        COMMIT_DATE=$(git log -1 --format=%cd --date=short $HISTORICAL_COMMIT 2>/dev/null)
        echo "      Found commit: ${HISTORICAL_COMMIT:0:7} ($COMMIT_DATE)"
        
        # Checkout historical commit (detached HEAD is fine)
        echo "      Checking out historical commit..."
        if ! git checkout -f $HISTORICAL_COMMIT >/dev/null 2>&1; then
            echo "      ❌ Failed to checkout commit"
            ((FAILURE_COUNT++))
            continue
        fi
        
        # Run SonarQube scan
        PROJECT_KEY="${REPO_NAME}-${DAYS_AGO}d"
        PROJECT_NAME="$REPO_NAME ($DAYS_AGO days ago - $COMMIT_DATE)"
        
        echo "      Running SonarQube scan..."
        echo "      Project key: $PROJECT_KEY"
        
        # Run scan (no timeout on macOS as timeout command not available by default)
        if sonar-scanner \
            -Dsonar.projectKey="$PROJECT_KEY" \
            -Dsonar.projectName="$PROJECT_NAME" \
            -Dsonar.sources="$SOURCE_DIR" \
            -Dsonar.host.url="$SONARQUBE_URL" \
            -Dsonar.token="$SONARQUBE_TOKEN" \
            > /tmp/sonar_scan_${REPO_NAME}_${DAYS_AGO}d.log 2>&1; then
            
            echo "      ✅ Scan completed successfully"
            ((SUCCESS_COUNT++))
        else
            echo "      ❌ Scan failed"
            echo "         Check log: /tmp/sonar_scan_${REPO_NAME}_${DAYS_AGO}d.log"
            ((FAILURE_COUNT++))
        fi
        
        # Clean up .scannerwork directory
        rm -rf .scannerwork 2>/dev/null
    done
    
    # Return to original branch (force to avoid issues)
    echo ""
    echo "   🔄 Returning to $CURRENT_BRANCH..."
    git reset --hard HEAD >/dev/null 2>&1
    git clean -fd >/dev/null 2>&1
    git checkout -f "$CURRENT_BRANCH" >/dev/null 2>&1
    
    cd ..
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 SCAN SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Total scans attempted: $TOTAL_SCANS"
echo "  ✓ Successful: $SUCCESS_COUNT"
echo "  ✗ Failed: $FAILURE_COUNT"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "  ✅ Created $SUCCESS_COUNT SonarQube projects"
    echo ""
fi

if [ $FAILURE_COUNT -eq 0 ]; then
    echo "  🎉 All scans completed successfully!"
    echo ""
    echo "  Next step: Run the multi-window training DAG in Airflow"
    echo "  Expected R² improvement: 0.26 → 0.5-0.7 (2-3x better)"
    echo ""
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "  ⚠️  Some scans failed, but $SUCCESS_COUNT succeeded"
    echo "  💡 You can proceed with training using the successful scans"
    echo ""
else
    echo "  ❌ All scans failed. Check:"
    echo "     • SonarQube is running: curl http://localhost:9000"
    echo "     • Repos are cloned and unshallowed"
    echo "     • sonar-scanner is installed"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
