#!/bin/bash
#
# Example: Run standalone risk prediction on readlike-me repository
#
# This script demonstrates how to use predictmeifyoucan.py to analyze
# a local git repository and generate risk predictions.
#

# Configuration  
REPO_PATH="/Users/megi/Documents/Other/TechDebtPOC/TDGPTRepos/codewave"
MODEL_PATH="/Users/megi/Documents/Other/TechDebtPOC/code-risk-guard/dags/src/models/artifacts/xgboost_degradation_model_v14.pkl"
OUTPUT_DIR="results"
BRANCH="main"
MAX_COMMITS=10000
WINDOW_SIZE_DAYS=150

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Standalone Risk Prediction Example${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if repository exists
if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}❌ Repository not found: $REPO_PATH${NC}"
    echo "Please update REPO_PATH in this script"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model not found: $MODEL_PATH${NC}"
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

echo -e "${GREEN}✓${NC} Repository: $REPO_PATH"
echo -e "${GREEN}✓${NC} Model: $MODEL_PATH"
echo -e "${GREEN}✓${NC} Output directory: $OUTPUT_DIR"
echo -e "${GREEN}✓${NC} Branch: $BRANCH"
echo -e "${GREEN}✓${NC} Max commits: $MAX_COMMITS"
echo -e "${GREEN}✓${NC} Window size: $WINDOW_SIZE_DAYS days"
echo ""

# Run prediction
echo -e "${BLUE}Running prediction...${NC}"
echo ""

python predictmeifyoucan.py \
    --repo-path "$REPO_PATH" \
    --model-path "$MODEL_PATH" \
    --branch "$BRANCH" \
    --max-commits $MAX_COMMITS \
    --window-size-days $WINDOW_SIZE_DAYS \
    --output "$OUTPUT_DIR"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}✓ Prediction complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/predictions_<timestamp>.csv"
    echo ""
    echo "To view the latest results:"
    echo "  ls -lt $OUTPUT_DIR/predictions_*.csv | head -1"
    echo "  # or"
    echo "  cat \$(ls -t $OUTPUT_DIR/predictions_*.csv | head -1)"
else
    echo ""
    echo -e "${RED}❌ Prediction failed. Check the error messages above.${NC}"
    exit 1
fi
