#!/bin/bash
# Professional test runner script with all features

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MM-Rec Test Suite Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create reports directory
mkdir -p test_reports

# Parse arguments
PARALLEL=false
COVERAGE=true
HTML_REPORT=true
JSON_REPORT=true
BENCHMARK=false
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --no-html)
            HTML_REPORT=false
            shift
            ;;
        --no-json)
            JSON_REPORT=false
            shift
            ;;
        -b|--benchmark)
            BENCHMARK=true
            shift
            ;;
        -m|--markers)
            MARKERS="-m $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest mm_rec/tests/ -v --tb=short --durations=10"

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    NUM_WORKERS=$(python3 -c "import os; print(min(os.cpu_count() or 1, 8))")
    PYTEST_CMD="$PYTEST_CMD -n $NUM_WORKERS"
    echo -e "${YELLOW}Running tests in parallel with $NUM_WORKERS workers${NC}"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=mm_rec --cov-report=html:test_reports/coverage --cov-report=term-missing --cov-report=xml:test_reports/coverage.xml"
fi

# Add HTML report
if [ "$HTML_REPORT" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --html=test_reports/report.html --self-contained-html"
fi

# Add JSON report
if [ "$JSON_REPORT" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --json-report --json-report-file=test_reports/report.json"
fi

# Add benchmark
if [ "$BENCHMARK" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --benchmark-only --benchmark-sort=mean"
fi

# Add markers
if [ ! -z "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

echo -e "${GREEN}Running: $PYTEST_CMD${NC}"
echo ""

# Run tests
eval $PYTEST_CMD

EXIT_CODE=$?

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${GREEN}========================================${NC}"

if [ "$HTML_REPORT" = true ]; then
    echo -e "📊 HTML Report: test_reports/report.html"
fi

if [ "$JSON_REPORT" = true ]; then
    echo -e "📊 JSON Report: test_reports/report.json"
fi

if [ "$COVERAGE" = true ]; then
    echo -e "📊 Coverage Report: test_reports/coverage/index.html"
fi

echo ""

exit $EXIT_CODE

