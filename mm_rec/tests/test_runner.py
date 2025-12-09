#!/usr/bin/env python3
"""
Professional Test Runner
Uses pytest with all professional plugins for comprehensive testing.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_tests(
    parallel=False,
    coverage=True,
    html_report=True,
    json_report=True,
    benchmark=False,
    markers=None,
    verbose=True
):
    """Run tests with professional infrastructure."""
    
    # Create reports directory
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Build pytest command
    cmd = ["pytest", "mm_rec/tests/", "-v", "--tb=short", "--durations=10"]
    
    # Parallel execution
    if parallel:
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 8)
        cmd.extend(["-n", str(num_workers)])
        print(f"🔀 Running tests in parallel with {num_workers} workers")
    
    # Coverage
    if coverage:
        cmd.extend([
            "--cov=mm_rec",
            "--cov-report=html:test_reports/coverage",
            "--cov-report=term-missing",
            "--cov-report=xml:test_reports/coverage.xml"
        ])
    
    # HTML report
    if html_report:
        cmd.extend([
            "--html=test_reports/report.html",
            "--self-contained-html"
        ])
    
    # JSON report
    if json_report:
        cmd.extend([
            "--json-report",
            "--json-report-file=test_reports/report.json"
        ])
    
    # Benchmark
    if benchmark:
        cmd.extend([
            "--benchmark-only",
            "--benchmark-sort=mean"
        ])
    
    # Markers
    if markers:
        cmd.extend(["-m", markers])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print()
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    
    # Print summary
    print()
    print("=" * 80)
    print("Test Reports")
    print("=" * 80)
    if html_report:
        print(f"📊 HTML Report: test_reports/report.html")
    if json_report:
        print(f"📊 JSON Report: test_reports/report.json")
    if coverage:
        print(f"📊 Coverage Report: test_reports/coverage/index.html")
    print("=" * 80)
    
    return result.returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Test Runner")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON report")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmarks only")
    parser.add_argument("-m", "--markers", help="Test markers (e.g., 'not slow')")
    
    args = parser.parse_args()
    
    sys.exit(run_tests(
        parallel=args.parallel,
        coverage=not args.no_coverage,
        html_report=not args.no_html,
        json_report=not args.no_json,
        benchmark=args.benchmark,
        markers=args.markers
    ))

