#!/usr/bin/env python3
"""
MM-Rec Test Runner by Category
Run specific test categories
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from mm_rec.tests import (
    test_components,
    test_associative_scan_validation,
    test_32k_sequence,
    test_gradients,
    test_gradient_flow_detailed,
)


# Test categories
TEST_CATEGORIES = {
    "components": {
        "name": "Component Tests",
        "description": "Tests for core components (MemoryState, MDI, HDS, Attention, MMRecBlock)",
        "module": test_components,
    },
    "associative_scan": {
        "name": "Associative Scan Tests",
        "description": "Tests for associative scan kernel validation",
        "module": test_associative_scan_validation,
    },
    "32k": {
        "name": "32K Sequence Tests",
        "description": "Tests for long sequence processing (32K tokens)",
        "module": test_32k_sequence,
    },
    "gradients": {
        "name": "Gradient Tests",
        "description": "Tests for gradient correctness and numerical stability",
        "module": test_gradients,
    },
    "gradient_flow": {
        "name": "Gradient Flow Tests",
        "description": "Detailed tests for gradient flow analysis",
        "module": test_gradient_flow_detailed,
    },
    "all": {
        "name": "All Tests",
        "description": "Run all test categories",
        "module": None,
    },
}


def run_category(category: str, verbosity: int = 2):
    """Run tests for a specific category."""
    if category not in TEST_CATEGORIES:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(TEST_CATEGORIES.keys())}")
        return False
    
    cat_info = TEST_CATEGORIES[category]
    print("=" * 80)
    print(f"{cat_info['name']}")
    print("=" * 80)
    print(f"Description: {cat_info['description']}")
    print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if category == "all":
        # Run all categories
        for cat_name, cat_info in TEST_CATEGORIES.items():
            if cat_name != "all" and cat_info["module"]:
                try:
                    tests = loader.loadTestsFromModule(cat_info["module"])
                    suite.addTests(tests)
                    print(f"✅ Loaded tests from {cat_info['name']}")
                except Exception as e:
                    print(f"⚠️  Failed to load tests from {cat_info['name']}: {e}")
    else:
        # Run specific category
        if cat_info["module"]:
            try:
                tests = loader.loadTestsFromModule(cat_info["module"])
                suite.addTests(tests)
            except Exception as e:
                print(f"❌ Failed to load tests: {e}")
                return False
        else:
            print("❌ No module for this category")
            return False
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()
    
    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MM-Rec tests by category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available categories:
  components       - Component tests (MemoryState, MDI, HDS, etc.)
  associative_scan - Associative scan kernel validation
  32k              - 32K sequence length tests
  gradients        - Gradient correctness tests
  gradient_flow    - Detailed gradient flow analysis
  all              - Run all test categories

Examples:
  python run_tests_by_category.py components
  python run_tests_by_category.py all -v
  python run_tests_by_category.py gradients -q
        """
    )
    
    parser.add_argument(
        "category",
        choices=list(TEST_CATEGORIES.keys()),
        help="Test category to run"
    )
    
    parser.add_argument(
        "-v", "--verbosity",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="Verbosity level (0=quiet, 1=normal, 2=verbose, 3=very verbose)"
    )
    
    args = parser.parse_args()
    
    # Run tests
    success = run_category(args.category, verbosity=args.verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

