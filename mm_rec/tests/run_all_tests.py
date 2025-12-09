#!/usr/bin/env python3
"""
MM-Rec Test Suite Runner
Runs all tests and generates a comprehensive report
"""

import sys
import os
import unittest
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from mm_rec.tests import test_components
from mm_rec.tests import test_associative_scan_validation
from mm_rec.tests import test_32k_sequence
from mm_rec.tests import test_gradients
from mm_rec.tests import test_gradient_flow_detailed


def create_test_suite():
    """Create a test suite with all test modules."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    test_modules = [
        test_components,
        test_associative_scan_validation,
        test_32k_sequence,
        test_gradients,
        test_gradient_flow_detailed,
    ]
    
    for module in test_modules:
        try:
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"✅ Loaded tests from {module.__name__}")
        except Exception as e:
            print(f"⚠️  Failed to load tests from {module.__name__}: {e}")
    
    return suite


def run_tests(verbosity=2):
    """Run all tests and return results."""
    print("=" * 80)
    print("MM-Rec Test Suite")
    print("=" * 80)
    print()
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    elapsed_time = time.time() - start_time
    
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
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print()
    
    if result.failures:
        print("=" * 80)
        print("Failures:")
        print("=" * 80)
        for test, traceback in result.failures:
            print(f"\n❌ {test}")
            print(traceback)
    
    if result.errors:
        print("=" * 80)
        print("Errors:")
        print("=" * 80)
        for test, traceback in result.errors:
            print(f"\n❌ {test}")
            print(traceback)
    
    if result.skipped:
        print("=" * 80)
        print("Skipped Tests:")
        print("=" * 80)
        for test, reason in result.skipped:
            print(f"\n⏭️  {test}: {reason}")
    
    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0
    return success, result


if __name__ == "__main__":
    # Parse command line arguments
    verbosity = 2
    if len(sys.argv) > 1:
        if sys.argv[1] == "-v":
            verbosity = 2
        elif sys.argv[1] == "-q":
            verbosity = 0
        elif sys.argv[1] == "-vv":
            verbosity = 3
    
    # Run tests
    success, result = run_tests(verbosity=verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

