#!/usr/bin/env python3
"""
Comprehensive Test Runner for Data Mining Project
Run all tests for data processing, clustering, and ML algorithms.
"""

import subprocess
import sys
import time
import os

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_test_file(test_file, description):
    """Run a test file and report results."""
    print_header(description)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, test_file)
    
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            cwd=script_dir,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {str(e)}")
        return False

def main():
    """Main test runner."""
    print_header("DATA MINING PROJECT - COMPREHENSIVE TEST SUITE")
    print("📋 This will run all tests for:")
    print("   1. Data Processing API")
    print("   2. Clustering Algorithms")
    print("   3. Machine Learning Algorithms")
    print("\n⚠️  Make sure the Flask server is running at http://localhost:5001")
    print("   (Run 'python run.py' in a separate terminal)\n")
    
    input("Press Enter to continue with tests...")
    
    results = {}
    
    # Test 1: Data Processing
    results['data_processing'] = run_test_file(
        'test_data_processing.py',
        'TEST 1: Data Processing API'
    )
    time.sleep(2)
    
    # Test 2: Clustering
    results['clustering'] = run_test_file(
        'test_clustering_logic.py',
        'TEST 2: Clustering Algorithms'
    )
    time.sleep(2)
    
    # Test 3: Machine Learning
    results['ml_logic'] = run_test_file(
        'test_ml_logic.py',
        'TEST 3: Machine Learning Algorithms'
    )
    
    # Summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
