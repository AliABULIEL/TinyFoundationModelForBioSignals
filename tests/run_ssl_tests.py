#!/usr/bin/env python3
"""Run SSL unit tests and display results.

Usage:
    python tests/run_ssl_tests.py
    python tests/run_ssl_tests.py --verbose
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
import subprocess
from pathlib import Path

def run_tests(verbose=False):
    """Run SSL unit tests."""
    
    test_files = [
        'tests/test_ssl_masking.py',
        'tests/test_ssl_objectives.py'
    ]
    
    print("=" * 70)
    print("SSL Unit Tests")
    print("=" * 70)
    
    all_passed = True
    
    for test_file in test_files:
        print(f"\n📋 Running: {test_file}")
        print("-" * 70)
        
        cmd = ['pytest', test_file, '-v' if verbose else '-q', '--tb=short']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode != 0:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All SSL tests passed!")
    else:
        print("❌ Some tests failed. See output above.")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    sys.exit(run_tests(verbose))
