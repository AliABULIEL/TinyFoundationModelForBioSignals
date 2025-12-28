#!/usr/bin/env python3
"""Verify test suite after fixes."""

import subprocess
import sys

def run_specific_tests():
    """Run the previously failing tests to verify fixes."""
    
    test_cases = [
        "tests/test_detect.py::TestPPGDetection::test_ppg_synthetic_regular",
        "tests/test_detect.py::test_integration_ecg_ppg",
        "tests/test_quality.py::TestECGQuality::test_template_correlation_consistent",
        "tests/test_quality.py::TestPPGQuality::test_ppg_ssqi_poor_signal",
        "tests/test_quality.py::TestPPGQuality::test_ppg_ssqi_flat_signal",
        "tests/test_splits.py::test_single_subject",
        "tests/test_windows.py::test_normalization_stats_zscore",
        "tests/test_windows.py::test_normalization_stats_minmax",
        "tests/test_windows.py::test_normalization_robust"
    ]
    
    print("Testing previously failing tests...")
    print("=" * 60)
    
    failed = []
    passed = []
    
    for test in test_cases:
        print(f"\nRunning: {test.split('::')[-1]}...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✅ PASSED")
            passed.append(test)
        else:
            print("  ❌ FAILED")
            failed.append(test)
            # Show error details
            if "FAILED" in result.stdout:
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if "FAILED" in line or "AssertionError" in line:
                        print(f"    Error: {lines[i]}")
                        break
    
    print("\n" + "=" * 60)
    print(f"Results: {len(passed)}/{len(test_cases)} tests passed")
    
    if failed:
        print("\nStill failing:")
        for test in failed:
            print(f"  - {test.split('::')[-1]}")
    else:
        print("\n🎉 All previously failing tests now pass!")
    
    return len(failed) == 0

def run_full_suite():
    """Run the full test suite."""
    print("\n\nRunning full test suite...")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    # Parse results
    if "passed" in result.stdout:
        lines = result.stdout.split('\n')
        for line in lines:
            if "passed" in line and "failed" in line:
                print(f"\n📊 {line}")
                break
    
    return result.returncode == 0

if __name__ == "__main__":
    # Test specific fixes
    fixes_ok = run_specific_tests()
    
    # Run full suite
    suite_ok = run_full_suite()
    
    if fixes_ok and suite_ok:
        print("\n✅ All tests passing - ready for next step!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests still failing - check output above")
        sys.exit(1)
