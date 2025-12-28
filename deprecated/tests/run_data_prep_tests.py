#!/usr/bin/env python3
"""
Comprehensive Test Runner for Data Preparation Enhancements

Runs all tests for:
- Clinical label extraction
- Label-window alignment
- Enhanced BUTPPG loader with windowing
- Integration tests

Author: Senior Data Engineering Team
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import time
from typing import Dict, Tuple

# Import test modules
sys.path.insert(0, str(Path(__file__).parent))
from test_clinical_labels import (
    TestLabelConfig,
    TestClinicalLabelExtractor,
    TestClinicalLabelsIntegration
)
from test_label_alignment import (
    TestWindowLabel,
    TestLabelWindowAligner,
    TestLabelAlignmentIntegration
)
from test_butppg_loader_enhanced import (
    TestBUTPPGLoaderBasic,
    TestBUTPPGLoaderWindowing,
    TestBUTPPGLoaderResampling,
    TestBUTPPGLoaderQuality,
    TestBUTPPGLoaderNormalization,
    TestBUTPPGLoaderIntegration
)


def run_test_suite(test_class, suite_name: str) -> Tuple[bool, Dict]:
    """Run a test suite and return results."""
    print(f"\n{'='*70}")
    print(f"Running {suite_name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    elapsed_time = time.time() - start_time
    
    return result.wasSuccessful(), {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'time': elapsed_time
    }


def main():
    """Run all data preparation tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Data Preparation Enhancement - Comprehensive Test Suite         ║
    ║                                                                  ║
    ║  Testing:                                                        ║
    ║    • Clinical Label Extraction                                   ║
    ║    • Label-Window Alignment                                      ║
    ║    • BUTPPG Loader with Windowing                                ║
    ║    • Integration & End-to-End Workflows                          ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    all_results = {}
    all_passed = True
    
    # Test Clinical Labels
    print("\n" + "█" * 70)
    print("MODULE 1: CLINICAL LABEL EXTRACTION")
    print("█" * 70)
    
    passed, stats = run_test_suite(TestLabelConfig, "Label Configuration Tests")
    all_results['LabelConfig'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestClinicalLabelExtractor, "Clinical Label Extractor Tests")
    all_results['ClinicalLabelExtractor'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestClinicalLabelsIntegration, "Clinical Labels Integration Tests")
    all_results['ClinicalLabelsIntegration'] = stats
    all_passed = all_passed and passed
    
    # Test Label Alignment
    print("\n" + "█" * 70)
    print("MODULE 2: LABEL-WINDOW ALIGNMENT")
    print("█" * 70)
    
    passed, stats = run_test_suite(TestWindowLabel, "Window Label Tests")
    all_results['WindowLabel'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestLabelWindowAligner, "Label Window Aligner Tests")
    all_results['LabelWindowAligner'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestLabelAlignmentIntegration, "Label Alignment Integration Tests")
    all_results['LabelAlignmentIntegration'] = stats
    all_passed = all_passed and passed
    
    # Test BUTPPG Loader
    print("\n" + "█" * 70)
    print("MODULE 3: BUTPPG LOADER WITH WINDOWING")
    print("█" * 70)
    
    passed, stats = run_test_suite(TestBUTPPGLoaderBasic, "BUTPPG Loader Basic Tests")
    all_results['BUTPPGLoaderBasic'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestBUTPPGLoaderWindowing, "BUTPPG Loader Windowing Tests")
    all_results['BUTPPGLoaderWindowing'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestBUTPPGLoaderResampling, "BUTPPG Loader Resampling Tests")
    all_results['BUTPPGLoaderResampling'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestBUTPPGLoaderQuality, "BUTPPG Loader Quality Tests")
    all_results['BUTPPGLoaderQuality'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestBUTPPGLoaderNormalization, "BUTPPG Loader Normalization Tests")
    all_results['BUTPPGLoaderNormalization'] = stats
    all_passed = all_passed and passed
    
    passed, stats = run_test_suite(TestBUTPPGLoaderIntegration, "BUTPPG Loader Integration Tests")
    all_results['BUTPPGLoaderIntegration'] = stats
    all_passed = all_passed and passed
    
    # Print summary
    print("\n" + "═" * 70)
    print("TEST SUMMARY")
    print("═" * 70)
    
    total_tests = sum(r['tests_run'] for r in all_results.values())
    total_failures = sum(r['failures'] for r in all_results.values())
    total_errors = sum(r['errors'] for r in all_results.values())
    total_time = sum(r['time'] for r in all_results.values())
    
    print(f"\nTotal Tests Run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Total Time: {total_time:.2f}s")
    
    print("\nDetailed Results by Module:")
    print("-" * 70)
    for module_name, stats in all_results.items():
        status = "✓ PASS" if stats['failures'] == 0 and stats['errors'] == 0 else "✗ FAIL"
        print(f"{module_name:40s} {status:10s} "
              f"({stats['tests_run']} tests, {stats['time']:.2f}s)")
    
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
