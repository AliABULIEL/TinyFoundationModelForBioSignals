#!/usr/bin/env python3
"""
Master Test Runner: All Research Pipeline Phases

Runs all phase tests in sequence to validate the entire pipeline.

Run:
    python tests/run_all_phases.py
    
    # Or with custom options:
    python tests/run_all_phases.py --data-dir data/vitaldb_windows --quick
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import subprocess
import time
from typing import List, Tuple


def run_test(script: Path, args: List[str] = None) -> Tuple[bool, float]:
    """Run a test script and return success status and runtime."""
    cmd = [sys.executable, str(script)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*70}")
    print(f"Running: {script.name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run all research pipeline phase tests"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/vitaldb_windows',
        help='Path to VitalDB windows'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use smaller datasets for faster testing'
    )
    parser.add_argument(
        '--skip-phase',
        type=int,
        action='append',
        help='Skip phase number (can specify multiple times)'
    )
    args = parser.parse_args()
    
    skip_phases = set(args.skip_phase or [])
    
    print("="*70)
    print("RESEARCH PIPELINE: ALL PHASES TEST")
    print("="*70)
    print("\nThis will run all phase tests in sequence:\n")
    print("  Phase 0: Environment Setup")
    print("  Phase 1: Data Preparation")
    print("  Phase 2: SSL Pretraining (smoke test)")
    print("  Phase 3: Fine-tuning (smoke test)")
    print()
    
    if args.quick:
        print("Mode: QUICK (smaller datasets)")
    else:
        print("Mode: FULL")
    
    if skip_phases:
        print(f"Skipping phases: {sorted(skip_phases)}")
    
    input("\nPress Enter to continue...")
    
    tests_dir = Path(__file__).parent
    results = []
    total_time = 0.0
    
    # Phase 0: Environment
    if 0 not in skip_phases:
        success, elapsed = run_test(tests_dir / 'test_phase0_setup.py')
        results.append(('Phase 0: Environment', success, elapsed))
        total_time += elapsed
        
        if not success:
            print("\n❌ Phase 0 failed. Fix environment issues before continuing.")
            return 1
    
    # Phase 1: Data Preparation
    if 1 not in skip_phases:
        phase1_args = ['--data-dir', args.data_dir]
        success, elapsed = run_test(tests_dir / 'test_phase1_data_prep.py', phase1_args)
        results.append(('Phase 1: Data Prep', success, elapsed))
        total_time += elapsed
        
        if not success:
            print("\n❌ Phase 1 failed. Prepare data before continuing.")
            return 1
    
    # Phase 2: SSL Pretraining smoke test
    if 2 not in skip_phases:
        phase2_args = ['--data-dir', args.data_dir]
        if args.quick:
            phase2_args.extend(['--max-windows', '32'])
        else:
            phase2_args.extend(['--max-windows', '64'])
        
        success, elapsed = run_test(tests_dir / 'test_phase2_ssl_smoke.py', phase2_args)
        results.append(('Phase 2: SSL Smoke Test', success, elapsed))
        total_time += elapsed
        
        if not success:
            print("\n❌ Phase 2 failed. Check SSL implementation.")
            # Continue anyway for Phase 3 test
    
    # Phase 3: Fine-tuning smoke test
    if 3 not in skip_phases:
        phase3_args = []
        if args.quick:
            phase3_args.extend(['--n-samples', '64'])
        else:
            phase3_args.extend(['--n-samples', '128'])
        
        # Try to use Phase 2 checkpoint if available
        phase2_ckpt = Path('artifacts/smoke_ssl/checkpoint.pt')
        if phase2_ckpt.exists():
            phase3_args.extend(['--pretrained', str(phase2_ckpt)])
        
        success, elapsed = run_test(tests_dir / 'test_phase3_finetune_smoke.py', phase3_args)
        results.append(('Phase 3: Finetune Smoke Test', success, elapsed))
        total_time += elapsed
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} phases passed\n")
    
    for name, success, elapsed in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {name:30} | {elapsed:6.1f}s")
    
    print(f"\nTotal runtime: {total_time:.1f}s (~{total_time/60:.1f} min)")
    
    if passed == total:
        print("\n" + "="*70)
        print("🎉 ALL PHASES PASSED!")
        print("="*70)
        print("\nYour research pipeline is ready!")
        print("\nNext steps:")
        print("  1. Run full SSL pretraining:")
        print("     python scripts/pretrain_vitaldb_ssl.py \\")
        print("       --config configs/ssl_pretrain.yaml \\")
        print("       --data-dir data/vitaldb_windows \\")
        print("       --epochs 100")
        print()
        print("  2. After pretraining, fine-tune on BUT-PPG:")
        print("     python scripts/finetune_butppg.py \\")
        print("       --pretrained artifacts/foundation_model/best_model.pt \\")
        print("       --data-dir data/but_ppg \\")
        print("       --epochs 50")
        print("="*70)
        return 0
    else:
        print("\n⚠️  Some phases failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
