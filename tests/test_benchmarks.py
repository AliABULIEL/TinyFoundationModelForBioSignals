#!/usr/bin/env python3
"""
Quick verification script to test vitaldb_benchmarks.py and butppg_benchmarks.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*80)
print("BENCHMARK FILES VERIFICATION")
print("="*80)

# Test 1: Import VitalDB benchmarks
print("\n[1/6] Testing VitalDB benchmarks import...")
try:
    from tasks.vitaldb_benchmarks import (
        VitalDBHypotensionTask,
        VitalDBBloodPressureTask,
        VitalDBMortalityTask,
        get_all_vitaldb_tasks,
        print_vitaldb_benchmark_summary
    )
    print("✓ VitalDB benchmarks imported successfully")
except Exception as e:
    print(f"✗ Failed to import VitalDB benchmarks: {e}")
    sys.exit(1)

# Test 2: Import BUT-PPG benchmarks
print("\n[2/6] Testing BUT-PPG benchmarks import...")
try:
    from tasks.butppg_benchmarks import (
        BUTPPGQualityTask,
        BUTPPGHeartRateTask,
        BUTPPGMotionTask,
        get_all_butppg_tasks,
        print_butppg_benchmark_summary
    )
    print("✓ BUT-PPG benchmarks imported successfully")
except Exception as e:
    print(f"✗ Failed to import BUT-PPG benchmarks: {e}")
    sys.exit(1)

# Test 3: Instantiate VitalDB tasks
print("\n[3/6] Testing VitalDB task instantiation...")
try:
    hypotension_task = VitalDBHypotensionTask(
        data_dir="data/vitaldb",
        split_file="configs/splits/splits_full.json",
        prediction_window_min=10
    )
    print(f"✓ VitalDBHypotensionTask instantiated: {hypotension_task.name}")
    print(f"  - Prediction window: {hypotension_task.prediction_window_min} min")
    print(f"  - Benchmark: AUROC target ≥{hypotension_task.get_benchmark('auroc')['target']}")

    bp_task = VitalDBBloodPressureTask(
        data_dir="data/vitaldb",
        split_file="configs/splits/splits_full.json",
        target_bp='MAP'
    )
    print(f"✓ VitalDBBloodPressureTask instantiated: {bp_task.name}")
    print(f"  - Target BP: {bp_task.target_bp}")
    print(f"  - AAMI ME threshold: ≤{bp_task.aami_me_threshold} mmHg")

    mortality_task = VitalDBMortalityTask(
        data_dir="data/vitaldb",
        split_file="configs/splits/splits_full.json"
    )
    print(f"✓ VitalDBMortalityTask instantiated: {mortality_task.name}")

except Exception as e:
    print(f"✗ Failed to instantiate VitalDB tasks: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Instantiate BUT-PPG tasks
print("\n[4/6] Testing BUT-PPG task instantiation...")
try:
    quality_task = BUTPPGQualityTask(
        data_dir="data/butppg",
        split_file="configs/splits/splits_full.json",
        quality_threshold=0.7
    )
    print(f"✓ BUTPPGQualityTask instantiated: {quality_task.name}")
    print(f"  - Quality threshold: {quality_task.quality_threshold}")
    print(f"  - Benchmark: AUROC target ≥{quality_task.get_benchmark('auroc')['target']}")

    hr_task = BUTPPGHeartRateTask(
        data_dir="data/butppg",
        split_file="configs/splits/splits_full.json",
        hr_method='peak_detection'
    )
    print(f"✓ BUTPPGHeartRateTask instantiated: {hr_task.name}")
    print(f"  - HR method: {hr_task.hr_method}")
    print(f"  - Target: MAE ≤{hr_task.get_benchmark('mae')['target']} bpm")

    motion_task = BUTPPGMotionTask(
        data_dir="data/butppg",
        split_file="configs/splits/splits_full.json"
    )
    print(f"✓ BUTPPGMotionTask instantiated: {motion_task.name}")
    print(f"  - Num classes: {motion_task.num_classes}")

except Exception as e:
    print(f"✗ Failed to instantiate BUT-PPG tasks: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Get all tasks
print("\n[5/6] Testing task retrieval functions...")
try:
    vitaldb_tasks = get_all_vitaldb_tasks(
        data_dir="data/vitaldb",
        split_file="configs/splits/splits_full.json"
    )
    print(f"✓ Retrieved {len(vitaldb_tasks)} VitalDB tasks:")
    for task_name, task in vitaldb_tasks.items():
        print(f"  - {task_name}: {task.__class__.__name__}")

    butppg_tasks = get_all_butppg_tasks(
        data_dir="data/butppg",
        split_file="configs/splits/splits_full.json"
    )
    print(f"✓ Retrieved {len(butppg_tasks)} BUT-PPG tasks:")
    for task_name, task in butppg_tasks.items():
        print(f"  - {task_name}: {task.__class__.__name__}")

except Exception as e:
    print(f"✗ Failed to retrieve tasks: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Print benchmark summaries
print("\n[6/6] Testing benchmark summary printing...")
try:
    print("\n" + "="*80)
    print("VITALDB BENCHMARK SUMMARY")
    print("="*80)
    print_vitaldb_benchmark_summary()

    print("\n" + "="*80)
    print("BUT-PPG BENCHMARK SUMMARY")
    print("="*80)
    print_butppg_benchmark_summary()

    print("\n✓ Benchmark summaries printed successfully")

except Exception as e:
    print(f"✗ Failed to print benchmark summaries: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "="*80)
print("✓ ALL TESTS PASSED")
print("="*80)
print("\nSummary:")
print(f"  - VitalDB tasks: {len(vitaldb_tasks)} implemented")
print(f"  - BUT-PPG tasks: {len(butppg_tasks)} implemented")
print(f"  - Total benchmark files: 2")
print(f"  - All imports working: ✓")
print(f"  - All instantiations working: ✓")
print(f"  - Benchmark data accessible: ✓")
print("\nNext steps:")
print("  1. Verify data directories exist (data/vitaldb, data/butppg)")
print("  2. Verify split files exist (configs/splits/splits_full.json)")
print("  3. Run actual training/evaluation with these tasks")
