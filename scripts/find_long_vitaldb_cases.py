#!/usr/bin/env python3
"""
Find VitalDB cases with long recordings.

This script queries VitalDB metadata to identify cases with sufficient duration
for SSL pre-training (>10 minutes).
"""

import ssl
import os
import sys

# Configure SSL
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    except ImportError:
        pass
except:
    pass

try:
    import vitaldb
    print("✓ VitalDB loaded successfully")
except ImportError:
    print("❌ VitalDB not available. Install with: pip install vitaldb")
    sys.exit(1)

print("\n" + "="*80)
print("🔍 Finding VitalDB cases with long recordings")
print("="*80)

# VitalDB doesn't provide a direct API to list all cases with metadata
# We'll need to check a range of case IDs
print("\nChecking case range 1-6500 for track availability and duration...")
print("This may take a few minutes...")

suitable_cases = []
checked = 0
errors = 0

# Check every 10th case for speed (sample)
for case_id in range(1, 6500, 10):
    try:
        # Try to get track list
        vf = vitaldb.VitalFile(case_id)
        trks = vf.get_tracknames()

        if not trks:
            continue

        has_pleth = 'SNUADC/PLETH' in trks
        has_ecg = 'SNUADC/ECG_II' in trks

        if has_pleth and has_ecg:
            # Get duration by checking track info
            # VitalDB tracks have .recs attribute with number of records
            try:
                # Get a track to check duration
                pleth_info = vf.get_track_info('SNUADC/PLETH')
                if pleth_info:
                    # Duration in seconds (recs / sample_rate)
                    nrecs = pleth_info.get('recs', 0)
                    fs = pleth_info.get('fs', 100)  # Default 100 Hz for PLETH
                    duration_sec = nrecs / fs if fs > 0 else 0
                    duration_min = duration_sec / 60.0

                    suitable_cases.append({
                        'case_id': case_id,
                        'duration_min': duration_min,
                        'duration_sec': duration_sec
                    })
            except:
                pass

        checked += 1
        if checked % 100 == 0:
            print(f"  Checked {checked} cases, found {len(suitable_cases)} suitable...")

    except Exception as e:
        errors += 1
        if errors > 100:
            print(f"\n⚠️  Too many errors, stopping check")
            break
        continue

print(f"✓ Found {len(suitable_cases)} cases with both PLETH and ECG_II")

# Sort by duration
suitable_cases.sort(key=lambda x: x['duration_min'], reverse=True)

# Show statistics
durations = [c['duration_min'] for c in suitable_cases]
import numpy as np

print("\n" + "="*80)
print("📊 DURATION STATISTICS")
print("="*80)
print(f"Mean duration: {np.mean(durations):.1f} minutes")
print(f"Median duration: {np.median(durations):.1f} minutes")
print(f"Min duration: {np.min(durations):.1f} minutes")
print(f"Max duration: {np.max(durations):.1f} minutes")

# Show cases by duration range
print("\n" + "="*80)
print("📊 CASES BY DURATION")
print("="*80)

ranges = [
    (0, 1, "< 1 min (very short)"),
    (1, 5, "1-5 min (short)"),
    (5, 10, "5-10 min (medium)"),
    (10, 30, "10-30 min (good)"),
    (30, 60, "30-60 min (long)"),
    (60, 120, "1-2 hours (very long)"),
    (120, 999, "> 2 hours (surgical)"),
]

for min_dur, max_dur, label in ranges:
    count = sum(1 for c in suitable_cases if min_dur <= c['duration_min'] < max_dur)
    pct = count / len(suitable_cases) * 100
    print(f"{label:25s}: {count:5d} cases ({pct:5.1f}%)")

# Show top 20 longest cases
print("\n" + "="*80)
print("🏆 TOP 20 LONGEST CASES")
print("="*80)
print(f"{'Case ID':>10s} {'Duration (min)':>15s} {'Duration (hours)':>18s}")
print("-"*50)

for case in suitable_cases[:20]:
    print(f"{case['case_id']:10d} {case['duration_min']:15.1f} {case['duration_min']/60:18.2f}")

# Find good cases for SSL (>10 min)
long_cases = [c for c in suitable_cases if c['duration_min'] >= 10]

print("\n" + "="*80)
print(f"💡 RECOMMENDATION")
print("="*80)
print(f"Cases with ≥10 minutes: {len(long_cases)} ({len(long_cases)/len(suitable_cases)*100:.1f}%)")

if len(long_cases) > 0:
    print(f"\nTop 50 case IDs for SSL pre-training:")
    case_ids = [c['case_id'] for c in long_cases[:50]]

    # Print in groups of 10 for readability
    for i in range(0, min(50, len(case_ids)), 10):
        batch = case_ids[i:i+10]
        print(f"  {batch}")

    # Save to file
    output_file = "data/vitaldb_long_cases.txt"
    with open(output_file, 'w') as f:
        for c in long_cases:
            f.write(f"{c['case_id']},{c['duration_min']:.1f}\n")

    print(f"\n✓ Full list saved to: {output_file}")
    print(f"  Format: case_id,duration_minutes")

print("\n" + "="*80)
print("✅ Done!")
print("="*80)
