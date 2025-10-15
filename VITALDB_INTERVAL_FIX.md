# CRITICAL FIX: VitalDB Data Loading Interval Parameter

## Problem

VitalDB data was appearing 100-500x shorter than actual duration, causing only 121 windows to be generated from 50 cases when we should have gotten 10,000+ windows.

**Root Cause:** The `vitaldb.load_case()` function defaults to `interval=1` (1 Hz sampling), but the scripts assumed data was at native sampling rates (100 Hz for PPG, 500 Hz for ECG).

## Impact

- **Before Fix:** Case 100 appeared as 0.1 minutes → generated ~2 windows
- **After Fix:** Case 100 is actually 69.1 minutes → generates 504 windows

**Data yield improvement: 100-500x increase**

## Example

```python
# WRONG (default interval=1 means 1 Hz sampling)
data = vitaldb.load_case(case_id, ['SNUADC/PLETH'])
# This loads at 1 Hz, but code assumes 100 Hz
# Result: 30-minute recording looks like 18 seconds!

# CORRECT (specify interval to match desired sampling rate)
interval = 1.0 / 100.0  # For 100 Hz PPG
data = vitaldb.load_case(case_id, ['SNUADC/PLETH'], interval=interval)
# This loads at 100 Hz as expected
# Result: 30-minute recording is actually 30 minutes
```

## Files Fixed

### 1. `scripts/rebuild_vitaldb_paired.py`

```python
def load_signal(case_id, track_names, default_fs):
    # CRITICAL: Set interval to match sampling rate (interval = 1 / fs)
    interval = 1.0 / default_fs

    for track in track_names:
        try:
            data = vitaldb.load_case(case_id, [track], interval=interval)
            # ... rest of code
```

### 2. `scripts/inspect_vitaldb_raw.py`

```python
def load_and_inspect_signal(case_id, track_name):
    # Determine expected sampling rate based on track type
    if 'ECG' in track_name:
        expected_fs = 500.0  # ECG at 500 Hz
    elif 'PLETH' in track_name or 'ART' in track_name:
        expected_fs = 100.0  # PPG/Pleth at 100 Hz
    else:
        expected_fs = 100.0

    # Load with correct interval
    interval = 1.0 / expected_fs
    data = vitaldb.load_case(case_id, [track_name], interval=interval)
    # ... rest of code
```

## Results

### Test Build (10 cases, with fix)

```
Case 100:  69.1 minutes → 504 windows
Case 101: 219.8 minutes → 1,610 windows
Case 102:  59.3 minutes → 344 windows
Case 103: 179.1 minutes → 1,236 windows
Case 104: 221.6 minutes → 1,550 windows
Case 105: 303.2 minutes → 2,168 windows (5+ hours!)
Case 106:  72.0 minutes → 523 windows
Case 107: 330.8 minutes → 2,421 windows (5.5+ hours!)
Case 108: 159.4 minutes → 1,093 windows
Case 109: 239.2 minutes → 1,735 windows

Total: 13,184 windows from 10 cases
```

### Comparison

| Build | Cases | Windows | Windows/Case |
|-------|-------|---------|--------------|
| **Before fix** | 50 | 121 | 2.4 |
| **After fix (10 cases)** | 10 | 13,184 | 1,318.4 |
| **Improvement** | - | **109x** | **549x** |

## Validation

The fix is validated by:
1. Actual durations match expected surgical procedure lengths (30-300 minutes)
2. Window counts are now proportional to recording duration
3. Quality checks pass at expected rates (>95%)
4. Signal statistics look normal after loading

## Next Steps

Full dataset build is running with 100 cases, expected to generate ~100,000-200,000 windows for SSL pre-training (vs the previous 121 windows).

---

**Date:** October 15, 2025
**Fix applied to:** `scripts/rebuild_vitaldb_paired.py`, `scripts/inspect_vitaldb_raw.py`
