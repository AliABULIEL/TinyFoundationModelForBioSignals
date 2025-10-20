#!/usr/bin/env python3
"""
Quick test to understand BUT-PPG dataset structure.
"""
import wfdb
import numpy as np
from pathlib import Path

# Test loading recording 100001
record_id = "100001"
data_dir = Path("data/but_ppg/dataset")

print(f"Testing record: {record_id}")
print("=" * 80)

# Try loading PPG file
print("\n1. Loading PPG file:")
print(f"   Path: {data_dir / record_id / f'{record_id}_PPG'}")
try:
    ppg_record = wfdb.rdrecord(str(data_dir / record_id / f"{record_id}_PPG"))
    print(f"   ✓ Loaded successfully")
    print(f"   - Number of signals: {ppg_record.n_sig}")
    print(f"   - Signal names: {ppg_record.sig_name}")
    print(f"   - Sampling frequency: {ppg_record.fs} Hz")
    print(f"   - Signal shape: {ppg_record.p_signal.shape}")
    print(f"   - Duration: {ppg_record.sig_len / ppg_record.fs:.2f} seconds")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Try loading ECG file
print("\n2. Loading ECG file:")
print(f"   Path: {data_dir / record_id / f'{record_id}_ECG'}")
try:
    ecg_record = wfdb.rdrecord(str(data_dir / record_id / f"{record_id}_ECG"))
    print(f"   ✓ Loaded successfully")
    print(f"   - Number of signals: {ecg_record.n_sig}")
    print(f"   - Signal names: {ecg_record.sig_name}")
    print(f"   - Sampling frequency: {ecg_record.fs} Hz")
    print(f"   - Signal shape: {ecg_record.p_signal.shape}")
    print(f"   - Duration: {ecg_record.sig_len / ecg_record.fs:.2f} seconds")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Try loading as single record (what the script currently does)
print("\n3. Loading as single record (current script approach):")
print(f"   Path: {data_dir / record_id / record_id}")
try:
    record = wfdb.rdrecord(str(data_dir / record_id / record_id))
    print(f"   ✓ Loaded successfully")
    print(f"   - Number of signals: {record.n_sig}")
    print(f"   - Signal names: {record.sig_name}")
    print(f"   - Sampling frequency: {record.fs} Hz")
    print(f"   - Signal shape: {record.p_signal.shape}")
    print(f"   - Duration: {record.sig_len / record.fs:.2f} seconds")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("The dataset has separate PPG and ECG files.")
print("We need to load them separately and synchronize them.")
print("=" * 80)
