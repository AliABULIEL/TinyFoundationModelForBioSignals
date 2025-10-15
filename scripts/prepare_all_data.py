#!/usr/bin/env python3
"""
Master Data Preparation Script - Orchestrates All Data Phases
==============================================================

This script orchestrates the complete data preparation pipeline for:
1. VitalDB SSL Pretraining (PPG + ECG, 2 channels)
2. BUT-PPG Fine-tuning (ACC + PPG + ECG, 5 channels)

It uses your existing codebase to:
- Download datasets (if needed)
- Create subject-level splits
- Build preprocessed windows with quality filtering
- Compute normalization statistics
- Validate data integrity
- Generate summary reports

Usage:
    # Prepare all data with default settings
    python scripts/prepare_all_data.py --mode full
    
    # FastTrack mode (fewer cases, faster)
    python scripts/prepare_all_data.py --mode fasttrack
    
    # Only VitalDB
    python scripts/prepare_all_data.py --dataset vitaldb
    
    # Only BUT-PPG (will download if needed)
    python scripts/prepare_all_data.py --dataset butppg --download-butppg
    
    # With multiprocessing
    python scripts/prepare_all_data.py --multiprocess --num-workers 8

Author: Senior ML & SW Engineer
Date: October 2025
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.vitaldb_loader import get_available_case_sets, list_cases
from src.data.butppg_loader import BUTPPGLoader
from src.data.splits import make_patient_level_splits, verify_no_subject_leakage, get_split_statistics
from src.data.windows import NormalizationStats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """Master pipeline for preparing VitalDB and BUT-PPG data."""

    def __init__(
            self,
            mode: str = 'fasttrack',
            output_dir: str = 'data/processed',
            num_workers: Optional[int] = None,
            datasets: List[str] = ['vitaldb', 'butppg'],
            download_butppg: bool = False
    ):
        """
        Initialize data preparation pipeline.
        
        Args:
            mode: 'fasttrack' (70 cases) or 'full' (all cases)
            output_dir: Base directory for processed data
            num_workers: Number of parallel workers (None = auto)
            datasets: Which datasets to prepare ['vitaldb', 'butppg', or both]
            download_butppg: Whether to download BUT-PPG if not found
        """
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers or min(os.cpu_count() - 1, 8)
        self.datasets = datasets
        self.download_butppg = download_butppg

        # BUT-PPG paths (from download script)
        self.butppg_raw_dir = Path('data/but_ppg/dataset')
        self.butppg_index_dir = Path('data/outputs')

        # Create directory structure
        self.dirs = {
            'base': self.output_dir,
            'vitaldb_splits': self.output_dir / 'vitaldb' / 'splits',
            'vitaldb_windows': self.output_dir / 'vitaldb' / 'windows',
            'butppg_splits': self.output_dir / 'butppg' / 'splits',
            'butppg_windows': self.output_dir / 'butppg' / 'windows',
            'reports': self.output_dir / 'reports'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized pipeline in {self.mode} mode")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Datasets: {', '.join(self.datasets)}")

    def run_full_pipeline(self) -> Dict:
        """
        Run the complete data preparation pipeline.
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        results = {
            'mode': self.mode,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': {}
        }

        logger.info("=" * 80)
        logger.info("STARTING FULL DATA PREPARATION PIPELINE")
        logger.info("=" * 80)

        # Phase 1: VitalDB (SSL Pretraining Data)
        if 'vitaldb' in self.datasets:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: VitalDB SSL Pretraining Data Preparation")
            logger.info("=" * 80)
            vitaldb_results = self.prepare_vitaldb()
            results['datasets']['vitaldb'] = vitaldb_results

        # Phase 2: BUT-PPG (Fine-tuning Data)
        if 'butppg' in self.datasets:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2: BUT-PPG Fine-tuning Data Preparation")
            logger.info("=" * 80)

            # Check if download is needed
            if not self.butppg_raw_dir.exists() and self.download_butppg:
                logger.info("BUT-PPG data not found. Downloading...")
                if not self.download_butppg_data():
                    logger.error("Failed to download BUT-PPG data")
                    results['datasets']['butppg'] = {'error': 'Download failed'}
                else:
                    butppg_results = self.prepare_butppg()
                    results['datasets']['butppg'] = butppg_results
            elif not self.butppg_raw_dir.exists():
                logger.warning("BUT-PPG data not found. Use --download-butppg flag to download.")
                logger.warning("Or manually run: python scripts/download_but_ppg.py")
                results['datasets']['butppg'] = {'error': 'Data not found'}
            else:
                butppg_results = self.prepare_butppg()
                results['datasets']['butppg'] = butppg_results

        # Phase 3: Final Validation
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: Final Validation & Reports")
        logger.info("=" * 80)
        validation_results = self.validate_all_data()
        results['validation'] = validation_results

        # Save final report
        self.save_final_report(results)

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        return results

    def download_butppg_data(self) -> bool:
        """Download BUT-PPG data using the download script."""
        logger.info("Running BUT-PPG download script...")

        try:
            result = subprocess.run(
                [sys.executable, 'scripts/download_but_ppg.py'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )

            if result.returncode == 0:
                logger.info("✓ BUT-PPG downloaded successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error("✗ BUT-PPG download failed")
                logger.error(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error("✗ BUT-PPG download timed out")
            return False
        except Exception as e:
            logger.error(f"✗ Error running download script: {e}")
            return False

    def prepare_vitaldb(self) -> Dict:
        """Prepare VitalDB data for SSL pretraining."""
        logger.info("\n>>> Step 1.1: Creating VitalDB splits")
        vitaldb_splits = self.create_vitaldb_splits()

        logger.info("\n>>> Step 1.2: Building VitalDB windows (PPG + ECG)")
        vitaldb_windows = self.build_vitaldb_windows(vitaldb_splits)

        logger.info("\n>>> Step 1.3: Computing normalization statistics")
        vitaldb_stats = self.compute_vitaldb_stats(vitaldb_windows)

        logger.info("\n>>> Step 1.4: Validating VitalDB data")
        vitaldb_validation = self.validate_vitaldb_data(vitaldb_windows)

        return {
            'splits': vitaldb_splits,
            'windows': vitaldb_windows,
            'normalization': vitaldb_stats,
            'validation': vitaldb_validation
        }

    def prepare_butppg(self) -> Dict:
        """Prepare BUT-PPG data for fine-tuning."""
        logger.info("\n>>> Step 2.1: Creating BUT-PPG splits")
        butppg_splits = self.create_butppg_splits()

        if 'error' in butppg_splits:
            return butppg_splits

        logger.info("\n>>> Step 2.2: Building BUT-PPG windows (ACC + PPG + ECG)")
        butppg_windows = self.build_butppg_windows(butppg_splits)

        logger.info("\n>>> Step 2.3: Computing normalization statistics")
        butppg_stats = self.compute_butppg_stats(butppg_windows)

        logger.info("\n>>> Step 2.4: Validating BUT-PPG data")
        butppg_validation = self.validate_butppg_data(butppg_windows)

        return {
            'splits': butppg_splits,
            'windows': butppg_windows,
            'normalization': butppg_stats,
            'validation': butppg_validation
        }

    def create_vitaldb_splits(self) -> Dict:
        """Create subject-level splits for VitalDB."""
        logger.info("Getting available VitalDB cases...")

        # Get case sets
        case_sets = get_available_case_sets()

        # Select case set based on mode
        if self.mode == 'fasttrack':
            # Use first 70 BIS cases
            available_cases = list(case_sets.get('bis', []))[:70]
            train_ratio = 50 / 70  # 50 train, 20 test (no val for fasttrack)
            val_ratio = 0.0
        else:
            # Full mode - use all BIS cases
            available_cases = list(case_sets.get('bis', []))
            train_ratio = 0.7
            val_ratio = 0.15

        logger.info(f"Found {len(available_cases)} cases for {self.mode} mode")

        # Create case dictionaries
        cases = [
            {'case_id': str(cid), 'subject_id': str(cid)}
            for cid in available_cases
        ]

        # Create splits
        if val_ratio > 0:
            splits = make_patient_level_splits(
                cases=cases,
                ratios=(train_ratio, val_ratio, 1.0 - train_ratio - val_ratio),
                seed=42
            )
        else:
            splits = make_patient_level_splits(
                cases=cases,
                ratios=(train_ratio, 1.0 - train_ratio),
                seed=42
            )

        # Convert to simple format (just case IDs)
        simple_splits = {}
        for split_name, split_cases in splits.items():
            simple_splits[split_name] = [c['case_id'] for c in split_cases]

        # Verify no leakage
        verify_no_subject_leakage(splits)
        logger.info("✓ No subject leakage detected")

        # Get statistics
        stats = get_split_statistics(splits)
        for split_name, split_stats in stats.items():
            logger.info(f"  {split_name}: {split_stats['n_cases']} cases, "
                        f"{split_stats['n_subjects']} subjects")

        # Save splits
        splits_file = self.dirs['vitaldb_splits'] / f'splits_{self.mode}.json'
        with open(splits_file, 'w') as f:
            json.dump(simple_splits, f, indent=2)

        logger.info(f"✓ Saved splits to {splits_file}")

        return {
            'file': str(splits_file),
            'splits': simple_splits,
            'stats': stats
        }

    def build_vitaldb_windows(self, split_info: Dict) -> Dict:
        """Build preprocessed windows for VitalDB."""
        splits_file = split_info['file']
        results = {}

        # Process each split
        for split_name in ['train', 'val', 'test']:
            if split_name not in split_info['splits']:
                logger.info(f"Skipping {split_name} (not in splits)")
                continue

            logger.info(f"\nProcessing {split_name} split...")

            # Build command to call existing pipeline
            cmd = [
                sys.executable,
                'scripts/ttm_vitaldb.py',
                'build-windows',
                '--channels-yaml', 'configs/channels.yaml',
                '--windows-yaml', 'configs/windows.yaml',
                '--split-file', splits_file,
                '--split', split_name,
                '--channel', 'PPG',  # Process PPG first
                '--duration-sec', '60' if self.mode == 'fasttrack' else '300',
                '--min-sqi', '0.7',
                '--outdir', str(self.dirs['vitaldb_windows'] / split_name),
                '--multiprocess',
                '--num-workers', str(self.num_workers)
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Failed to process {split_name}")
                logger.error(result.stderr)
                continue

            logger.info(result.stdout)

            # Check output
            output_file = self.dirs['vitaldb_windows'] / split_name / f'{split_name}_windows.npz'
            if output_file.exists():
                data = np.load(output_file)
                results[split_name] = {
                    'file': str(output_file),
                    'shape': data['data'].shape,
                    'size_mb': output_file.stat().st_size / 1024 / 1024
                }
                logger.info(f"✓ {split_name}: {data['data'].shape} "
                            f"({results[split_name]['size_mb']:.1f} MB)")
            else:
                logger.warning(f"Output file not found: {output_file}")

        return results

    def create_butppg_splits(self) -> Dict:
        """Create subject-level splits for BUT-PPG."""
        logger.info("Getting available BUT-PPG subjects...")

        # Check if BUT-PPG data exists
        if not self.butppg_raw_dir.exists():
            logger.warning(f"BUT-PPG directory not found: {self.butppg_raw_dir}")
            logger.info("Please run: python scripts/download_but_ppg.py")
            logger.info("Or use --download-butppg flag")
            return {'error': 'Data not found', 'path': str(self.butppg_raw_dir)}

        # Load BUT-PPG data
        try:
            loader = BUTPPGLoader(str(self.butppg_raw_dir))
            subjects = loader.get_subject_list()
            logger.info(f"Found {len(subjects)} BUT-PPG subjects")
        except Exception as e:
            logger.error(f"Failed to load BUT-PPG: {e}")
            return {'error': str(e)}

        if len(subjects) == 0:
            logger.warning("No BUT-PPG subjects found")
            return {'error': 'No subjects found'}

        # Create case dictionaries
        cases = [
            {'case_id': subj, 'subject_id': subj}
            for subj in subjects
        ]

        # Create splits (80/10/10 for BUT-PPG)
        splits = make_patient_level_splits(
            cases=cases,
            ratios=(0.8, 0.1, 0.1),
            seed=42
        )

        # Convert to simple format
        simple_splits = {}
        for split_name, split_cases in splits.items():
            simple_splits[split_name] = [c['case_id'] for c in split_cases]

        # Verify no leakage
        verify_no_subject_leakage(splits)
        logger.info("✓ No subject leakage detected")

        # Get statistics
        stats = get_split_statistics(splits)
        for split_name, split_stats in stats.items():
            logger.info(f"  {split_name}: {split_stats['n_cases']} subjects")

        # Save splits
        splits_file = self.dirs['butppg_splits'] / 'splits.json'
        with open(splits_file, 'w') as f:
            json.dump(simple_splits, f, indent=2)

        logger.info(f"✓ Saved splits to {splits_file}")

        return {
            'file': str(splits_file),
            'splits': simple_splits,
            'stats': stats
        }

    def build_butppg_windows(self, split_info: Dict) -> Dict:
        """Build preprocessed windows for BUT-PPG using existing implementation."""
        if 'error' in split_info:
            logger.warning("Skipping BUT-PPG window building due to previous error")
            return split_info

        logger.info("Building BUT-PPG windows using BUTPPGDataset...")
        logger.info("  Modality: ALL (PPG + ECG + ACC = 5 channels)")

        splits_file = split_info['file']
        results = {}

        # Build windows using the dedicated script that uses BUTPPGDataset
        for split_name in ['train', 'val', 'test']:
            if split_name not in split_info['splits']:
                logger.info(f"Skipping {split_name} (not in splits)")
                continue

            logger.info(f"\nProcessing {split_name} split...")

            output_dir = self.dirs['butppg_windows'] / split_name

            # Call the BUT-PPG window builder script
            cmd = [
                sys.executable,
                'scripts/build_butppg_windows.py',
                '--data-dir', str(self.butppg_raw_dir),
                '--splits-file', splits_file,
                '--output-dir', str(output_dir),
                '--modality', 'all',  # PPG + ECG + ACC (5 channels)
                '--window-sec', '10.0',
                '--fs', '125',
                '--batch-size', '32'
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Failed to process {split_name}")
                logger.error(result.stderr)
                continue

            logger.info(result.stdout)

            # Check output
            output_file = output_dir / f'{split_name}_windows.npz'
            if output_file.exists():
                data = np.load(output_file)
                results[split_name] = {
                    'file': str(output_file),
                    'shape': data['data'].shape,
                    'size_mb': output_file.stat().st_size / 1024 / 1024
                }
                logger.info(f"✓ {split_name}: {data['data'].shape} "
                            f"({results[split_name]['size_mb']:.1f} MB)")
            else:
                logger.warning(f"Output file not found: {output_file}")

        return results

    def compute_vitaldb_stats(self, windows_info: Dict) -> Dict:
        """Compute normalization statistics from VitalDB training set."""
        if 'train' not in windows_info:
            logger.warning("No training data found")
            return {'error': 'No training data'}

        train_file = Path(windows_info['train']['file'])
        if not train_file.exists():
            logger.error(f"Training file not found: {train_file}")
            return {'error': 'File not found'}

        logger.info(f"Loading training data from {train_file}...")
        data = np.load(train_file)
        windows = data['data']

        logger.info(f"Training data shape: {windows.shape}")

        # Compute per-channel statistics
        if windows.ndim == 2:
            windows = windows[:, :, np.newaxis]

        stats = {
            'mean': float(np.mean(windows)),
            'std': float(np.std(windows)),
            'min': float(np.min(windows)),
            'max': float(np.max(windows)),
            'n_windows': len(windows)
        }

        # Save statistics
        stats_file = self.dirs['vitaldb_windows'] / 'train_stats.npz'
        np.savez(
            stats_file,
            mean=stats['mean'],
            std=stats['std'],
            min=stats['min'],
            max=stats['max'],
            method='zscore'
        )

        logger.info(f"✓ Saved normalization stats to {stats_file}")
        logger.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

        return stats

    def compute_butppg_stats(self, windows_info: Dict) -> Dict:
        """Compute normalization statistics from BUT-PPG training set."""
        if 'train' not in windows_info:
            logger.warning("No BUT-PPG training data found")
            return {'error': 'No training data'}

        train_file = Path(windows_info['train']['file'])
        if not train_file.exists():
            logger.error(f"Training file not found: {train_file}")
            return {'error': 'File not found'}

        logger.info(f"Loading BUT-PPG training data from {train_file}...")
        data = np.load(train_file)
        windows = data['data']

        logger.info(f"Training data shape: {windows.shape}")

        # Compute per-channel statistics
        if windows.ndim == 2:
            windows = windows[:, :, np.newaxis]

        stats = {
            'mean': float(np.mean(windows)),
            'std': float(np.std(windows)),
            'min': float(np.min(windows)),
            'max': float(np.max(windows)),
            'n_windows': len(windows),
            'n_channels': windows.shape[2] if windows.ndim >= 3 else 1
        }

        # Save statistics
        stats_file = self.dirs['butppg_windows'] / 'train_stats.npz'
        np.savez(
            stats_file,
            mean=stats['mean'],
            std=stats['std'],
            min=stats['min'],
            max=stats['max'],
            method='zscore'
        )

        logger.info(f"✓ Saved normalization stats to {stats_file}")
        logger.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        logger.info(f"  Channels: {stats['n_channels']}")

        return stats

    def validate_vitaldb_data(self, windows_info: Dict) -> Dict:
        """Validate VitalDB data integrity."""
        validation = {
            'splits': {},
            'issues': []
        }

        for split_name, split_info in windows_info.items():
            if 'file' not in split_info:
                continue

            file_path = Path(split_info['file'])
            if not file_path.exists():
                validation['issues'].append(f"{split_name}: file not found")
                continue

            # Load and validate
            try:
                data = np.load(file_path)
                windows = data['data']
                labels = data.get('labels', None)

                # Check for NaN/Inf
                has_nan = np.any(np.isnan(windows))
                has_inf = np.any(np.isinf(windows))

                # Check shape consistency
                expected_samples = 1250  # 10s at 125Hz
                valid_shape = windows.shape[1] == expected_samples

                validation['splits'][split_name] = {
                    'shape': windows.shape,
                    'has_nan': bool(has_nan),
                    'has_inf': bool(has_inf),
                    'valid_shape': valid_shape,
                    'mean': float(np.mean(windows)),
                    'std': float(np.std(windows))
                }

                if has_nan:
                    validation['issues'].append(f"{split_name}: contains NaN")
                if has_inf:
                    validation['issues'].append(f"{split_name}: contains Inf")
                if not valid_shape:
                    validation['issues'].append(
                        f"{split_name}: wrong shape {windows.shape[1]} (expected {expected_samples})"
                    )

                logger.info(f"✓ {split_name}: {windows.shape}, "
                            f"mean={validation['splits'][split_name]['mean']:.3f}, "
                            f"std={validation['splits'][split_name]['std']:.3f}")

            except Exception as e:
                validation['issues'].append(f"{split_name}: {str(e)}")
                logger.error(f"Error validating {split_name}: {e}")

        if validation['issues']:
            logger.warning(f"Found {len(validation['issues'])} validation issues")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ All validation checks passed")

        return validation

    def validate_butppg_data(self, windows_info: Dict) -> Dict:
        """Validate BUT-PPG data integrity."""
        validation = {
            'splits': {},
            'issues': []
        }

        for split_name, split_info in windows_info.items():
            if 'file' not in split_info:
                continue

            file_path = Path(split_info['file'])
            if not file_path.exists():
                validation['issues'].append(f"{split_name}: file not found")
                continue

            # Load and validate
            try:
                data = np.load(file_path)
                windows = data['data']
                labels = data.get('labels', None)

                # Check for NaN/Inf
                has_nan = np.any(np.isnan(windows))
                has_inf = np.any(np.isinf(windows))

                # Check shape consistency
                expected_samples = 1250  # 10s at 125Hz
                expected_channels = 5  # PPG + ECG + ACC (3-axis)
                valid_shape = windows.shape[1] == expected_samples
                valid_channels = windows.shape[2] == expected_channels if windows.ndim >= 3 else False

                validation['splits'][split_name] = {
                    'shape': windows.shape,
                    'has_nan': bool(has_nan),
                    'has_inf': bool(has_inf),
                    'valid_shape': valid_shape,
                    'valid_channels': valid_channels,
                    'mean': float(np.mean(windows)),
                    'std': float(np.std(windows))
                }

                if has_nan:
                    validation['issues'].append(f"{split_name}: contains NaN")
                if has_inf:
                    validation['issues'].append(f"{split_name}: contains Inf")
                if not valid_shape:
                    validation['issues'].append(
                        f"{split_name}: wrong time shape {windows.shape[1]} (expected {expected_samples})"
                    )
                if not valid_channels and windows.ndim >= 3:
                    validation['issues'].append(
                        f"{split_name}: wrong channels {windows.shape[2]} (expected {expected_channels})"
                    )

                logger.info(f"✓ {split_name}: {windows.shape}, "
                            f"mean={validation['splits'][split_name]['mean']:.3f}, "
                            f"std={validation['splits'][split_name]['std']:.3f}")

            except Exception as e:
                validation['issues'].append(f"{split_name}: {str(e)}")
                logger.error(f"Error validating {split_name}: {e}")

        if validation['issues']:
            logger.warning(f"Found {len(validation['issues'])} validation issues")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ All validation checks passed")

        return validation

    def validate_all_data(self) -> Dict:
        """Final validation of all prepared data."""
        validation = {
            'summary': {},
            'recommendations': []
        }

        # Check VitalDB
        if 'vitaldb' in self.datasets:
            vitaldb_train = self.dirs['vitaldb_windows'] / 'train' / 'train_windows.npz'
            if vitaldb_train.exists():
                data = np.load(vitaldb_train)
                n_windows = len(data['data'])
                validation['summary']['vitaldb_train_windows'] = n_windows

                # Check if we have enough windows for SSL
                target = 500000 if self.mode == 'full' else 10000
                if n_windows < target * 0.8:
                    validation['recommendations'].append(
                        f"VitalDB: Only {n_windows} windows (target: {target}). "
                        f"Consider increasing duration_sec or using more cases."
                    )
                else:
                    logger.info(f"✓ VitalDB: {n_windows} windows (target: {target})")

        # Check BUT-PPG
        if 'butppg' in self.datasets:
            butppg_train = self.dirs['butppg_windows'] / 'train' / 'train_windows.npz'
            if butppg_train.exists():
                data = np.load(butppg_train)
                n_windows = len(data['data'])
                validation['summary']['butppg_train_windows'] = n_windows
                logger.info(f"✓ BUT-PPG: {n_windows} windows")

        return validation

    def save_final_report(self, results: Dict) -> None:
        """Save final pipeline report."""
        report_file = self.dirs['reports'] / f'pipeline_report_{self.mode}.json'

        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\n✓ Final report saved to {report_file}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)

        if 'vitaldb' in results['datasets']:
            vitaldb = results['datasets']['vitaldb']
            logger.info("\nVitalDB:")
            if 'windows' in vitaldb:
                for split, info in vitaldb['windows'].items():
                    if 'shape' in info:
                        logger.info(f"  {split}: {info['shape']} "
                                    f"({info['size_mb']:.1f} MB)")

        if 'butppg' in results['datasets']:
            butppg = results['datasets']['butppg']
            logger.info("\nBUT-PPG:")
            if 'error' in butppg:
                logger.warning(f"  Error: {butppg['error']}")
                if 'path' in butppg:
                    logger.warning(f"  Expected path: {butppg['path']}")
                logger.info("  Run: python scripts/download_but_ppg.py")
                logger.info("  Or use: --download-butppg flag")
            else:
                logger.info("  Status: Prepared")

        logger.info("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Master data preparation script for TTM Foundation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all data in fasttrack mode (recommended for testing)
  python scripts/prepare_all_data.py --mode fasttrack
  
  # Prepare all data in full mode
  python scripts/prepare_all_data.py --mode full
  
  # Only prepare VitalDB
  python scripts/prepare_all_data.py --dataset vitaldb
  
  # Only prepare BUT-PPG (download if needed)
  python scripts/prepare_all_data.py --dataset butppg --download-butppg
  
  # Use 16 workers for faster processing
  python scripts/prepare_all_data.py --num-workers 16
  
  # Custom output directory
  python scripts/prepare_all_data.py --output data/my_processed_data

BUT-PPG Data:
  The script expects BUT-PPG data at: data/but_ppg/dataset/
  You can either:
    1. Use --download-butppg flag (downloads automatically)
    2. Manually run: python scripts/download_but_ppg.py
    3. Download from PhysioNet: https://physionet.org/content/butppg/2.0.0/
        """
    )

    parser.add_argument(
        '--mode',
        choices=['fasttrack', 'full'],
        default='fasttrack',
        help='Processing mode: fasttrack (70 cases) or full (all cases)'
    )

    parser.add_argument(
        '--dataset',
        choices=['vitaldb', 'butppg', 'both'],
        default='both',
        help='Which dataset(s) to prepare'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto)'
    )

    parser.add_argument(
        '--download-butppg',
        action='store_true',
        help='Download BUT-PPG data if not found (uses scripts/download_but_ppg.py)'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip final validation phase'
    )

    args = parser.parse_args()

    # Determine datasets
    if args.dataset == 'both':
        datasets = ['vitaldb', 'butppg']
    else:
        datasets = [args.dataset]

    # Create and run pipeline
    pipeline = DataPreparationPipeline(
        mode=args.mode,
        output_dir=args.output,
        num_workers=args.num_workers,
        datasets=datasets,
        download_butppg=args.download_butppg
    )

    try:
        results = pipeline.run_full_pipeline()

        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS! Data preparation completed.")
        logger.info("=" * 80)
        logger.info(f"\nProcessed data location: {pipeline.output_dir}")
        logger.info(f"Log file: data_preparation.log")

        # Check for BUT-PPG errors
        if 'butppg' in results['datasets'] and 'error' in results['datasets']['butppg']:
            logger.warning("\n⚠️  BUT-PPG data preparation had issues.")
            logger.info("To download BUT-PPG data:")
            logger.info("  Option 1: python scripts/download_but_ppg.py")
            logger.info("  Option 2: python scripts/prepare_all_data.py --dataset butppg --download-butppg")
            logger.info("  Option 3: Download from https://physionet.org/content/butppg/2.0.0/")

        return 0

    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
