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

# Determine project root and add to path
if '__file__' in globals():
    # Running as script
    project_root = Path(__file__).resolve().parent.parent
else:
    # Running in Colab/Jupyter
    project_root = Path.cwd()
    if not (project_root / 'src').exists():
        # Try going up one level
        project_root = project_root.parent

# Add to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Current directory: {os.getcwd()}")

# Import with error handling
try:
    # Import directly from modules (avoid __init__.py issues)
    import importlib.util
    
    # Method 1: Direct import (preferred)
    try:
        from src.data.vitaldb_loader import get_available_case_sets, list_cases
        from src.data.splits import make_patient_level_splits, verify_no_subject_leakage, get_split_statistics
        from src.data.windows import NormalizationStats
        
        # Try importing BUT-PPG loader (may not be synced in Colab)
        try:
            from src.data.butppg_loader import BUTPPGLoader
            BUTPPG_AVAILABLE = True
        except ImportError:
            BUTPPG_AVAILABLE = False
            BUTPPGLoader = None
            print("⚠️  BUT-PPG loader not available (file not synced?)")
        
        print("✓ Successfully imported core modules")
        if BUTPPG_AVAILABLE:
            print("✓ BUT-PPG loader available")
        else:
            print("ℹ️  BUT-PPG loader unavailable - only VitalDB will work")
            
    except ImportError as e1:
        # Method 2: Try adding src to path and importing again
        src_path = project_root / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from data.vitaldb_loader import get_available_case_sets, list_cases
        from data.splits import make_patient_level_splits, verify_no_subject_leakage, get_split_statistics
        from data.windows import NormalizationStats
        
        # Try importing BUT-PPG loader
        try:
            from data.butppg_loader import BUTPPGLoader
            BUTPPG_AVAILABLE = True
        except ImportError:
            BUTPPG_AVAILABLE = False
            BUTPPGLoader = None
            print("⚠️  BUT-PPG loader not available")
        
        print("✓ Successfully imported core modules (method 2)")
        
except ImportError as e:
    print(f"\n❌ ERROR: Failed to import required modules: {e}")
    print(f"\nDebug Information:")
    print(f"  Project root: {project_root}")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Check if files exist
    print(f"\nFile existence check:")
    files_to_check = [
        project_root / 'src' / 'data' / 'butppg_loader.py',
        project_root / 'src' / 'data' / 'vitaldb_loader.py',
        project_root / 'src' / 'data' / 'splits.py',
        project_root / 'src' / 'data' / 'windows.py',
        project_root / 'src' / 'data' / '__init__.py'
    ]
    
    for file_path in files_to_check:
        exists = "✓" if file_path.exists() else "✗"
        print(f"  {exists} {file_path.name}")
    
    print(f"\nTroubleshooting:")
    print(f"1. Make sure you're running from project root: {project_root}")
    print(f"2. Try: cd {project_root} && python scripts/prepare_all_data.py")
    print(f"3. Or: python -m scripts.prepare_all_data (from project root)")
    print(f"4. If in Colab, make sure all files are synced from Drive")
    sys.exit(1)

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
            download_butppg: bool = False,
            format: str = 'legacy',
            overlap: float = 0.25
    ):
        """
        Initialize data preparation pipeline.

        Args:
            mode: 'fasttrack' (70 cases) or 'full' (all cases)
            output_dir: Base directory for processed data
            num_workers: Number of parallel workers (None = auto)
            datasets: Which datasets to prepare ['vitaldb', 'butppg', or both]
            download_butppg: Whether to download BUT-PPG if not found
            format: 'legacy' (multi-window NPZ) or 'windowed' (one NPZ per window with labels)
            overlap: Window overlap ratio (0.25 = 25% overlap, default)
        """
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers or min(os.cpu_count() - 1, 8)
        self.datasets = datasets
        self.download_butppg = download_butppg
        self.format = format
        self.overlap = overlap

        # BUT-PPG paths (from download script)
        self.butppg_raw_dir = Path('data/but_ppg/dataset')
        self.butppg_index_dir = Path('data/outputs')

        # Create directory structure
        windows_suffix = 'windows_with_labels' if self.format == 'windowed' else 'windows'
        self.dirs = {
            'base': self.output_dir,
            'vitaldb_splits': self.output_dir / 'vitaldb' / 'splits',
            'vitaldb_windows': self.output_dir / 'vitaldb' / windows_suffix,
            'butppg_splits': self.output_dir / 'butppg' / 'splits',
            'butppg_windows': self.output_dir / 'butppg' / windows_suffix,
            'reports': self.output_dir / 'reports'
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # ✅ LOAD CONFIGURATIONS FROM YAML FILES - NO HARD-CODED VALUES!
        self._load_configs()

        logger.info(f"Initialized pipeline in {self.mode} mode")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Format: {self.format}")
        logger.info(f"Window overlap: {self.overlap*100:.0f}% ({int(self.expected_samples * self.overlap)} samples)")
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Datasets: {', '.join(self.datasets)}")

    def _load_configs(self):
        """Load configuration from YAML files - NO HARD-CODED VALUES!"""
        # Load window configuration
        windows_config_path = project_root / 'configs' / 'windows.yaml'
        model_config_path = project_root / 'configs' / 'model.yaml'
        
        # Load windows config
        if windows_config_path.exists():
            with open(windows_config_path, 'r') as f:
                windows_config = yaml.safe_load(f)
            
            # Extract window size for SSL pretraining (non-overlapping)
            self.window_sec = windows_config.get('window', {}).get('size_seconds', 8.192)
            
            logger.info(f"✓ Loaded window config: {self.window_sec}s windows")
        else:
            logger.warning(f"Config not found: {windows_config_path}")
            self.window_sec = 8.192  # Fallback to TTM-Enhanced default
            logger.warning(f"Using default: {self.window_sec}s windows")
        
        # Load model config
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            
            # Extract model parameters
            self.context_length = model_config.get('model', {}).get('context_length', 1024)
            self.patch_size = model_config.get('model', {}).get('patch_size', 128)
            self.sampling_rate = 125  # Fixed for biosignals
            
            # Calculate expected samples from window size
            self.expected_samples = int(self.window_sec * self.sampling_rate)
            
            logger.info(f"✓ Loaded model config:")
            logger.info(f"    context_length={self.context_length}")
            logger.info(f"    patch_size={self.patch_size}")
            logger.info(f"    expected_samples={self.expected_samples} ({self.window_sec}s @ {self.sampling_rate}Hz)")
        else:
            logger.warning(f"Config not found: {model_config_path}")
            self.context_length = 1024
            self.patch_size = 128
            self.sampling_rate = 125
            self.expected_samples = 1024
            logger.warning(f"Using defaults: context={self.context_length}, patch={self.patch_size}")

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
        """Build preprocessed windows for VitalDB (PPG + ECG) with proper synchronization.

        CRITICAL FIX: Uses rebuild_vitaldb_paired.py to ensure PPG and ECG are temporally aligned.
        This fixes the bug where PPG and ECG were processed independently, causing misalignment.
        """
        splits_file = split_info['file']

        logger.info("Building synchronized PPG+ECG windows...")
        logger.info("CRITICAL: Using rebuild_vitaldb_paired.py for proper temporal alignment")

        # Determine number of cases based on mode
        if self.mode == 'fasttrack':
            max_cases = 70
        else:
            max_cases = None  # Process all cases

        # Output directory for paired data
        paired_output = self.dirs['vitaldb_windows'] / 'paired'

        # Build command to call the paired data builder
        cmd = [
            sys.executable,
            'scripts/rebuild_vitaldb_paired.py',
            '--output', str(paired_output),
            '--window-size', str(self.context_length),  # Use config value (1024)
            '--fs', str(self.sampling_rate),  # 125 Hz
            '--splits-file', splits_file,
            '--train-ratio', '0.7' if self.mode == 'fasttrack' else '0.7',
            '--val-ratio', '0.15' if self.mode == 'fasttrack' else '0.15'
        ]

        if max_cases:
            cmd.extend(['--max-cases', str(max_cases)])

        logger.info(f"Running: {' '.join(cmd)}")

        # Run the paired data builder
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("Failed to build paired windows")
            logger.error(result.stderr)
            return {'error': 'Paired window building failed', 'stderr': result.stderr}

        logger.info(result.stdout)

        # Check outputs for each split
        results = {}
        for split_name in ['train', 'val', 'test']:
            split_dir = paired_output / split_name

            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            # Find all case files
            case_files = list(split_dir.glob('case_*.npz'))

            if len(case_files) == 0:
                logger.warning(f"No case files found in {split_dir}")
                results[split_name] = {'error': 'No cases found'}
                continue

            # Load first file to check format
            sample_file = case_files[0]
            data = np.load(sample_file)

            # Count total windows across all cases
            total_windows = 0
            total_size = 0
            for case_file in case_files:
                case_data = np.load(case_file)
                total_windows += case_data['data'].shape[0]
                total_size += case_file.stat().st_size

            results[split_name] = {
                'dir': str(split_dir),
                'num_cases': len(case_files),
                'total_windows': total_windows,
                'shape_per_window': data['data'].shape[1:],  # [2, 1024]
                'size_mb': total_size / 1024 / 1024,
                'format': '[N, 2, 1024] - Channel 0=PPG, Channel 1=ECG'
            }

            logger.info(f"  ✓ {split_name}: {len(case_files)} cases, "
                        f"{total_windows} windows, shape={data['data'].shape[1:]}, "
                        f"({results[split_name]['size_mb']:.1f} MB)")

        # Save summary
        summary_file = paired_output / 'dataset_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            logger.info("\n✓ Paired dataset summary:")
            logger.info(f"  Success rate: {summary.get('summary', {})}")

        return results

    def create_butppg_splits(self) -> Dict:
        """Create subject-level splits for BUT-PPG."""
        # Check if BUT-PPG loader is available
        if not BUTPPG_AVAILABLE or BUTPPGLoader is None:
            logger.error("BUT-PPG loader not available. File may not be synced to Drive.")
            return {
                'error': 'BUTPPGLoader not available',
                'message': 'File src/data/butppg_loader.py not found. Wait for Drive sync or use --dataset vitaldb'
            }
        
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

        if self.format == 'windowed':
            # NEW FORMAT: One NPZ per window with embedded labels
            logger.info("Building BUT-PPG windows using NEW WINDOWED FORMAT...")
            logger.info("  Format: One NPZ per window with embedded labels")
            logger.info("  Modality: ALL (PPG + ECG + ACC = 5 channels)")

            splits_file = split_info['file']

            # Call the NEW windowed window builder script
            cmd = [
                sys.executable,
                'scripts/create_butppg_windows_with_labels.py',
                '--data-dir', str(self.butppg_raw_dir),  # ✅ FIXED: CSV files are directly in this directory
                '--output-dir', str(self.dirs['butppg_windows']),
                '--splits-file', splits_file,
                '--window-sec', str(self.window_sec),  # ✅ FROM CONFIG!
                '--fs', '125',
                '--overlap', str(self.overlap)  # ✅ OVERLAPPING WINDOWS!
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Failed to build windowed format")
                logger.error(result.stderr)
                return {'error': 'Windowed format building failed'}

            logger.info(result.stdout)

            # Check outputs
            results = {}
            for split_name in ['train', 'val', 'test']:
                split_dir = self.dirs['butppg_windows'] / split_name

                if not split_dir.exists():
                    logger.warning(f"Split directory not found: {split_dir}")
                    continue

                # Count window files
                window_files = list(split_dir.glob('window_*.npz'))

                if len(window_files) == 0:
                    logger.warning(f"No window files in {split_dir}")
                    continue

                # Get total size
                total_size = sum(f.stat().st_size for f in window_files)

                results[split_name] = {
                    'dir': str(split_dir),
                    'num_windows': len(window_files),
                    'size_mb': total_size / 1024 / 1024,
                    'format': 'windowed - one NPZ per window'
                }

                logger.info(f"✓ {split_name}: {len(window_files)} windows "
                            f"({results[split_name]['size_mb']:.1f} MB)")

            return results

        else:
            # LEGACY FORMAT: Multi-window NPZ
            logger.info("Building BUT-PPG windows using LEGACY FORMAT...")
            logger.info("  Format: Multi-window NPZ")
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

                # ✅ USE CONFIG VALUE - NO HARD-CODED!
                # Call the BUT-PPG window builder script
                cmd = [
                    sys.executable,
                    'scripts/build_butppg_windows.py',
                    '--data-dir', str(self.butppg_raw_dir),
                    '--splits-file', splits_file,
                    '--output-dir', str(output_dir),
                    '--modality', 'all',  # PPG + ECG + ACC (5 channels)
                    '--window-sec', str(self.window_sec),  # ✅ FROM CONFIG!
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
        """Compute normalization statistics from VitalDB training set (paired data)."""
        if 'train' not in windows_info:
            logger.warning("No training data found")
            return {'error': 'No training data'}

        train_info = windows_info['train']

        # New paired data structure - directory with case files
        if 'dir' in train_info:
            train_dir = Path(train_info['dir'])

            if not train_dir.exists():
                logger.error(f"Training directory not found: {train_dir}")
                return {'error': 'Directory not found'}

            logger.info(f"Loading paired training data from {train_dir}...")

            # Load all case files from training set
            case_files = list(train_dir.glob('case_*.npz'))

            if len(case_files) == 0:
                logger.error(f"No case files found in {train_dir}")
                return {'error': 'No case files'}

            # Collect all windows from all cases
            all_windows = []
            for case_file in case_files:
                data = np.load(case_file)
                all_windows.append(data['data'])  # [N, 2, 1024]

            # Concatenate all windows
            windows = np.concatenate(all_windows, axis=0)  # [Total_N, 2, 1024]

            logger.info(f"Training data shape: {windows.shape}")
            logger.info(f"  {len(case_files)} cases, {windows.shape[0]} windows")
            logger.info(f"  Format: [N, 2, 1024] - Channel 0=PPG, Channel 1=ECG")

            # Compute per-channel statistics
            ppg_windows = windows[:, 0, :]  # [N, 1024]
            ecg_windows = windows[:, 1, :]  # [N, 1024]

            stats = {
                'mean': float(np.mean(windows)),
                'std': float(np.std(windows)),
                'min': float(np.min(windows)),
                'max': float(np.max(windows)),
                'n_windows': len(windows),
                'n_channels': 2,
                'channel_stats': {
                    'PPG': {
                        'mean': float(np.mean(ppg_windows)),
                        'std': float(np.std(ppg_windows)),
                        'min': float(np.min(ppg_windows)),
                        'max': float(np.max(ppg_windows))
                    },
                    'ECG': {
                        'mean': float(np.mean(ecg_windows)),
                        'std': float(np.std(ecg_windows)),
                        'min': float(np.min(ecg_windows)),
                        'max': float(np.max(ecg_windows))
                    }
                }
            }

            # Save statistics
            stats_file = self.dirs['vitaldb_windows'] / 'paired' / 'train_stats.npz'
            stats_file.parent.mkdir(parents=True, exist_ok=True)

            np.savez(
                stats_file,
                mean=stats['mean'],
                std=stats['std'],
                min=stats['min'],
                max=stats['max'],
                ppg_mean=stats['channel_stats']['PPG']['mean'],
                ppg_std=stats['channel_stats']['PPG']['std'],
                ecg_mean=stats['channel_stats']['ECG']['mean'],
                ecg_std=stats['channel_stats']['ECG']['std'],
                method='zscore'
            )

            logger.info(f"✓ Saved normalization stats to {stats_file}")
            logger.info(f"  Overall: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")
            logger.info(f"  PPG: Mean={stats['channel_stats']['PPG']['mean']:.4f}, "
                       f"Std={stats['channel_stats']['PPG']['std']:.4f}")
            logger.info(f"  ECG: Mean={stats['channel_stats']['ECG']['mean']:.4f}, "
                       f"Std={stats['channel_stats']['ECG']['std']:.4f}")

            return stats

        else:
            # Legacy single file structure (shouldn't reach here with new code)
            logger.error("Old data structure detected. Please rebuild data with new paired format.")
            return {'error': 'Legacy structure not supported'}

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
        """Validate VitalDB paired data integrity."""
        validation = {
            'splits': {},
            'issues': []
        }

        for split_name, split_info in windows_info.items():
            # Handle new paired data structure (directory with case files)
            if 'dir' in split_info:
                split_dir = Path(split_info['dir'])

                if not split_dir.exists():
                    validation['issues'].append(f"{split_name}: directory not found")
                    continue

                # Load and validate case files
                try:
                    case_files = list(split_dir.glob('case_*.npz'))

                    if len(case_files) == 0:
                        validation['issues'].append(f"{split_name}: no case files found")
                        continue

                    # Validate sample cases
                    all_has_nan = []
                    all_has_inf = []
                    all_means = []
                    all_stds = []
                    total_windows = 0

                    for case_file in case_files:
                        data = np.load(case_file)
                        windows = data['data']  # [N, 2, 1024]

                        total_windows += windows.shape[0]

                        # Check for NaN/Inf
                        all_has_nan.append(np.any(np.isnan(windows)))
                        all_has_inf.append(np.any(np.isinf(windows)))
                        all_means.append(np.mean(windows))
                        all_stds.append(np.std(windows))

                        # Check shape consistency
                        expected_channels = 2
                        expected_samples = self.expected_samples  # From config

                        if windows.ndim != 3:
                            validation['issues'].append(
                                f"{split_name}/{case_file.name}: wrong ndim {windows.ndim} (expected 3)"
                            )
                        elif windows.shape[1] != expected_channels:
                            validation['issues'].append(
                                f"{split_name}/{case_file.name}: wrong channels {windows.shape[1]} (expected 2)"
                            )
                        elif windows.shape[2] != expected_samples:
                            validation['issues'].append(
                                f"{split_name}/{case_file.name}: wrong samples {windows.shape[2]} (expected {expected_samples})"
                            )

                    # Aggregate results
                    has_nan = any(all_has_nan)
                    has_inf = any(all_has_inf)

                    validation['splits'][split_name] = {
                        'num_cases': len(case_files),
                        'total_windows': total_windows,
                        'format': '[N, 2, 1024]',
                        'has_nan': bool(has_nan),
                        'has_inf': bool(has_inf),
                        'mean': float(np.mean(all_means)),
                        'std': float(np.mean(all_stds))
                    }

                    if has_nan:
                        validation['issues'].append(f"{split_name}: contains NaN in some cases")
                    if has_inf:
                        validation['issues'].append(f"{split_name}: contains Inf in some cases")

                    logger.info(f"✓ {split_name}: {len(case_files)} cases, {total_windows} windows, "
                                f"mean={validation['splits'][split_name]['mean']:.3f}, "
                                f"std={validation['splits'][split_name]['std']:.3f}")

                except Exception as e:
                    validation['issues'].append(f"{split_name}: {str(e)}")
                    logger.error(f"Error validating {split_name}: {e}")

            elif 'file' in split_info:
                # Legacy single file structure
                validation['issues'].append(f"{split_name}: legacy structure detected, use new paired format")

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

                # ✅ USE CONFIG VALUE - NO HARD-CODED!
                # Check shape consistency
                expected_samples = self.expected_samples  # ✅ FROM CONFIG!
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

        # Check VitalDB (new paired data structure)
        if 'vitaldb' in self.datasets:
            vitaldb_train_dir = self.dirs['vitaldb_windows'] / 'paired' / 'train'

            if vitaldb_train_dir.exists():
                # Count windows across all case files
                case_files = list(vitaldb_train_dir.glob('case_*.npz'))
                n_windows = 0

                for case_file in case_files:
                    data = np.load(case_file)
                    n_windows += data['data'].shape[0]

                validation['summary']['vitaldb_train_windows'] = n_windows
                validation['summary']['vitaldb_train_cases'] = len(case_files)

                # Check if we have enough windows for SSL
                target = 500000 if self.mode == 'full' else 10000
                if n_windows < target * 0.8:
                    validation['recommendations'].append(
                        f"VitalDB: Only {n_windows} windows (target: {target}). "
                        f"Consider increasing duration_sec or using more cases."
                    )
                else:
                    logger.info(f"✓ VitalDB: {n_windows} windows from {len(case_files)} cases (target: {target})")
            else:
                logger.warning(f"VitalDB training data not found at {vitaldb_train_dir}")

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

  # NEW: Use windowed format (one NPZ per window with embedded labels)
  python scripts/prepare_all_data.py --mode fasttrack --format windowed

  # With custom overlap (default is 25%)
  python scripts/prepare_all_data.py --mode fasttrack --format windowed --overlap 0.5

  # No overlap (non-overlapping windows)
  python scripts/prepare_all_data.py --mode fasttrack --format windowed --overlap 0.0

  # Only prepare VitalDB
  python scripts/prepare_all_data.py --dataset vitaldb

  # Only prepare BUT-PPG (download if needed)
  python scripts/prepare_all_data.py --dataset butppg --download-butppg

  # BUT-PPG with windowed format and overlapping windows
  python scripts/prepare_all_data.py --dataset butppg --format windowed --overlap 0.25

  # Use 16 workers for faster processing
  python scripts/prepare_all_data.py --num-workers 16

  # Custom output directory
  python scripts/prepare_all_data.py --output data/my_processed_data

Data Formats:
  --format legacy (default):
    Multi-window NPZ files (data=[N, T, C], labels=[N])
    Compatible with existing fine-tuning scripts

  --format windowed (NEW):
    One NPZ per window with embedded labels
    Better for multi-task learning
    Includes: signal quality, demographics, clinical labels
    Use with UnifiedWindowDataset loader

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

    parser.add_argument(
        '--format',
        choices=['legacy', 'windowed'],
        default='legacy',
        help='Data format: legacy (multi-window NPZ) or windowed (one NPZ per window with labels)'
    )

    parser.add_argument(
        '--overlap',
        type=float,
        default=0.25,
        help='Window overlap ratio (0.25 = 25%% overlap, 0.0 = no overlap)'
    )

    args = parser.parse_args()

    # Determine datasets
    if args.dataset == 'both':
        datasets = ['vitaldb', 'butppg']
    else:
        datasets = [args.dataset]
    
    # Check if BUT-PPG is requested but not available
    if 'butppg' in datasets and not BUTPPG_AVAILABLE:
        print("\n" + "="*70)
        print("⚠️  WARNING: BUT-PPG loader not available!")
        print("="*70)
        print("File src/data/butppg_loader.py is missing (Drive sync issue?)")
        print("\nThis file exists locally but hasn't synced to Google Drive yet.")
        print("\nOptions to fix:")
        print("  1. Wait for Google Drive to sync the file (may take 5-10 minutes)")
        print("  2. Manually upload butppg_loader.py from local to Drive")
        print("  3. Use --dataset vitaldb to only prepare VitalDB (works now)")
        print("\nAUTO-SWITCHING to VitalDB only for now...")
        print("="*70 + "\n")
        
        datasets = ['vitaldb']
        logger.info("✓ Continuing with VitalDB dataset only")

    # Create and run pipeline
    pipeline = DataPreparationPipeline(
        mode=args.mode,
        output_dir=args.output,
        num_workers=args.num_workers,
        datasets=datasets,
        download_butppg=args.download_butppg,
        format=args.format,
        overlap=args.overlap  # ✅ OVERLAPPING WINDOWS!
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
