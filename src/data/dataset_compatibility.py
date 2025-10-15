"""Dataset Compatibility Validator for Transfer Learning.

Ensures VitalDB and BUT PPG datasets produce compatible outputs
for proper transfer learning.
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


class DatasetCompatibilityValidator:
    """Validate compatibility between VitalDB and BUT PPG datasets."""
    
    def __init__(self, vitaldb_dataset, butppg_dataset):
        """Initialize validator.
        
        Args:
            vitaldb_dataset: VitalDBDataset instance
            butppg_dataset: BUTPPGDataset instance
        """
        self.vitaldb = vitaldb_dataset
        self.butppg = butppg_dataset
        self.checks = {}
    
    def validate_all(self) -> Tuple[bool, Dict]:
        """Run all compatibility checks.
        
        Returns:
            Tuple of (overall_compatible, detailed_checks)
        """
        print("\n" + "="*70)
        print("DATASET COMPATIBILITY VALIDATION")
        print("="*70)
        
        # Run all checks
        self.validate_preprocessing()
        self.validate_output_shapes()
        self.validate_signal_statistics()
        self.validate_qc_compatibility()
        self.validate_batch_compatibility()
        
        # Compute overall compatibility
        overall = all(check['passed'] for check in self.checks.values())
        
        # Print summary
        self._print_summary()
        
        return overall, self.checks
    
    def validate_preprocessing(self):
        """Validate that preprocessing parameters match."""
        print("\n1. Preprocessing Configuration")
        print("-" * 70)
        
        checks = {
            'filter_type': None,
            'filter_order': None,
            'filter_band': None,
            'target_fs': None,
            'window_sec': None,
            'hop_sec': None
        }
        
        # Filter type
        vdb_filter = self.vitaldb.filter_params.get('type', 'unknown')
        but_filter = self.butppg.preprocessing_config['filter_type']
        checks['filter_type'] = vdb_filter == but_filter
        print(f"  Filter type: VitalDB={vdb_filter}, BUT={but_filter} "
              f"{'✓' if checks['filter_type'] else '✗ MISMATCH'}")
        
        # Filter order
        vdb_order = self.vitaldb.filter_params.get('order', -1)
        but_order = self.butppg.preprocessing_config['filter_order']
        checks['filter_order'] = vdb_order == but_order
        print(f"  Filter order: VitalDB={vdb_order}, BUT={but_order} "
              f"{'✓' if checks['filter_order'] else '✗ MISMATCH'}")
        
        # Filter band
        vdb_band = self.vitaldb.filter_params.get('band', [])
        but_band = self.butppg.preprocessing_config['filter_band']
        checks['filter_band'] = vdb_band == but_band
        print(f"  Filter band: VitalDB={vdb_band}, BUT={but_band} "
              f"{'✓' if checks['filter_band'] else '✗ MISMATCH'}")
        
        # Sampling rate
        vdb_fs = self.vitaldb.target_fs
        but_fs = self.butppg.target_fs
        checks['target_fs'] = vdb_fs == but_fs
        print(f"  Target fs: VitalDB={vdb_fs}Hz, BUT={but_fs}Hz "
              f"{'✓' if checks['target_fs'] else '✗ MISMATCH'}")
        
        # Window parameters
        vdb_window = self.vitaldb.window_sec
        but_window = self.butppg.window_sec
        checks['window_sec'] = abs(vdb_window - but_window) < 0.01
        print(f"  Window size: VitalDB={vdb_window}s, BUT={but_window}s "
              f"{'✓' if checks['window_sec'] else '✗ MISMATCH'}")
        
        vdb_hop = self.vitaldb.hop_sec
        but_hop = self.butppg.hop_sec
        checks['hop_sec'] = abs(vdb_hop - but_hop) < 0.01
        print(f"  Hop size: VitalDB={vdb_hop}s, BUT={but_hop}s "
              f"{'✓' if checks['hop_sec'] else '✗ MISMATCH'}")
        
        self.checks['preprocessing'] = {
            'passed': all(checks.values()),
            'details': checks
        }
    
    def validate_output_shapes(self):
        """Validate that output shapes match."""
        print("\n2. Output Shape Compatibility")
        print("-" * 70)
        
        try:
            # Get sample from each dataset
            vdb_sample = self.vitaldb[0]
            but_sample = self.butppg[0]
            
            # Extract signals (handle different return formats)
            vdb_sig1 = vdb_sample[0] if isinstance(vdb_sample, tuple) else vdb_sample
            but_sig1 = but_sample[0] if isinstance(but_sample, tuple) else but_sample
            
            # Check shapes
            shape_match = vdb_sig1.shape == but_sig1.shape
            
            print(f"  VitalDB sample shape: {vdb_sig1.shape}")
            print(f"  BUT PPG sample shape: {but_sig1.shape}")
            print(f"  Shape match: {'✓' if shape_match else '✗ MISMATCH'}")
            
            # Check dtype
            dtype_match = vdb_sig1.dtype == but_sig1.dtype
            print(f"  VitalDB dtype: {vdb_sig1.dtype}")
            print(f"  BUT PPG dtype: {but_sig1.dtype}")
            print(f"  Dtype match: {'✓' if dtype_match else '✗ MISMATCH'}")
            
            self.checks['output_shapes'] = {
                'passed': shape_match and dtype_match,
                'details': {
                    'vitaldb_shape': vdb_sig1.shape,
                    'butppg_shape': but_sig1.shape,
                    'shape_match': shape_match,
                    'dtype_match': dtype_match
                }
            }
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            self.checks['output_shapes'] = {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def validate_signal_statistics(self):
        """Validate that signal statistics are comparable."""
        print("\n3. Signal Statistics")
        print("-" * 70)
        
        try:
            # Sample multiple windows
            n_samples = min(10, len(self.vitaldb), len(self.butppg))
            
            vdb_means = []
            vdb_stds = []
            but_means = []
            but_stds = []
            
            for i in range(n_samples):
                vdb_sample = self.vitaldb[i]
                but_sample = self.butppg[i]
                
                vdb_sig = vdb_sample[0] if isinstance(vdb_sample, tuple) else vdb_sample
                but_sig = but_sample[0] if isinstance(but_sample, tuple) else but_sample
                
                vdb_means.append(vdb_sig.mean().item())
                vdb_stds.append(vdb_sig.std().item())
                but_means.append(but_sig.mean().item())
                but_stds.append(but_sig.std().item())
            
            vdb_mean_avg = np.mean(vdb_means)
            vdb_std_avg = np.mean(vdb_stds)
            but_mean_avg = np.mean(but_means)
            but_std_avg = np.mean(but_stds)
            
            print(f"  VitalDB: mean={vdb_mean_avg:.4f}, std={vdb_std_avg:.4f}")
            print(f"  BUT PPG: mean={but_mean_avg:.4f}, std={but_std_avg:.4f}")
            
            # Check if both are normalized (mean ≈ 0, std ≈ 1)
            vdb_normalized = abs(vdb_mean_avg) < 0.1 and abs(vdb_std_avg - 1.0) < 0.2
            but_normalized = abs(but_mean_avg) < 0.1 and abs(but_std_avg - 1.0) < 0.2
            
            both_normalized = vdb_normalized and but_normalized
            print(f"  Both normalized (mean≈0, std≈1): {'✓' if both_normalized else '✗'}")
            
            self.checks['statistics'] = {
                'passed': both_normalized,
                'details': {
                    'vitaldb_mean': vdb_mean_avg,
                    'vitaldb_std': vdb_std_avg,
                    'butppg_mean': but_mean_avg,
                    'butppg_std': but_std_avg,
                    'both_normalized': both_normalized
                }
            }
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            self.checks['statistics'] = {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def validate_qc_compatibility(self):
        """Validate that QC is applied consistently."""
        print("\n4. Quality Control Compatibility")
        print("-" * 70)
        
        checks = {
            'qc_enabled': None,
            'min_valid_ratio': None
        }
        
        # Check QC enabled
        vdb_qc = self.vitaldb.enable_qc
        but_qc = self.butppg.enable_qc
        checks['qc_enabled'] = vdb_qc == but_qc
        print(f"  QC enabled: VitalDB={vdb_qc}, BUT={but_qc} "
              f"{'✓' if checks['qc_enabled'] else '✗ MISMATCH'}")
        
        # Check min valid ratio
        vdb_ratio = self.vitaldb.min_valid_ratio
        but_ratio = self.butppg.min_valid_ratio
        checks['min_valid_ratio'] = abs(vdb_ratio - but_ratio) < 0.01
        print(f"  Min valid ratio: VitalDB={vdb_ratio}, BUT={but_ratio} "
              f"{'✓' if checks['min_valid_ratio'] else '✗ MISMATCH'}")
        
        self.checks['qc'] = {
            'passed': all(checks.values()),
            'details': checks
        }
    
    def validate_batch_compatibility(self):
        """Validate that batches can be processed together."""
        print("\n5. Batch Processing Compatibility")
        print("-" * 70)
        
        try:
            from torch.utils.data import DataLoader
            
            # Create small dataloaders
            vdb_loader = DataLoader(self.vitaldb, batch_size=4, shuffle=False)
            but_loader = DataLoader(self.butppg, batch_size=4, shuffle=False)
            
            # Get one batch from each
            vdb_batch = next(iter(vdb_loader))
            but_batch = next(iter(but_loader))
            
            # Extract tensors
            vdb_tensor = vdb_batch[0] if isinstance(vdb_batch, tuple) else vdb_batch
            but_tensor = but_batch[0] if isinstance(but_batch, tuple) else but_batch
            
            # Check if shapes are compatible for concatenation
            compatible = vdb_tensor.shape[1:] == but_tensor.shape[1:]
            
            print(f"  VitalDB batch: {vdb_tensor.shape}")
            print(f"  BUT PPG batch: {but_tensor.shape}")
            print(f"  Can concatenate: {'✓' if compatible else '✗ INCOMPATIBLE'}")
            
            # Try to create mixed batch
            if compatible:
                try:
                    mixed_batch = torch.cat([vdb_tensor[:2], but_tensor[:2]], dim=0)
                    print(f"  Mixed batch shape: {mixed_batch.shape} ✓")
                    mixed_success = True
                except Exception as e:
                    print(f"  Mixed batch failed: {e} ✗")
                    mixed_success = False
            else:
                mixed_success = False
            
            self.checks['batch'] = {
                'passed': compatible and mixed_success,
                'details': {
                    'shapes_compatible': compatible,
                    'mixed_batch_success': mixed_success
                }
            }
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            self.checks['batch'] = {
                'passed': False,
                'details': {'error': str(e)}
            }
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        all_passed = True
        for check_name, check_result in self.checks.items():
            status = "✓ PASS" if check_result['passed'] else "✗ FAIL"
            print(f"  {check_name.upper()}: {status}")
            all_passed = all_passed and check_result['passed']
        
        print("="*70)
        if all_passed:
            print("✓ ALL CHECKS PASSED - Datasets are compatible for transfer learning")
        else:
            print("✗ SOME CHECKS FAILED - Fix issues before training")
        print("="*70 + "\n")
    
    def generate_report(self, output_path: str = None):
        """Generate detailed compatibility report.
        
        Args:
            output_path: Path to save report (optional)
        """
        report_lines = []
        report_lines.append("# Dataset Compatibility Report\n")
        report_lines.append(f"## VitalDB vs BUT PPG Compatibility Analysis\n")
        
        for check_name, check_result in self.checks.items():
            report_lines.append(f"\n### {check_name.upper()}\n")
            report_lines.append(f"**Status:** {'✓ PASS' if check_result['passed'] else '✗ FAIL'}\n")
            report_lines.append("\n**Details:**\n")
            
            for key, value in check_result['details'].items():
                report_lines.append(f"- {key}: {value}\n")
        
        report_text = "".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        
        return report_text


def validate_datasets(vitaldb_dataset, butppg_dataset, output_path: str = None) -> bool:
    """Convenience function to validate dataset compatibility.
    
    Args:
        vitaldb_dataset: VitalDBDataset instance
        butppg_dataset: BUTPPGDataset instance
        output_path: Optional path to save report
        
    Returns:
        True if compatible, False otherwise
    """
    validator = DatasetCompatibilityValidator(vitaldb_dataset, butppg_dataset)
    compatible, checks = validator.validate_all()
    
    if output_path:
        validator.generate_report(output_path)
    
    return compatible
