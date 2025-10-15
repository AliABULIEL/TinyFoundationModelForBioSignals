"""
Quick configuration update script to use TTM-Enhanced pretrained weights
Updates windows.yaml to match TTM-Enhanced dimensions (context=1024, patch=128)
"""

import yaml
from pathlib import Path

def update_windows_config():
    """Update windows.yaml to match TTM-Enhanced configuration"""
    
    config_path = Path("configs/windows.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print("=" * 70)
    print("UPDATING CONFIGURATION FOR TTM-ENHANCED")
    print("=" * 70)
    
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n📋 Current configuration:")
    print(f"  window_sec: {config.get('window_sec', 'Not set')}")
    print(f"  patch_size: {config.get('patch_size', 'Not set')}")
    
    # Backup original
    backup_path = config_path.with_suffix('.yaml.backup')
    with open(backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n✓ Backed up original config to: {backup_path}")
    
    # Update to TTM-Enhanced dimensions
    old_window_sec = config.get('window_sec')
    old_patch_size = config.get('patch_size')
    
    config['window_sec'] = 8.192  # 1024 samples @ 125Hz
    config['patch_size'] = 128     # 1.024 seconds per patch
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("\n✓ Updated configuration:")
    print(f"  window_sec: {old_window_sec} → 8.192 (1024 samples @ 125Hz)")
    print(f"  patch_size: {old_patch_size} → 128 (1.024s patches)")
    
    print("\n" + "=" * 70)
    print("✅ CONFIGURATION UPDATED FOR TTM-ENHANCED!")
    print("=" * 70)
    
    print("\n📝 Next steps:")
    print("  1. Rebuild VitalDB windows:")
    print("     python3 -m scripts.prepare_all_data --mode fasttrack --dataset vitaldb")
    print()
    print("  2. Run SSL pretraining (will use IBM's pretrained TTM-Enhanced!):")
    print("     python3 scripts/pretrain_vitaldb_ssl.py --mode fasttrack --epochs 10")
    
    print("\n💡 What changed:")
    print("  • Window duration: 10.0s → 8.192s")
    print("  • Samples per window: 1250 → 1024")  
    print("  • Patch size: 125 → 128 samples")
    print("  • Number of patches: 10 → 8")
    print()
    print("  ✅ Now matches IBM's TTM-Enhanced pretrained model!")
    print("  ✅ Will load 4M parameter foundation model")
    print("  ✅ Domain adaptation: General time series → Biosignals")
    
    return True

if __name__ == "__main__":
    import sys
    success = update_windows_config()
    sys.exit(0 if success else 1)
