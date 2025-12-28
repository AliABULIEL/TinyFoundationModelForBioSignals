import sys
import importlib.util

# Check if vitaldb is installed
spec = importlib.util.find_spec('vitaldb')
if spec is None:
    print("❌ VitalDB package is NOT installed")
    print("\nInstalling VitalDB...")
else:
    print("✅ VitalDB package is installed")

    # Try to import and check
    try:
        import vitaldb
        print(f"✅ VitalDB imported successfully")
        print(f"   Version: {vitaldb.__version__ if hasattr(vitaldb, '__version__') else 'Unknown'}")

        # Check available case sets
        if hasattr(vitaldb, 'caseids_bis'):
            print(f"   BIS cases available: {len(vitaldb.caseids_bis)}")
        else:
            print("   ⚠️ BIS cases not found")

    except Exception as e:
        print(f"❌ Error importing VitalDB: {e}")