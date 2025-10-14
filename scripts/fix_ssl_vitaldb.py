#!/usr/bin/env python3
"""
Fix SSL certificates for VitalDB real data access.
Run this BEFORE trying to load real VitalDB data.
"""

import os
import sys
import ssl
import subprocess
from pathlib import Path


def fix_ssl_for_vitaldb():
    """Fix SSL certificates specifically for VitalDB."""
    
    print("=" * 60)
    print(" Fixing SSL for Real VitalDB Data Access")
    print("=" * 60)
    
    success = False
    
    # Method 1: Install/upgrade certifi
    print("\n1. Installing/upgrading certifi package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"], 
                      check=True, capture_output=True, text=True)
        print("✓ Certifi installed/upgraded")
        
        # Set environment variables
        import certifi
        cert_path = certifi.where()
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        print(f"✓ SSL certificates set to: {cert_path}")
        success = True
        
    except Exception as e:
        print(f"✗ Certifi installation failed: {e}")
        
    # Method 2: macOS specific - Install Certificates.command
    if sys.platform == 'darwin':  # macOS
        print("\n2. Checking for macOS Install Certificates.command...")
        
        # Common Python installation paths on macOS
        possible_paths = [
            Path(sys.prefix) / 'Install Certificates.command',
            Path('/Applications/Python 3.10/Install Certificates.command'),
            Path('/Applications/Python 3.11/Install Certificates.command'),
            Path('/Applications/Python 3.12/Install Certificates.command'),
            Path.home() / 'Library/Python/3.10/Install Certificates.command',
        ]
        
        cert_command = None
        for path in possible_paths:
            if path.exists():
                cert_command = path
                break
                
        if cert_command:
            print(f"✓ Found: {cert_command}")
            try:
                subprocess.run(['/bin/bash', str(cert_command)], check=True)
                print("✓ Certificates installed via Install Certificates.command")
                success = True
            except Exception as e:
                print(f"✗ Failed to run Install Certificates.command: {e}")
        else:
            print("✗ Install Certificates.command not found")
            print("\nManual fix:")
            print("1. Open Finder")
            print("2. Go to Applications > Python 3.x")
            print("3. Double-click 'Install Certificates.command'")
            
    # Method 3: Create unverified context (less secure but works)
    if not success:
        print("\n3. Setting unverified SSL context (temporary workaround)...")
        print("⚠ Warning: This is less secure but allows VitalDB to work")
        
        try:
            # This will be imported by vitaldb_loader.py
            ssl._create_default_https_context = ssl._create_unverified_context
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            print("✓ SSL verification disabled (temporary)")
            success = True
        except Exception as e:
            print(f"✗ Failed to set unverified context: {e}")
            
    # Test the fix
    print("\n" + "=" * 60)
    print("Testing SSL fix with VitalDB...")
    
    try:
        import vitaldb
        # Try a simple API call
        cases = vitaldb.find_cases('PLETH')
        if len(cases) > 0:
            print(f"✅ SUCCESS! Found {len(cases)} VitalDB cases")
            print("SSL is working correctly for VitalDB!")
            
            # Save SSL fix to config
            config_file = Path.home() / '.vitaldb_ssl_fixed'
            config_file.write_text('SSL_FIXED=1')
            print(f"\n✓ SSL fix saved to {config_file}")
            
        else:
            print("⚠ VitalDB connected but no cases found")
            
    except ImportError:
        print("✗ VitalDB not installed")
        print("Install with: pip install vitaldb")
        
    except Exception as e:
        if 'SSL' in str(e) or 'certificate' in str(e).lower():
            print(f"✗ SSL still not working: {e}")
            print("\nAdditional options:")
            print("1. Try running with: PYTHONHTTPSVERIFY=0 python <your_script.py>")
            print("2. Update Python to latest version")
            print("3. On macOS, update system certificates:")
            print("   brew install ca-certificates")
        else:
            print(f"✗ Other error: {e}")
            
    print("\n" + "=" * 60)
    if success:
        print("✅ SSL fix applied!")
        print("\nNow you can run:")
        print("  python scripts/test_real_vitaldb.py")
        print("  python scripts/run_multimodal_pipeline.py")
    else:
        print("❌ SSL fix incomplete. See manual instructions above.")
        
    return success


if __name__ == "__main__":
    fix_ssl_for_vitaldb()
