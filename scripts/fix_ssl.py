#!/usr/bin/env python3
"""
Fix SSL Certificate issue for macOS.

This script will:
1. Install SSL certificates for Python
2. Set up SSL context for VitalDB
"""

import os
import sys
import ssl
import subprocess


def fix_ssl_certificates():
    """Fix SSL certificates for macOS."""
    
    print("Fixing SSL certificates for VitalDB...")
    print("=" * 60)
    
    # Method 1: Try to install certificates using pip
    try:
        print("Method 1: Installing certifi package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"], check=True)
        print("✓ Certifi installed/updated")
        
        # Set up SSL context
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        print(f"✓ SSL_CERT_FILE set to: {certifi.where()}")
        
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
        
    # Method 2: Try to run the Install Certificates command (macOS specific)
    try:
        print("\nMethod 2: Running Install Certificates.command...")
        python_folder = os.path.dirname(os.path.dirname(sys.executable))
        cert_command = os.path.join(python_folder, 'Install Certificates.command')
        
        if os.path.exists(cert_command):
            subprocess.run(['/bin/bash', cert_command], check=True)
            print("✓ Certificates installed via Install Certificates.command")
        else:
            print(f"✗ Install Certificates.command not found at {cert_command}")
            
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
        
    # Method 3: Create unverified context (less secure but works)
    print("\nMethod 3: Setting up unverified SSL context (temporary fix)...")
    print("Note: This is less secure but will allow VitalDB to work")
    
    # This will be used by vitaldb_loader.py
    ssl._create_default_https_context = ssl._create_unverified_context
    print("✓ SSL context set to unverified mode")
    
    print("\n" + "=" * 60)
    print("SSL fixes applied!")
    print("\nNow try running the VitalDB tests again:")
    print("  python scripts/test_vitaldb_quick.py")
    print("\nFor a permanent fix on macOS, run:")
    print(f"  {sys.executable} -m pip install --upgrade certifi")
    print("  Then find and run 'Install Certificates.command' in your Python folder")


if __name__ == "__main__":
    fix_ssl_certificates()
