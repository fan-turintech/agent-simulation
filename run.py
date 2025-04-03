#!/usr/bin/env python3
"""
Quick launcher for the Darwinian Evolution Simulation
"""

import os
import sys
import subprocess
import importlib

def check_and_install_requirements():
    """Check if all required packages are installed, and install them if needed"""
    requirements = ['pygame', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in requirements:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Attempting to install the missing packages...")
        
        try:
            # Use pip to install the missing packages
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Successfully installed all required packages.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install packages using pip.")
            print("Please manually install the required packages with:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main entry point for the simulation launcher"""
    print("Darwinian Evolution Simulation")
    print("------------------------------")
    
    if not check_and_install_requirements():
        return 1
    
    print("Starting simulation...")
    
    # Import and run the main simulation
    from main import main
    main()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
