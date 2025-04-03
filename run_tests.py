#!/usr/bin/env python3
"""
Run tests for Darwinian Simulation
"""
import sys
import subprocess
import os

def check_and_install_test_requirements():
    """Check and install test requirements"""
    test_requirements_file = os.path.join('tests', 'requirements_test.txt')
    if os.path.exists(test_requirements_file):
        print("Installing test requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", test_requirements_file])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install test requirements.")
            return False
    else:
        print(f"Test requirements file not found: {test_requirements_file}")
        return False

def run_tests():
    """Run pytest"""
    print("Running tests...")
    result = subprocess.call([sys.executable, "-m", "pytest", "-v", "tests"])
    return result

if __name__ == "__main__":
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for test requirements
    if not check_and_install_test_requirements():
        print("Skipping requirement installation, attempting to run tests anyway...")
    
    # Run tests
    sys.exit(run_tests())
