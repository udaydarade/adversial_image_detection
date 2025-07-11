#!/usr/bin/env python3

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print(f"✓ Success ({elapsed:.1f}s)")
        if result.stdout:
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed ({elapsed:.1f}s)")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check Python 3
    try:
        subprocess.run([sys.executable, '--version'], check=True)
        print(f"✓ Python 3: {sys.executable}")
    except:
        print("✗ Python 3 not found")
        return False
    
    # Check Python 2.7
    try:
        subprocess.run(['python2.7', '--version'], check=True, capture_output=True)
        print("✓ Python 2.7 found")
    except:
        print("✗ Python 2.7 not found (needed for NPAQ)")
        print("  Install with: sudo apt-get install python2.7")
        return False
    
    # Check if venv exists
    if os.path.exists('venv'):
        print("✓ Virtual environment exists")
    else:
        print("✗ Virtual environment not found")
        print("  Run: make setup")
        return False
    
    # Check if NPAQ exists
    if os.path.exists('npaq'):
        print("✓ NPAQ repository found")
    else:
        print("✗ NPAQ not found")
        print("  Run: git clone https://github.com/Shivvrat/npaq.git")
        return False
    
    return True

def main():
    """Run the complete pipeline"""
    print("Adversarial Data Detection Pipeline")
    print("===================================")
    
    if not check_requirements():
        print("\nPlease fix the requirements and try again.")
        sys.exit(1)
    
    # Pipeline steps
    steps = [
        (['venv/bin/python', 'train_bnn.py'], "Training BNN model"),
        (['python2.7', 'run_npaq.py'], "Generating CNF with NPAQ"),
        (['venv/bin/python', 'find_garbage.py'], "Detecting garbage samples"),
        (['venv/bin/python', 'retrain.py'], "Retraining with garbage labels")
    ]
    
    success = True
    for cmd, description in steps:
        if not run_command(cmd, description):
            success = False
            print(f"\nPipeline failed at: {description}")
            break
    
    if success:
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        print("\nResults:")
        print("- Trained BNN model: bnn_checkpoint.pth")
        print("- CNF representation: *.cnf")
        print("- Garbage samples: garbage_samples/")
        print("- Improved model: improved_bnn.pth")
        print("\nCheck retraining_report.txt for final results")
    else:
        print("\nPipeline failed. Check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()