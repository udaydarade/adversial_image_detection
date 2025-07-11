import subprocess
import os
import sys

def setup_npaq():
    """Check if NPAQ is installed and set up"""
    print("Checking NPAQ installation...")
    
    # Check if npaq directory exists
    if not os.path.exists('npaq'):
        print("NPAQ not found. Please clone from: https://github.com/Shivvrat/npaq")
        print("Run: git clone https://github.com/Shivvrat/npaq.git")
        sys.exit(1)
    
    # Check if mlp2cnf is compiled
    if not os.path.exists('npaq/mlp2cnf/mlp2cnf'):
        print("Compiling mlp2cnf...")
        subprocess.run(['make'], cwd='npaq/mlp2cnf', check=True)

def run_npaq_encode():
    """Run NPAQ to encode BNN to CNF"""
    setup_npaq()
    
    print("\nRunning NPAQ to encode BNN to CNF...")
    
    # NPAQ command to encode 1blk_100 architecture
    cmd = [
        'python2.7', 'npaq/npaq', 'bnn',
        '--arch', '1blk_100',
        '--dataset', 'mnist',
        '--resize', '28,28',
        'encode'
    ]
    
    try:
        # Run NPAQ
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("NPAQ encoding successful!")
            print("CNF file should be generated")
            
            # Look for generated CNF file
            cnf_files = [f for f in os.listdir('.') if f.endswith('.cnf')]
            if cnf_files:
                print(f"Found CNF file: {cnf_files[0]}")
                return cnf_files[0]
        else:
            print("NPAQ encoding failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error running NPAQ: {e}")
        
    return None

def run_npaq_robustness():
    """Run NPAQ to check robustness and identify weak samples"""
    print("\nRunning NPAQ robustness analysis...")
    
    # This will help identify samples with weak boundaries
    perturbation_sizes = [1, 2, 3]  # L1 distances
    
    results = {}
    
    for perturb in perturbation_sizes:
        cmd = [
            'python2.7', 'npaq/npaq', 'bnn',
            '--arch', '1blk_100',
            '--dataset', 'mnist',
            '--resize', '28,28',
            'quant-robust', str(perturb),
            '--just-encode'  # Just encode, don't run ApproxMC
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Generated robustness CNF for perturbation={perturb}")
                results[perturb] = True
            else:
                print(f"Failed for perturbation={perturb}")
                results[perturb] = False
        except Exception as e:
            print(f"Error: {e}")
            results[perturb] = False
    
    return results

def main():
    # First encode the model
    cnf_file = run_npaq_encode()
    
    if cnf_file:
        print(f"\nCNF file generated: {cnf_file}")
        
        # Also run robustness analysis
        robustness_results = run_npaq_robustness()
        print("\nRobustness analysis complete")
        
        # Save the CNF file path for the next step
        with open('cnf_path.txt', 'w') as f:
            f.write(cnf_file)
    else:
        print("Failed to generate CNF file")

if __name__ == '__main__':
    main()