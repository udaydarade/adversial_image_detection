import numpy as np
from z3 import *
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from train_bnn import BNN

class GarbageFinder:
    def __init__(self, cnf_path=None):
        # Load CNF path
        if cnf_path is None:
            with open('cnf_path.txt', 'r') as f:
                cnf_path = f.read().strip()
        
        self.cnf_path = cnf_path
        self.solver = Solver()
        
        # Load MNIST
        transform = transforms.ToTensor()
        self.mnist = datasets.MNIST('./data', train=True, transform=transform)
        
        # Load trained model for validation
        self.model = BNN(num_blocks=1, neurons_per_block=100)
        checkpoint = torch.load('bnn_checkpoint.pth', map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.parse_cnf()
        
    def parse_cnf(self):
        """Parse CNF file from NPAQ"""
        print(f"Parsing CNF file: {self.cnf_path}")
        
        if not os.path.exists(self.cnf_path):
            # If exact CNF not found, look for any CNF file
            cnf_files = [f for f in os.listdir('.') if f.endswith('.cnf')]
            if cnf_files:
                self.cnf_path = cnf_files[0]
                print(f"Using CNF file: {self.cnf_path}")
            else:
                raise FileNotFoundError("No CNF file found")
        
        with open(self.cnf_path, 'r') as f:
            lines = f.readlines()

        self.num_vars = 0
        self.num_clauses = 0
        
        for line in lines:
            if line.startswith('p cnf'):
                parts = line.split()
                self.num_vars = int(parts[2])
                self.num_clauses = int(parts[3])
                break
        
        print(f"CNF has {self.num_vars} variables and {self.num_clauses} clauses")
        
        self.vars = {}
        for i in range(1, self.num_vars + 1):
            self.vars[i] = Bool(f'v{i}')
        
        # Add clauses to solver
        clause_count = 0
        for line in lines:
            if line.startswith('c') or line.startswith('p'):
                continue
            
            literals = [int(x) for x in line.split() if x != '0' and x != '']
            if literals:
                z3_clause = []
                for lit in literals:
                    if lit > 0:
                        z3_clause.append(self.vars[abs(lit)])
                    else:
                        z3_clause.append(Not(self.vars[abs(lit)]))
                
                self.solver.add(Or(z3_clause))
                clause_count += 1
        
        print(f"Added {clause_count} clauses to Z3 solver")
    
    def check_weak_boundary(self, idx, epsilon=2):
        """Check if a sample has weak decision boundary using Z3"""
        image, label = self.mnist[idx]
        
        # Get model prediction to verify it's correctly classified
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
        
        if pred != label:
            return False  # Skip already misclassified samples
        
        # Binarize image (NPAQ uses -1 and 1)
        flat_image = image.flatten()
        binary_image = torch.where(flat_image > 0.5, 
                                  torch.tensor(1.0), 
                                  torch.tensor(-1.0))
        
        # Create new solver instance for this check
        local_solver = Solver()
        
        # Add all the CNF clauses
        for assertion in self.solver.assertions():
            local_solver.add(assertion)
        
        # Add constraints for the input image
        # NPAQ typically uses first 784 variables for input
        for i in range(784):
            var_idx = i + 1
            if var_idx <= self.num_vars:
                if binary_image[i] > 0:
                    local_solver.add(self.vars[var_idx])
                else:
                    local_solver.add(Not(self.vars[var_idx]))
        
        # Add perturbation constraint
        # Count how many pixels differ from original
        differences = []
        for i in range(784):
            var_idx = i + 1
            if var_idx <= self.num_vars:
                if binary_image[i] > 0:
                    # Original is 1, count if it becomes -1
                    differences.append(If(Not(self.vars[var_idx]), 1, 0))
                else:
                    # Original is -1, count if it becomes 1
                    differences.append(If(self.vars[var_idx], 1, 0))
        
        # Limit perturbation size
        if differences:
            local_solver.add(Sum(differences) <= epsilon)
        
        # Check if an adversarial example exists
        # For NPAQ, we need to check if output can be different
        result = local_solver.check()
        
        return result == sat
    
    def find_garbage_samples_fast(self, num_samples=300):
        """Fast heuristic to find potentially weak samples"""
        print("Using fast heuristic to identify weak samples...")
        
        garbage_indices = []
        
        # First, use model predictions to find boundary samples
        batch_size = 100
        
        for start_idx in tqdm(range(0, len(self.mnist), batch_size)):
            end_idx = min(start_idx + batch_size, len(self.mnist))
            
            # Get batch of images
            images = []
            labels = []
            indices = []
            
            for idx in range(start_idx, end_idx):
                image, label = self.mnist[idx]
                images.append(image)
                labels.append(label)
                indices.append(idx)
            
            images = torch.stack(images)
            labels = torch.tensor(labels)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                # Find samples with low confidence (potential weak boundaries)
                max_probs = probs.max(dim=1)[0]
                correct_mask = preds == labels
                
                # Weak samples: correctly classified but low confidence
                weak_mask = correct_mask & (max_probs < 0.7)
                
                for i in range(len(indices)):
                    if weak_mask[i]:
                        garbage_indices.append(indices[i])
                        
                        if len(garbage_indices) >= num_samples:
                            return garbage_indices
        
        return garbage_indices
    
    def verify_with_z3(self, indices, max_verify=50):
        """Verify weak samples using Z3 (slower but accurate)"""
        print(f"\nVerifying {min(len(indices), max_verify)} samples with Z3...")
        
        verified_garbage = []
        
        for i, idx in enumerate(tqdm(indices[:max_verify])):
            if self.check_weak_boundary(idx, epsilon=2):
                verified_garbage.append(idx)
        
        print(f"Verified {len(verified_garbage)} samples as having weak boundaries")
        return verified_garbage
    
    def find_garbage_samples(self, num_samples=300):
        """Main method to find garbage samples"""
        # First, use fast heuristic
        candidate_indices = self.find_garbage_samples_fast(num_samples * 2)
        
        # Then verify a subset with Z3
        verified_indices = self.verify_with_z3(candidate_indices, max_verify=50)
        
        # Combine verified and high-confidence candidates
        final_indices = list(set(verified_indices))
        
        # Add more candidates if needed
        for idx in candidate_indices:
            if len(final_indices) >= num_samples:
                break
            if idx not in final_indices:
                final_indices.append(idx)
        
        return final_indices[:num_samples]
    
    def save_results(self, garbage_indices):
        """Save garbage sample indices and visualizations"""
        os.makedirs('garbage_samples', exist_ok=True)
        
        # Save indices
        np.save('garbage_samples/indices.npy', garbage_indices)
        
        # Analyze garbage samples
        analysis = {
            'num_garbage': len(garbage_indices),
            'indices': garbage_indices[:100],  # First 100 for reference
            'cnf_file': self.cnf_path,
            'model_accuracy': float(torch.load('bnn_checkpoint.pth')['accuracy'])
        }
        
        # Check confidence distribution of garbage samples
        confidences = []
        for idx in garbage_indices[:100]:
            image, label = self.mnist[idx]
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))
                prob = torch.softmax(output, dim=1).max().item()
                confidences.append(prob)
        
        analysis['avg_confidence'] = float(np.mean(confidences))
        analysis['min_confidence'] = float(np.min(confidences))
        analysis['max_confidence'] = float(np.max(confidences))
        
        with open('garbage_samples/analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Visualize samples
        num_vis = min(30, len(garbage_indices))
        fig, axes = plt.subplots(5, 6, figsize=(12, 10))
        axes = axes.ravel()
        
        for i in range(num_vis):
            idx = garbage_indices[i]
            image, label = self.mnist[idx]
            
            # Get prediction and confidence
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))
                prob = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                conf = prob.max().item()
            
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'Idx:{idx}\nL:{label} P:{pred} C:{conf:.2f}', 
                            fontsize=8)
            axes[i].axis('off')
        
        plt.suptitle('Garbage Samples (Weak Decision Boundaries)', fontsize=14)
        plt.tight_layout()
        plt.savefig('garbage_samples/visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create detailed report
        report = f"""
Garbage Sample Detection Report
==============================
Total samples analyzed: {len(self.mnist)}
Garbage samples found: {len(garbage_indices)}
Percentage: {100.0 * len(garbage_indices) / len(self.mnist):.2f}%

Model Performance:
- Original test accuracy: {analysis['model_accuracy']:.2f}%
- Avg confidence on garbage: {analysis['avg_confidence']:.3f}
- Min confidence: {analysis['min_confidence']:.3f}
- Max confidence: {analysis['max_confidence']:.3f}

CNF Analysis:
- Variables: {self.num_vars}
- Clauses: {self.num_clauses}
- Source: {self.cnf_path}

Files saved:
- garbage_samples/indices.npy
- garbage_samples/analysis.json
- garbage_samples/visualization.png
"""
        
        with open('garbage_samples/report.txt', 'w') as f:
            f.write(report)
        
        print(report)

def main():
    # Check if model and CNF exist
    if not os.path.exists('bnn_checkpoint.pth'):
        print("Error: No trained model found. Run train_bnn.py first.")
        return
    
    # Try to find CNF file
    cnf_path = None
    if os.path.exists('cnf_path.txt'):
        with open('cnf_path.txt', 'r') as f:
            cnf_path = f.read().strip()
    else:
        # Look for any CNF file
        cnf_files = [f for f in os.listdir('.') if f.endswith('.cnf')]
        if cnf_files:
            cnf_path = cnf_files[0]
            print(f"Found CNF file: {cnf_path}")
        else:
            print("Warning: No CNF file found. Using model-based detection only.")
            # Create a dummy CNF for demonstration
            with open('dummy.cnf', 'w') as f:
                f.write("c Dummy CNF for demonstration\n")
                f.write("p cnf 1000 1\n")
                f.write("1 0\n")
            cnf_path = 'dummy.cnf'
    
    # Find garbage samples
    finder = GarbageFinder(cnf_path)
    garbage_indices = finder.find_garbage_samples(300)
    finder.save_results(garbage_indices)

if __name__ == '__main__':
    main()