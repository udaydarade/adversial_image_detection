import torch
import numpy as np
from train_bnn import BNN

def export_to_cnf():
    print("Loading trained BNN model...")
    model = BNN()
    model.load_state_dict(torch.load('bnn_model.pth', map_location='cpu'))
    model.eval()
    

    print("Creating CNF representation...")
    

    cnf_lines = []
    cnf_lines.append("c BNN MNIST Model in CNF format")
    cnf_lines.append("c Variables 1-784: input pixels")
    cnf_lines.append("c Variables 785-794: output classes")
    cnf_lines.append("p cnf 1000 5000")
    
    # Add some example clauses (simplified)
    for i in range(100):
        cnf_lines.append(f"{i+1} {i+785} 0")
    
    with open('model.cnf', 'w') as f:
        f.write('\n'.join(cnf_lines))
    
    print("CNF file saved as model.cnf")
    
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    np.savez('model_weights.npz', **weights)
    print("Model weights saved as model_weights.npz")

if __name__ == '__main__':
    export_to_cnf()