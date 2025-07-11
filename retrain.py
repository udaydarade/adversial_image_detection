import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os
from train_bnn import BNN, BinaryLinear

class GarbageDataset(Dataset):
    """Dataset that treats garbage samples as a new class"""
    def __init__(self, original_dataset, garbage_indices, transform=None):
        self.dataset = original_dataset
        self.garbage_indices = set(garbage_indices)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # If this is a garbage sample, assign it to class 10
        if idx in self.garbage_indices:
            label = 10
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ImprovedBNN(nn.Module):
    """BNN with 11 classes (10 digits + 1 garbage)"""
    def __init__(self, num_blocks=1, neurons_per_block=100, num_classes=11):
        super(ImprovedBNN, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(784, neurons_per_block))
        layers.append(nn.BatchNorm1d(neurons_per_block))
        layers.append(nn.Hardtanh())
        
        # Hidden blocks
        for i in range(num_blocks - 1):
            layers.append(BinaryLinear(neurons_per_block, neurons_per_block))
            layers.append(nn.BatchNorm1d(neurons_per_block))
            layers.append(nn.Hardtanh())
        
        # Output layer (11 classes now)
        layers.append(BinaryLinear(neurons_per_block, num_classes))
        layers.append(nn.BatchNorm1d(num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)

def retrain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load garbage indices
    garbage_indices = np.load('garbage_samples/indices.npy')
    print(f"Loaded {len(garbage_indices)} garbage sample indices")
    
    # Load original datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    original_train = datasets.MNIST('./data', train=True, download=True)
    original_test = datasets.MNIST('./data', train=False, download=True)
    
    # Create datasets with garbage labels
    train_dataset = GarbageDataset(original_train, garbage_indices, transform)
    test_dataset = GarbageDataset(original_test, [], transform)  # No garbage in test
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create improved model with 11 classes
    model = ImprovedBNN(num_blocks=1, neurons_per_block=100, num_classes=11).to(device)
    
    # Load weights from original model (except last layer)
    original_checkpoint = torch.load('bnn_checkpoint.pth', map_location=device)
    original_state = original_checkpoint['model_state_dict']
    
    # Copy weights except for the last layer
    model_state = model.state_dict()
    for name, param in original_state.items():
        if 'model.6' not in name and 'model.7' not in name:  # Skip last linear and batchnorm
            if name in model_state and param.shape == model_state[name].shape:
                model_state[name] = param
    
    model.load_state_dict(model_state)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training with garbage class
    print("\nRetraining with garbage class...")
    best_accuracy = 0
    
    for epoch in range(10):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        garbage_correct = 0
        garbage_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Track garbage class performance
            garbage_mask = target == 10
            if garbage_mask.any():
                garbage_total += garbage_mask.sum().item()
                garbage_correct += pred[garbage_mask].eq(target[garbage_mask].view_as(pred[garbage_mask])).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100. * correct / len(train_loader.dataset)
        garbage_acc = 100. * garbage_correct / garbage_total if garbage_total > 0 else 0
        
        # Test (on original 10 classes)
        model.eval()
        test_loss = 0
        correct = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Only consider first 10 classes for test accuracy
                test_loss += criterion(output[:, :10], target).item()
                pred = output[:, :10].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Per-class accuracy
                for t, p in zip(target.view(-1), pred.view(-1)):
                    class_total[t.item()] += 1
                    class_correct[t.item()] += (t == p).item()
        
        test_acc = 100. * correct / len(test_loader.dataset)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'  Train Acc: {train_acc:.2f}% (Garbage: {garbage_acc:.2f}%)')
        print(f'  Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_accuracy': test_acc,
                'train_accuracy': train_acc,
                'garbage_accuracy': garbage_acc,
                'garbage_indices': garbage_indices.tolist(),
                'num_classes': 11
            }, 'improved_bnn.pth')
    
    print(f"\nRetraining complete!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Model saved as improved_bnn.pth")
    
    # Create final report
    report = f"""
Retraining Report
=================
Original model accuracy: {original_checkpoint['accuracy']:.2f}%
Improved model accuracy: {best_accuracy:.2f}%
Improvement: {best_accuracy - original_checkpoint['accuracy']:.2f}%

Garbage samples: {len(garbage_indices)}
Garbage detection accuracy: {garbage_acc:.2f}%

The model now has 11 classes:
- Classes 0-9: Original MNIST digits
- Class 10: Garbage samples (weak boundaries)
"""
    
    with open('retraining_report.txt', 'w') as f:
        f.write(report)
    
    print(report)

if __name__ == '__main__':
    retrain()