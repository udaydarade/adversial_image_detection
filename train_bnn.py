import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

# BNN components based on BinaryNet.pytorch
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0
        return grad_input

class BinaryLinear(nn.Linear):
    def forward(self, input):
        binary_weight = BinaryActivation.apply(self.weight)
        if self.bias is None:
            return nn.functional.linear(input, binary_weight)
        else:
            return nn.functional.linear(input, binary_weight, self.bias)

# BNN model following NPAQ's expected format
class BNN(nn.Module):
    def __init__(self, input_size=784, num_blocks=1, neurons_per_block=100, num_classes=10):
        super(BNN, self).__init__()
        
        self.input_size = input_size
        self.num_blocks = num_blocks
        self.neurons_per_block = neurons_per_block
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, neurons_per_block))
        layers.append(nn.BatchNorm1d(neurons_per_block))
        layers.append(nn.Hardtanh())
        
        # Hidden blocks
        for i in range(num_blocks - 1):
            layers.append(BinaryLinear(neurons_per_block, neurons_per_block))
            layers.append(nn.BatchNorm1d(neurons_per_block))
            layers.append(nn.Hardtanh())
        
        # Output layer
        layers.append(BinaryLinear(neurons_per_block, num_classes))
        layers.append(nn.BatchNorm1d(num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.model(x)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model - using NPAQ naming convention
    num_blocks = 1
    neurons_per_block = 100
    model = BNN(input_size=784, num_blocks=num_blocks, 
                neurons_per_block=neurons_per_block).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("Training BNN...")
    for epoch in range(10):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Test
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save model in NPAQ expected format
    os.makedirs('models/mnist', exist_ok=True)
    
    # NPAQ expects filename: bnn_<input_size>_<num_blocks>blks_<neurons>_<output>.pt
    if num_blocks == 1:
        filename = f'bnn_784_1blk_100.pt'
    else:
        filename = f'bnn_784_{num_blocks}blks_{neurons_per_block}_10.pt'
    
    filepath = os.path.join('models/mnist', filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved as {filepath}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_blocks': num_blocks,
        'neurons_per_block': neurons_per_block,
        'accuracy': accuracy
    }, 'bnn_checkpoint.pth')

if __name__ == '__main__':
    train()